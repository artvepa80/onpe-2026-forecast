#!/usr/bin/env python3
"""
Modelo Bayesiano jerárquico en PyMC — forecast de la pelea Sánchez vs López Aliaga
por el 2° puesto en la primera vuelta 2026.

Modelo (para cada candidato k ∈ {Sánchez, LA}, por distrito d):

    y_{k,d} | n_d ~ Binomial(n_d, p_{k,d})
    logit(p_{k,d}) = α_{k, dep[d]} + β_k · logit_safe(pct_baseline_{k,d}) + ε_{k,d}
    α_{k,dep} ~ Normal(μ_k, σ_k,dep)
    ε_{k,d}   ~ Normal(0, σ_k,dist)

Donde:
  - y_{k,d} = votos del candidato k ya contados en distrito d
  - n_d = total de votos válidos contados en d
  - pct_baseline = % válido 2021 de Renovación Popular (LA) o JpP (proxy Sánchez)

Para las actas pendientes del distrito d:
  - n_pend_d = actas_pendientes · (votos_validos_contados / actas_contab)  (rate ya vista)
  - y_pend_{k,d} ~ Binomial(n_pend_d, p_{k,d})   (muestreo posterior)
  - y_final_{k} = sum_d (y_contados_{k,d} + y_pend_{k,d})
  - P(Sánchez > LA) = mean(y_final_Sanchez > y_final_LA)

Output: forecasts/pymc_<ts>.json con P(Sánchez>LA), CI 95% para cada candidato,
        y un breakdown por departamento.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

ROOT = Path(__file__).resolve().parent.parent
FORECASTS = ROOT / "forecasts"


def logit_safe(p, eps=1e-3):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


ECON_COVARIATES = [
    # (columna, nombre corto, escala recomendada)
    ("altitud_m", "altitud_km", 1 / 1000.0),     # altitud en km (media ~1.5, std ~1.3)
    ("idh_2019",  "idh",        1.0),            # ya en [0,1]
    ("pct_pobreza", "pobreza",  1 / 100.0),       # a proporción
    ("densidad_2020", "log_densidad", "log"),     # log1p por sesgo fuerte
    ("vuln_alim", "vuln_alim", 1.0),              # ya en [0,1]
]


def _scale(series: pd.Series, mode):
    if mode == "log":
        return np.log1p(series.fillna(series.median()))
    return series.fillna(series.median()) * mode


def load_panel(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["pct_RP_2021", "pct_JpP_2021"]).copy()
    df = df[df["votos_validos_contados"] > 0].reset_index(drop=True)
    df["pct_RP_2021"] = df["pct_RP_2021"].clip(lower=0.01) / 100.0
    df["pct_JpP_2021"] = df["pct_JpP_2021"].clip(lower=0.01) / 100.0
    df["votos_por_acta"] = df["votos_validos_contados"] / df["actas_contab"].clip(lower=1)

    # Separar pendientes normales (alta confianza) vs JEE (baja confianza + anulación)
    if "actas_pend_normal" not in df.columns:
        # Panel viejo sin desglose JEE — todo pendiente es "normal"
        df["actas_pend_normal"] = df["actas_pendientes"]
        df["actas_jee"] = 0
    df["n_pend_normal"] = (df["actas_pend_normal"].fillna(0) * df["votos_por_acta"]).round().astype(int)
    df["n_pend_jee"]    = (df["actas_jee"].fillna(0)       * df["votos_por_acta"]).round().astype(int)
    df["n_pend_est"]    = df["n_pend_normal"] + df["n_pend_jee"]  # compat retro

    # Covariables INEI transformadas + estandarizadas (z-score)
    for col, short, scale in ECON_COVARIATES:
        if col not in df.columns:
            df[short + "_z"] = 0.0
            continue
        s = _scale(df[col], scale)
        sd = max(float(s.std(ddof=0)), 1e-6)
        df[short + "_z"] = (s - s.mean()) / sd
    return df


def fit_candidate(df: pd.DataFrame, candidato: str, baseline_col: str,
                  draws: int, tune: int, chains: int, seed: int):
    """BetaBinomial jerárquico non-centered + covariables econométricas estandarizadas.

    logit(p_d) = α_dep[d] + β_2021·logit(baseline_d)
                 + γ1·altitud_z + γ2·idh_z + γ3·pobreza_z
                 + γ4·log_dens_z + γ5·vuln_alim_z
                 + ε_d
    """
    dep_idx, dep_levels = pd.factorize(df["ubigeo_dep"])
    y = df[f"votos_{candidato}"].astype(int).values
    n = df["votos_validos_contados"].astype(int).values
    x = logit_safe(df[baseline_col].values)

    # Matriz de covariables estandarizadas (n_dist × n_cov)
    cov_cols = [short + "_z" for _, short, _ in ECON_COVARIATES]
    Xcov = df[cov_cols].values.astype(float)
    n_cov = Xcov.shape[1]

    with pm.Model() as model:
        mu = pm.Normal("mu", 0.0, 2.5)
        sigma_dep = pm.HalfNormal("sigma_dep", 1.0)
        sigma_dist = pm.HalfNormal("sigma_dist", 0.5)
        beta = pm.Normal("beta", 1.0, 0.5)

        # Non-centered
        alpha_raw = pm.Normal("alpha_raw", 0.0, 1.0, shape=len(dep_levels))
        alpha = pm.Deterministic("alpha", mu + sigma_dep * alpha_raw)
        eps_raw = pm.Normal("eps_raw", 0.0, 1.0, shape=len(df))
        eps = sigma_dist * eps_raw

        # Coeficientes econométricos (una por covariable, prior débil N(0, 0.5))
        gamma = pm.Normal("gamma", 0.0, 0.5, shape=n_cov)

        logit_p = alpha[dep_idx] + beta * x + pm.math.dot(Xcov, gamma) + eps
        p = pm.Deterministic("p", pm.math.sigmoid(logit_p))

        kappa = pm.HalfNormal("kappa", sigma=15.0) + 1.0
        pm.BetaBinomial("y_obs", n=n, alpha=p * kappa, beta=(1 - p) * kappa, observed=y)

        idata = pm.sample(draws=draws, tune=tune, chains=chains,
                          cores=1, nuts_sampler="pymc",
                          random_seed=seed, progressbar=False,
                          target_accept=0.95, init="jitter+adapt_diag")
    return idata, dep_levels, cov_cols


def posterior_final_votes(idata, df, candidato,
                          swing_sd_logit: float = 0.08,
                          jee_extra_sd_logit: float = 0.15,
                          jee_anul_concentration: float = 40.0):
    """Genera vector (draws,) de votos finales nacionales del candidato.

    Modelo de las actas pendientes en DOS bloques:

      (A) Pendientes normales (contadas pronto, distribución estable):
         p_pend_normal = sigmoid(logit(p_d) + swing_nacional)
         y_pend_normal ~ BetaBinomial(n_normal, p·κ, (1-p)·κ)

      (B) Pendientes JEE (observadas — revisión legal, alto riesgo):
         p_pend_jee = sigmoid(logit(p_d) + swing_nacional + swing_jee)   ← ruido extra
         fraccion_anulacion ~ Beta(α=3, β=57)  ~ media 5%, p95 ≈ 12%
         y_pend_jee ~ BetaBinomial(n_jee·(1-frac_anulacion), p_jee·κ, (1-p_jee)·κ)

    jee_extra_sd_logit=0.15 ≈ ±3 pp adicionales en el % — captura incertidumbre
    por concentración geográfica (Lima=26.6% JEE, selva=18% JEE) y resolución
    legal de apoderados.
    """
    p_draws    = idata.posterior["p"].stack(sample=("chain", "draw")).values         # (dist, draws)
    kappa_draws = idata.posterior["kappa"].stack(sample=("chain", "draw")).values     # (draws,)
    n_normal = df["n_pend_normal"].values[:, None]                                    # (dist, 1)
    n_jee    = df["n_pend_jee"].values[:, None]                                       # (dist, 1)

    rng = np.random.default_rng(12345)
    n_draws = p_draws.shape[1]

    # Swings en escala logit
    swing_nacional = rng.normal(0.0, swing_sd_logit, size=n_draws)                    # (draws,)
    swing_jee      = rng.normal(0.0, jee_extra_sd_logit, size=n_draws)                # (draws,)

    logit_p = np.log(np.clip(p_draws, 1e-6, 1 - 1e-6) / np.clip(1 - p_draws, 1e-6, None))
    p_pend_normal = 1.0 / (1.0 + np.exp(-(logit_p + swing_nacional)))                 # (dist, draws)
    p_pend_jee    = 1.0 / (1.0 + np.exp(-(logit_p + swing_nacional + swing_jee)))     # (dist, draws)

    # (A) Pendientes normales
    a = np.clip(p_pend_normal * kappa_draws, 1e-3, None)
    b = np.clip((1 - p_pend_normal) * kappa_draws, 1e-3, None)
    y_normal = rng.binomial(n_normal, rng.beta(a, b))

    # (B) Pendientes JEE con fracción de anulación — POR DISTRITO (taxonomía)
    # Cada distrito tiene su propio prior Beta(α,β) con α/(α+β) = jee_expected_p_anul_d
    # y concentración total = jee_anul_concentration. Districts sin data usan 5% genérico.
    p_anul_prior = df.get("jee_expected_p_anul")
    if p_anul_prior is None:
        p_anul_mean = np.full(len(df), 0.05)
    else:
        p_anul_mean = np.clip(p_anul_prior.fillna(0.05).values, 0.01, 0.80)
    K = jee_anul_concentration
    alpha_d = (p_anul_mean * K).reshape(-1, 1)    # (dist, 1)
    beta_d  = ((1.0 - p_anul_mean) * K).reshape(-1, 1)
    # Broadcast con (1, draws) para vectorizar
    alpha_b = np.broadcast_to(alpha_d, (len(df), n_draws))
    beta_b  = np.broadcast_to(beta_d,  (len(df), n_draws))
    frac_anul = rng.beta(alpha_b, beta_b)                                             # (dist, draws)
    n_jee_efectivo = (n_jee * (1.0 - frac_anul)).round().astype(int)                  # (dist, draws)
    a_j = np.clip(p_pend_jee * kappa_draws, 1e-3, None)
    b_j = np.clip((1 - p_pend_jee) * kappa_draws, 1e-3, None)
    p_j_sample = rng.beta(a_j, b_j)
    y_jee = rng.binomial(n_jee_efectivo, p_j_sample)

    y_counted = df[f"votos_{candidato}"].values[:, None]
    total = (y_counted + y_normal + y_jee).sum(axis=0)                                # (draws,)
    return total


def summarize(votes: np.ndarray, label: str) -> dict:
    q = np.quantile(votes, [0.025, 0.5, 0.975])
    return {"candidato": label, "media": float(votes.mean()), "mediana": float(q[1]),
            "ci_low": float(q[0]), "ci_high": float(q[2])}


def _coef_summary(idata, name: str) -> dict:
    v = idata.posterior[name].stack(sample=("chain", "draw")).values.ravel()
    q = np.quantile(v, [0.025, 0.5, 0.975])
    return {"mediana": float(q[1]), "ci_low": float(q[0]), "ci_high": float(q[2]),
            "sign_cert": float((v > 0).mean() if q[1] > 0 else (v < 0).mean())}


def _jee_taxonomia_summary(df: pd.DataFrame) -> dict:
    """Lee el CSV de metadata JEE y devuelve resumen para Pages."""
    meta_path = Path(__file__).resolve().parent.parent / "data/actas_jee_metadata.csv"
    if not meta_path.exists():
        return {}
    try:
        m = pd.read_csv(meta_path)
        m = m[m["idEleccion"] == 10]  # presidencial
        # Normalizar razones
        def norm(r):
            r = (r or "").lower()
            tags = []
            if "impugnada" in r: tags.append("Acta impugnada")
            if "aritm" in r: tags.append("Acta con error aritmético")
            if "sin firmas" in r: tags.append("Acta sin firmas")
            if "ilegible" in r: tags.append("Acta ilegible")
            if "incompleta" in r: tags.append("Acta incompleta")
            return ", ".join(sorted(set(tags))) or "Otras"
        m["razon_norm"] = m["estadoDescripcionActaResolucion"].apply(norm)
        razones = m["razon_norm"].value_counts().head(10).to_dict()
        # Locales más observadas
        locales = (m.dropna(subset=["nombreLocalVotacion"])
                    .groupby(["ubigeoNivel03","nombreLocalVotacion"])
                    .size().sort_values(ascending=False).head(10))
        locales_list = [{"distrito": d, "local": l, "actas_obs": int(n)}
                        for (d, l), n in locales.items()]
        # Histograma horario
        m["hora"] = m["hora_digitaliz_lima"].str.extract(r" (\d{2}):")[0]
        hist = m["hora"].value_counts().sort_index().to_dict()
        # Flash + nocturnas
        flash_count = int(m["flag_flash_acta"].fillna(False).sum())
        nocturno_count = int(m["flag_hora_anomala"].fillna(False).sum())
        return {
            "razones_top": list(razones.items()),
            "locales_top": locales_list,
            "horas_digitalizacion": {k: int(v) for k, v in hist.items()},
            "flash_lt_30s": flash_count,
            "nocturno_2_5am": nocturno_count,
            "total_scrapeadas": int(len(m)),
        }
    except Exception as e:
        return {"error": str(e)}


def _gamma_summary(idata, cov_cols: list[str]) -> dict:
    v = idata.posterior["gamma"].stack(sample=("chain", "draw")).values  # (cov, draws)
    out = {}
    for i, col in enumerate(cov_cols):
        draws_i = v[i]
        q = np.quantile(draws_i, [0.025, 0.5, 0.975])
        out[col] = {
            "mediana": float(q[1]), "ci_low": float(q[0]), "ci_high": float(q[2]),
            "sign_cert": float((draws_i > 0).mean() if q[1] > 0 else (draws_i < 0).mean()),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=str(ROOT / "data/district_panel.csv"))
    ap.add_argument("--out-dir", default=str(FORECASTS))
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--tune", type=int, default=1500)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = load_panel(Path(args.panel))
    print(f"Panel: {len(df)} distritos con baseline + avance", file=sys.stderr)
    if len(df) < 100:
        print(f"ERROR: panel tiene solo {len(df)} filas (< 100) — abortando antes de samplear.",
              file=sys.stderr)
        sys.exit(2)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%MZ")
    out_path = Path(args.out_dir) / f"pymc_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print("Ajustando Sánchez...", file=sys.stderr)
    idata_s, _, cov_cols = fit_candidate(df, "Sanchez", "pct_JpP_2021",
                                         args.draws, args.tune, args.chains, args.seed)
    print("Ajustando López Aliaga...", file=sys.stderr)
    idata_la, _, _ = fit_candidate(df, "LA", "pct_RP_2021",
                                   args.draws, args.tune, args.chains, args.seed + 1)

    votes_s = posterior_final_votes(idata_s, df, "Sanchez")
    votes_la = posterior_final_votes(idata_la, df, "LA")

    p_sanchez_wins = float((votes_s > votes_la).mean())
    lead = votes_s - votes_la
    lead_q = np.quantile(lead, [0.025, 0.5, 0.975])

    report = {
        "modelo": "pymc_binomial_jerarquico",
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "snapshot_panel": str(Path(args.panel).name),
        "n_distritos_usados": len(df),
        "votos_contados_Sanchez": int(df["votos_Sanchez"].sum()),
        "votos_contados_LA": int(df["votos_LA"].sum()),
        "votos_pendientes_estimados": int(df["n_pend_est"].sum()),
        "votos_pend_normal": int(df["n_pend_normal"].sum()),
        "votos_pend_jee": int(df["n_pend_jee"].sum()),
        "jee_total_nacional": int(df["actas_jee"].sum()) if "actas_jee" in df.columns else 0,
        "jee_por_departamento": (
            df.groupby("departamento")["actas_jee"].sum()
              .sort_values(ascending=False).head(10).to_dict()
            if "actas_jee" in df.columns else {}
        ),
        "jee_taxonomia": _jee_taxonomia_summary(df),
        "prediccion": {
            "Sanchez": summarize(votes_s, "Sanchez"),
            "LA": summarize(votes_la, "LA"),
            "diferencia_Sanchez_minus_LA": {
                "mediana": float(lead_q[1]),
                "ci_low": float(lead_q[0]),
                "ci_high": float(lead_q[2]),
            },
            "P(Sanchez > LA)": p_sanchez_wins,
        },
        "sampler": {
            "draws": args.draws, "tune": args.tune, "chains": args.chains,
            "likelihood": "BetaBinomial (con overdispersion kappa)",
            "rhat_max_sanchez": float(az.summary(idata_s, var_names=["mu","beta","sigma_dep","sigma_dist","kappa","alpha","gamma"])["r_hat"].max()),
            "rhat_max_la": float(az.summary(idata_la, var_names=["mu","beta","sigma_dep","sigma_dist","kappa","alpha","gamma"])["r_hat"].max()),
            "kappa_mean_sanchez": float(idata_s.posterior["kappa"].mean().item()),
            "kappa_mean_la": float(idata_la.posterior["kappa"].mean().item()),
            "tiempo_segundos": round(time.time() - t0, 1),
        },
        "econometria": {
            "covariables": cov_cols,
            "beta_2021": {
                "Sanchez": _coef_summary(idata_s, "beta"),
                "LA": _coef_summary(idata_la, "beta"),
            },
            "gamma": {
                "Sanchez": _gamma_summary(idata_s, cov_cols),
                "LA": _gamma_summary(idata_la, cov_cols),
            },
        },
    }
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    # Además, el "latest" sobrescrito
    (Path(args.out_dir) / "pymc_latest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report["prediccion"], indent=2, ensure_ascii=False))
    print(f"\nGuardado: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
