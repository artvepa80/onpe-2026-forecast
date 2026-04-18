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
    df["n_pend_est"] = (df["actas_pendientes"] * df["votos_por_acta"]).round().astype(int)

    # Covariables INEI transformadas + estandarizadas (z-score) → coefs interpretables
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


def posterior_final_votes(idata, df, candidato, swing_sd_logit: float = 0.08):
    """Genera vector (draws,) de votos finales nacionales del candidato.

    Ingredientes de incertidumbre:
      1. Posterior de p_d por distrito (modelo jerárquico)
      2. BetaBinomial predictive (overdispersion extra-binomial)
      3. **National swing** por draw: N(0, swing_sd_logit) en escala logit,
         aplicado a las actas pendientes. Captura la incertidumbre de
         "las actas que faltan se comportan distinto a las contadas"
         (sesgo de llegada: rurales llegan tarde, extranjero atrasado).
         swing_sd_logit=0.08 ≈ ±1.5 pp de swing nacional en el %.
    """
    p_draws = idata.posterior["p"].stack(sample=("chain", "draw")).values         # (dist, draws)
    kappa_draws = idata.posterior["kappa"].stack(sample=("chain", "draw")).values   # (draws,)
    n_pend = df["n_pend_est"].values[:, None]                                     # (dist, 1)

    rng = np.random.default_rng(12345)
    n_draws = p_draws.shape[1]

    # National swing en escala logit, distinto en cada draw
    swing = rng.normal(0.0, swing_sd_logit, size=n_draws)                          # (draws,)
    logit_p = np.log(np.clip(p_draws, 1e-6, 1 - 1e-6) / np.clip(1 - p_draws, 1e-6, None))
    p_pend = 1.0 / (1.0 + np.exp(-(logit_p + swing)))                              # (dist, draws)

    # BetaBinomial predictive
    a = p_pend * kappa_draws
    b = (1.0 - p_pend) * kappa_draws
    a = np.clip(a, 1e-3, None); b = np.clip(b, 1e-3, None)
    p_draw_sample = rng.beta(a, b)
    y_pend = rng.binomial(n_pend, p_draw_sample)                                   # (dist, draws)
    y_counted = df[f"votos_{candidato}"].values[:, None]
    total = (y_counted + y_pend).sum(axis=0)                                       # (draws,)
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
