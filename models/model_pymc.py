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
from datetime import datetime
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


def load_panel(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Filtrar filas útiles: con actas contadas y baseline 2021
    df = df.dropna(subset=["pct_RP_2021", "pct_JpP_2021"]).copy()
    df = df[df["votos_validos_contados"] > 0].reset_index(drop=True)
    df["pct_RP_2021"] = df["pct_RP_2021"].clip(lower=0.01) / 100.0
    df["pct_JpP_2021"] = df["pct_JpP_2021"].clip(lower=0.01) / 100.0
    # Rate de votos válidos por acta (para proyectar pendientes)
    df["votos_por_acta"] = df["votos_validos_contados"] / df["actas_contab"].clip(lower=1)
    df["n_pend_est"] = (df["actas_pendientes"] * df["votos_por_acta"]).round().astype(int)
    return df


def fit_candidate(df: pd.DataFrame, candidato: str, baseline_col: str,
                  draws: int, tune: int, chains: int, seed: int):
    """Ajusta el modelo BetaBinomial jerárquico (con overdispersion) para un candidato."""
    dep_idx, dep_levels = pd.factorize(df["ubigeo_dep"])
    y = df[f"votos_{candidato}"].astype(int).values
    n = df["votos_validos_contados"].astype(int).values
    x = logit_safe(df[baseline_col].values)

    with pm.Model() as model:
        mu = pm.Normal("mu", 0.0, 2.5)
        sigma_dep = pm.HalfNormal("sigma_dep", 1.0)
        sigma_dist = pm.HalfNormal("sigma_dist", 0.5)
        alpha = pm.Normal("alpha", mu, sigma_dep, shape=len(dep_levels))
        beta = pm.Normal("beta", 1.0, 1.0)   # prior centrado en 1 (replica 2021)
        eps = pm.Normal("eps", 0.0, sigma_dist, shape=len(df))

        logit_p = alpha[dep_idx] + beta * x + eps
        p = pm.Deterministic("p", pm.math.sigmoid(logit_p))

        # Overdispersion: kappa chico = más ruido extra-binomial.
        # HalfNormal(50) deja kappa flexible entre ~10 y ~150.
        kappa = pm.HalfNormal("kappa", sigma=50.0)
        pm.BetaBinomial("y_obs", n=n, alpha=p * kappa, beta=(1 - p) * kappa, observed=y)

        idata = pm.sample(draws=draws, tune=tune, chains=chains,
                          random_seed=seed, progressbar=False,
                          target_accept=0.95)
    return idata, dep_levels


def posterior_final_votes(idata, df, candidato):
    """Genera vector (draws,) de votos finales nacionales del candidato.

    Usa BetaBinomial predictive para las actas pendientes — hereda la
    overdispersion del modelo y da CIs honestos.
    """
    p_draws = idata.posterior["p"].stack(sample=("chain", "draw")).values       # (dist, draws)
    kappa_draws = idata.posterior["kappa"].stack(sample=("chain", "draw")).values  # (draws,)
    n_pend = df["n_pend_est"].values[:, None]                                   # (dist, 1)

    rng = np.random.default_rng(12345)
    # BetaBinomial ~ Binomial(n, Beta(p*kappa, (1-p)*kappa))
    a = p_draws * kappa_draws
    b = (1.0 - p_draws) * kappa_draws
    # Nos protegemos contra α/β ≤ 0 por números
    a = np.clip(a, 1e-3, None); b = np.clip(b, 1e-3, None)
    p_draw = rng.beta(a, b)
    y_pend = rng.binomial(n_pend, p_draw)                                       # (dist, draws)
    y_counted = df[f"votos_{candidato}"].values[:, None]
    total = (y_counted + y_pend).sum(axis=0)                                    # (draws,)
    return total


def summarize(votes: np.ndarray, label: str) -> dict:
    q = np.quantile(votes, [0.025, 0.5, 0.975])
    return {"candidato": label, "media": float(votes.mean()), "mediana": float(q[1]),
            "ci_low": float(q[0]), "ci_high": float(q[2])}


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

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = Path(args.out_dir) / f"pymc_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print("Ajustando Sánchez...", file=sys.stderr)
    idata_s, _ = fit_candidate(df, "Sanchez", "pct_JpP_2021",
                               args.draws, args.tune, args.chains, args.seed)
    print("Ajustando López Aliaga...", file=sys.stderr)
    idata_la, _ = fit_candidate(df, "LA", "pct_RP_2021",
                                args.draws, args.tune, args.chains, args.seed + 1)

    votes_s = posterior_final_votes(idata_s, df, "Sanchez")
    votes_la = posterior_final_votes(idata_la, df, "LA")

    p_sanchez_wins = float((votes_s > votes_la).mean())
    lead = votes_s - votes_la
    lead_q = np.quantile(lead, [0.025, 0.5, 0.975])

    report = {
        "modelo": "pymc_binomial_jerarquico",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
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
            "rhat_max_sanchez": float(az.summary(idata_s, var_names=["mu","beta","sigma_dep","sigma_dist","kappa"])["r_hat"].max()),
            "rhat_max_la": float(az.summary(idata_la, var_names=["mu","beta","sigma_dep","sigma_dist","kappa"])["r_hat"].max()),
            "kappa_mean_sanchez": float(idata_s.posterior["kappa"].mean().item()),
            "kappa_mean_la": float(idata_la.posterior["kappa"].mean().item()),
            "tiempo_segundos": round(time.time() - t0, 1),
        },
    }
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    # Además, el "latest" sobrescrito
    (Path(args.out_dir) / "pymc_latest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report["prediccion"], indent=2, ensure_ascii=False))
    print(f"\nGuardado: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
