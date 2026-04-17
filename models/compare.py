#!/usr/bin/env python3
"""
Evaluación final — corre una vez cerrada la primera vuelta, cuando ONPE
publica el 100% de actas. Compara las predicciones de PyMC vs brms contra
los resultados oficiales y produce `forecasts/comparison_report.md` + JSON.

Métricas:
  * MAE (puntos porcentuales) sobre votos finales de Sánchez y LA
  * Cobertura del CI 95% (¿el intervalo contiene el valor real?)
  * Brier score para P(Sánchez > LA) — binario: quién quedó 2°
  * Signo correcto: ¿predijo la dirección del desenlace?

Uso:
  python3 models/compare.py \
      --pymc forecasts/pymc_latest.json \
      --brms forecasts/brms_latest.json \
      --final data/final_resultados.json

`final_resultados.json` es un archivo simple:
  {
    "Sanchez_votos": 2034512,
    "LA_votos": 2011330,
    "Sanchez_pct": 12.35,
    "LA_pct": 12.21,
    "timestamp": "2026-04-20T23:59:00"
  }
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


def load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def abs_err_votes(pred: dict, actual: int) -> int:
    return abs(int(pred["mediana"]) - actual)


def coverage(pred: dict, actual: int) -> bool:
    return pred["ci_low"] <= actual <= pred["ci_high"]


def brier(p_model: float, outcome: int) -> float:
    # outcome = 1 si Sánchez > LA en la realidad
    return (p_model - outcome) ** 2


def evaluate(model_report: dict, final: dict) -> dict:
    pred_s  = model_report["prediccion"]["Sanchez"]
    pred_la = model_report["prediccion"]["LA"]
    p_sanchez = model_report["prediccion"]["P(Sanchez > LA)"]
    outcome = int(final["Sanchez_votos"] > final["LA_votos"])

    mae_votes_s  = abs_err_votes(pred_s,  final["Sanchez_votos"])
    mae_votes_la = abs_err_votes(pred_la, final["LA_votos"])
    cov_s  = coverage(pred_s,  final["Sanchez_votos"])
    cov_la = coverage(pred_la, final["LA_votos"])

    sign_correct = (pred_s["mediana"] > pred_la["mediana"]) == bool(outcome)

    return {
        "modelo": model_report["modelo"],
        "mae_votos_Sanchez": mae_votes_s,
        "mae_votos_LA": mae_votes_la,
        "mae_votos_promedio": (mae_votes_s + mae_votes_la) / 2,
        "cobertura_CI95_Sanchez": bool(cov_s),
        "cobertura_CI95_LA": bool(cov_la),
        "brier_P_Sanchez_gana_2do": brier(p_sanchez, outcome),
        "signo_correcto": bool(sign_correct),
        "P_Sanchez_predicho": p_sanchez,
        "outcome_real": outcome,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pymc", default="forecasts/pymc_latest.json")
    ap.add_argument("--brms", default="forecasts/brms_latest.json")
    ap.add_argument("--final", required=True, help="JSON con resultados oficiales finales")
    ap.add_argument("--out-md", default="forecasts/comparison_report.md")
    ap.add_argument("--out-json", default="forecasts/comparison_report.json")
    args = ap.parse_args()

    pymc_rep = load(args.pymc)
    brms_rep = load(args.brms)
    final    = load(args.final)

    results = {
        "fecha_evaluacion": datetime.now().isoformat(timespec="seconds"),
        "final_oficial": final,
        "pymc": evaluate(pymc_rep, final),
        "brms": evaluate(brms_rep, final),
    }
    Path(args.out_json).write_text(json.dumps(results, indent=2, ensure_ascii=False))

    # Markdown
    rows = ["| Métrica | PyMC | brms |", "|---|---|---|"]
    keys = [
        ("mae_votos_Sanchez", "MAE votos Sánchez"),
        ("mae_votos_LA", "MAE votos LA"),
        ("mae_votos_promedio", "MAE promedio"),
        ("cobertura_CI95_Sanchez", "CI95 cubre Sánchez"),
        ("cobertura_CI95_LA", "CI95 cubre LA"),
        ("brier_P_Sanchez_gana_2do", "Brier P(Sánchez 2°)"),
        ("signo_correcto", "Dirección correcta"),
        ("P_Sanchez_predicho", "P(Sánchez>LA) predicho"),
    ]
    for k, label in keys:
        rows.append(f"| {label} | {results['pymc'][k]} | {results['brms'][k]} |")

    md = f"""# Comparación PyMC vs brms — primera vuelta 2026

**Resultado oficial:** Sánchez={final['Sanchez_votos']:,} votos ({final.get('Sanchez_pct','?')}%), \
LA={final['LA_votos']:,} votos ({final.get('LA_pct','?')}%) — \
2° puesto: **{'Sánchez' if final['Sanchez_votos']>final['LA_votos'] else 'LA'}**

{chr(10).join(rows)}

## Lectura
- **MAE** en votos: error absoluto de la mediana posterior contra resultado real.
- **Cobertura CI95:** ¿el intervalo del modelo contenía el resultado? Si no → subestimó incertidumbre.
- **Brier:** menor es mejor (0 = certeza perfecta, 0.25 = 50/50).
- **Dirección correcta:** ¿la mediana predijo al ganador del 2° puesto?
"""
    Path(args.out_md).write_text(md)
    print(md)


if __name__ == "__main__":
    main()
