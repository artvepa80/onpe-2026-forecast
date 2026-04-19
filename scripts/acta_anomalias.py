#!/usr/bin/env python3
"""
Análisis estadístico de anomalías sobre metadata de actas observadas (JEE).

Genera:
  - data/anomalias_summary.json : resumen para el dashboard
  - data/actas_sospechosas.csv  : actas flaggeadas con razón
  - Imprime reporte texto a stdout
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def load(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            # Normalize numeric/bool columns
            for k in ["totalElectoresHabiles", "ts_digitalizacion_ms", "ts_digitacion_ms",
                      "ts_contabilizacion_ms", "ts_observada_ms", "proc_segundos", "idEleccion"]:
                v = r.get(k)
                if v not in (None, ""):
                    try:
                        r[k] = int(float(v))
                    except ValueError:
                        r[k] = None
            for k in ["flag_flash_acta", "flag_hora_anomala"]:
                v = (r.get(k) or "").lower()
                r[k] = v == "true"
            rows.append(r)
    return rows


def analyze(rows: list[dict]) -> dict:
    n = len(rows)
    pres = [r for r in rows if r.get("idEleccion") == 10]
    senado = [r for r in rows if r.get("idEleccion") == 11]
    diputados = [r for r in rows if r.get("idEleccion") == 13]
    pandino = [r for r in rows if r.get("idEleccion") == 14]
    print(f"Total actas JEE scrapeadas: {n}", file=sys.stderr)
    print(f"  Presidencial: {len(pres)}", file=sys.stderr)
    print(f"  Senadores:    {len(senado)}", file=sys.stderr)
    print(f"  Diputados:    {len(diputados)}", file=sys.stderr)
    print(f"  Parl. Andino: {len(pandino)}", file=sys.stderr)

    # Razones de observación
    razones = Counter(r.get("estadoDescripcionActaResolucion","(sin dato)") for r in pres)
    # Consolidar duplicados tipo "Acta con X, Acta con X"
    razones_norm = Counter()
    for raz, c in razones.items():
        parts = sorted(set(p.strip() for p in (raz or "").split(",") if p.strip()))
        key = ", ".join(parts) if parts else "(sin dato)"
        razones_norm[key] += c

    # Distribución horaria de digitalización
    horas = Counter()
    for r in pres:
        hhmm = r.get("hora_digitaliz_lima") or ""
        if " " in hhmm:
            hora = int(hhmm.split(" ")[1].split(":")[0])
            horas[hora] += 1

    # Anomalías específicas
    flash = sum(1 for r in pres if r.get("flag_flash_acta"))
    nocturno = sum(1 for r in pres if r.get("flag_hora_anomala"))

    # Distribución por departamento
    dep_counts = Counter(r.get("ubigeoNivel01") for r in pres)

    # Locales de votación con más actas observadas (posible problema sistémico)
    local_counts = Counter((r.get("ubigeoNivel03"), r.get("nombreLocalVotacion")) for r in pres)
    local_sospechosos = [
        {"distrito": d, "local": l, "actas_obs": c}
        for (d, l), c in local_counts.most_common(20) if l
    ]

    summary = {
        "total_actas_jee_scrapeadas": n,
        "por_eleccion": {
            "presidencial": len(pres),
            "senadores": len(senado),
            "diputados": len(diputados),
            "parlamento_andino": len(pandino),
        },
        "presidencial": {
            "razones_top": razones_norm.most_common(15),
            "distribucion_horaria_digitalizacion": dict(sorted(horas.items())),
            "actas_flash_seg_lt_30": flash,
            "actas_nocturno_2_5am": nocturno,
            "por_departamento": dict(dep_counts.most_common()),
            "locales_con_mas_obs": local_sospechosos[:10],
        },
    }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=str(DATA / "actas_jee_metadata.csv"))
    ap.add_argument("--out-json", default=str(DATA / "anomalias_summary.json"))
    args = ap.parse_args()

    rows = load(Path(args.inp))
    summary = analyze(rows)

    Path(args.out_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print(f"\nResumen -> {args.out_json}", file=sys.stderr)

    # Print humano
    print("\n=== RAZONES DE OBSERVACIÓN (Presidencial) ===")
    for raz, c in summary["presidencial"]["razones_top"][:10]:
        pct = 100 * c / max(1, summary["por_eleccion"]["presidencial"])
        print(f"  {c:>4}  ({pct:>5.1f}%)  {raz}")

    print("\n=== DISTRIBUCIÓN HORARIA DIGITALIZACIÓN ===")
    hist = summary["presidencial"]["distribucion_horaria_digitalizacion"]
    total = sum(hist.values())
    for h in range(24):
        c = hist.get(str(h), hist.get(h, 0))
        bar = "█" * int(40 * c / max(1, max(hist.values())))
        print(f"  {h:02d}:00  {c:>4}  {bar}")

    print(f"\n=== ANOMALÍAS ===")
    print(f"  Flash (<30s): {summary['presidencial']['actas_flash_seg_lt_30']}")
    print(f"  Nocturno (2-5am Lima): {summary['presidencial']['actas_nocturno_2_5am']}")

    print("\n=== DEPARTAMENTOS CON MÁS OBSERVADAS (Top 10) ===")
    for dep, c in list(summary["presidencial"]["por_departamento"].items())[:10]:
        print(f"  {dep:<20} {c:>4}")

    print("\n=== LOCALES CON MÁS OBSERVADAS ===")
    for l in summary["presidencial"]["locales_con_mas_obs"]:
        print(f"  {l['actas_obs']:>3}  {l['distrito']}  {l['local']}")


if __name__ == "__main__":
    main()
