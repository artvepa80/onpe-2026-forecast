#!/usr/bin/env python3
"""
Prepara el panel de distritos para los modelos:

  data/raw_2021_partidos.csv  (baseline ONPE 2021, 1ra vuelta, por mesa/distrito)
  + snapshot actual de ONPE 2026 (vía API en vivo)
  -> data/district_panel.csv

Columnas del panel:
  ubigeo_dep, ubigeo_prov, ubigeo_dist, departamento, provincia, distrito,
  votos_Sanchez, votos_LA, votos_validos_contados, total_actas, actas_contab,
  actas_pendientes, pct_avance_dist,
  pct_RP_2021, pct_JpP_2021, total_emitidos_2021

Candidatos 2026 mapeados:
  Roberto Sánchez  -> cod_agrup = "16" (PARTIDO DEL BUEN GOBIERNO)
  Rafael López Aliaga -> cod_agrup = "35" (RENOVACIÓN POPULAR)

Proxies 2021:
  Sánchez -> Juntos por el Perú (Verónika Mendoza), cod_partido 00000024
  López Aliaga -> Renovación Popular (sí postuló en 2021), cod_partido 00000035

(Códigos confirmables con `python3 prep.py --show-partidos-2021`)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests

ROOT = Path(__file__).parent
DATA = ROOT / "data"

BASE = "https://resultadoelectoral.onpe.gob.pe/presentacion-backend"
ID_ELECCION = 10
ID_AMBITO = 1
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://resultadoelectoral.onpe.gob.pe/main/presidenciales",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Dest": "empty",
}

# Códigos a usar en el match (se pueden sobreescribir con flags)
COD_RP_2021 = "00000020"    # Renovación Popular — López Aliaga (2021)
COD_JPP_2021 = "00000008"   # Juntos por el Perú — Mendoza (proxy de Sánchez, 2021)
COD_LA_2026 = "35"       # Renovación Popular — Rafael López Aliaga
COD_SANCHEZ_2026 = "10"  # Juntos por el Perú — Roberto Sánchez Palomino


# ---------------------------------------------------------------------------
# Covariables INEI (ubigeo-peru-aumentado de Castagnetto)
# ---------------------------------------------------------------------------
def load_inei_covariates(path: Path) -> dict[str, dict]:
    """Devuelve {reniec_ubigeo: {altitud, idh, pct_pobreza, densidad, ...}}.

    Usamos la columna `reniec` para empatar con la codificación de ONPE
    (Lima=14, Callao=24). La columna `inei` usa codificación distinta
    (Lima=15, Callao=07).
    """
    out: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ub = row["reniec"]
            def _f(k):
                v = row.get(k, "") or ""
                try: return float(v)
                except ValueError: return None
            out[ub] = {
                "altitud_m": _f("altitude"),
                "idh_2019": _f("idh_2019"),
                "pct_pobreza": _f("pct_pobreza_total"),
                "pct_pobreza_ext": _f("pct_pobreza_extrema"),
                "densidad_2020": _f("pob_densidad_2020"),
                "vuln_alim": _f("indice_vulnerabilidad_alimentaria"),
                "macroregion": (row.get("macroregion_inei") or "").strip() or None,
                "lat": _f("latitude"),
                "lon": _f("longitude"),
                "superficie_km2": _f("superficie"),
            }
    return out


# ---------------------------------------------------------------------------
# 2021 baseline aggregation
# ---------------------------------------------------------------------------
def load_2021_baseline(path: Path, cod_la: str, cod_jpp: str) -> dict[str, dict]:
    """Devuelve {ubigeo_dist: {pct_RP_2021, pct_JpP_2021, total_emitidos_2021, depa, prov, dist}}"""
    out: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ub = row["ubigeo"]
            rec = out.setdefault(ub, {
                "departamento": row["departamento"],
                "provincia": row["provincia"],
                "distrito": row["distrito"],
                "pct_RP_2021": None,
                "pct_JpP_2021": None,
                "total_emitidos_2021": None,
            })
            cod = row["cod_partido"]
            if cod == cod_la:
                rec["pct_RP_2021"] = float(row["pct_validos"] or 0)
            elif cod == cod_jpp:
                rec["pct_JpP_2021"] = float(row["pct_validos"] or 0)
            # Aproximación simple de total emitidos: mayor total_votos de lista normalizado
            if row["nlista"]:
                try:
                    n = int(row["nlista"])
                    if rec["total_emitidos_2021"] is None or n > rec["total_emitidos_2021"]:
                        rec["total_emitidos_2021"] = n
                except ValueError:
                    pass
    return out


# ---------------------------------------------------------------------------
# 2026 live snapshot (reutiliza la misma API que onpe_distritos.py)
# ---------------------------------------------------------------------------
def _get(session, url, retries=3):
    for i in range(retries):
        r = session.get(url, timeout=20)
        if r.status_code == 204:
            return None
        if r.ok and r.headers.get("content-type", "").startswith("application/json"):
            return r.json().get("data")
        time.sleep(1.2 * (i + 1))
    r.raise_for_status()


def fetch_live_snapshot(session, dep_filter=None):
    """Iterador de dicts por distrito con votos actuales de Sánchez y LA + totales."""
    deps = _get(session, f"{BASE}/ubigeos/departamentos?idEleccion={ID_ELECCION}&idAmbitoGeografico={ID_AMBITO}") or []
    for dep in deps:
        if dep_filter and dep["ubigeo"] != dep_filter:
            continue
        print(f"[dep] {dep['nombre']}", file=sys.stderr)
        provs = _get(session, f"{BASE}/ubigeos/provincias?idEleccion={ID_ELECCION}&idAmbitoGeografico={ID_AMBITO}&idUbigeoDepartamento={dep['ubigeo']}") or []
        for prov in provs:
            dists = _get(session, f"{BASE}/ubigeos/distritos?idEleccion={ID_ELECCION}&idAmbitoGeografico={ID_AMBITO}&idUbigeoProvincia={prov['ubigeo']}") or []
            for dist in dists:
                url_cand = (
                    f"{BASE}/eleccion-presidencial/participantes-ubicacion-geografica-nombre"
                    f"?tipoFiltro=ubigeo_nivel_03&idAmbitoGeografico={ID_AMBITO}"
                    f"&ubigeoNivel1={dep['ubigeo']}&ubigeoNivel2={prov['ubigeo']}"
                    f"&ubigeoNivel3={dist['ubigeo']}&idEleccion={ID_ELECCION}"
                )
                url_tot = (
                    f"{BASE}/resumen-general/totales?idAmbitoGeografico={ID_AMBITO}"
                    f"&idEleccion={ID_ELECCION}&tipoFiltro=ubigeo_nivel_03"
                    f"&idUbigeoDepartamento={dep['ubigeo']}&idUbigeoProvincia={prov['ubigeo']}"
                    f"&idUbigeoDistrito={dist['ubigeo']}"
                )
                cands = _get(session, url_cand) or []
                tot = _get(session, url_tot) or {}
                votos_s = next((int(c["totalVotosValidos"]) for c in cands if c["codigoAgrupacionPolitica"] == COD_SANCHEZ_2026), 0)
                votos_la = next((int(c["totalVotosValidos"]) for c in cands if c["codigoAgrupacionPolitica"] == COD_LA_2026), 0)
                yield {
                    "ubigeo_dep": dep["ubigeo"],
                    "ubigeo_prov": prov["ubigeo"],
                    "ubigeo_dist": dist["ubigeo"],
                    "departamento": dep["nombre"],
                    "provincia": prov["nombre"],
                    "distrito": dist["nombre"],
                    "votos_Sanchez": votos_s,
                    "votos_LA": votos_la,
                    "votos_validos_contados": int(tot.get("totalVotosValidos") or 0),
                    "total_actas": int(tot.get("totalActas") or 0),
                    "actas_contab": int(tot.get("contabilizadas") or 0),
                    "pct_avance_dist": float(tot.get("actasContabilizadas") or 0),
                }
                time.sleep(0.12)


# ---------------------------------------------------------------------------
# Join + write
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-2021", default=str(DATA / "raw_2021_partidos.csv"))
    ap.add_argument("--inei", default=str(DATA / "inei/ubigeo_distrito.csv"),
                    help="CSV con covariables INEI por distrito (RENIEC ubigeo)")
    ap.add_argument("--out", default=str(DATA / "district_panel.csv"))
    ap.add_argument("--dep", help="Filtrar por ubigeo departamento")
    ap.add_argument("--cod-la-2021", default=COD_RP_2021)
    ap.add_argument("--cod-jpp-2021", default=COD_JPP_2021)
    ap.add_argument("--show-partidos-2021", action="store_true",
                    help="Listar (cod_partido, partido) únicos del baseline 2021 y salir")
    args = ap.parse_args()

    if args.show_partidos_2021:
        seen = {}
        with open(args.baseline_2021, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                seen.setdefault(row["cod_partido"], row["partido"])
        for c, p in sorted(seen.items()):
            print(f"{c}  {p}")
        return

    print(f"Cargando baseline 2021 ({args.baseline_2021})...", file=sys.stderr)
    baseline = load_2021_baseline(Path(args.baseline_2021), args.cod_la_2021, args.cod_jpp_2021)
    print(f"  distritos 2021: {len(baseline)}", file=sys.stderr)

    print(f"Cargando covariables INEI ({args.inei})...", file=sys.stderr)
    inei = load_inei_covariates(Path(args.inei)) if Path(args.inei).exists() else {}
    print(f"  distritos INEI: {len(inei)}", file=sys.stderr)

    fields = [
        "ubigeo_dep", "ubigeo_prov", "ubigeo_dist",
        "departamento", "provincia", "distrito",
        "votos_Sanchez", "votos_LA", "votos_validos_contados",
        "total_actas", "actas_contab", "actas_pendientes", "pct_avance_dist",
        "pct_RP_2021", "pct_JpP_2021", "total_emitidos_2021",
        # Covariables INEI (Capa 1 econométrica)
        "altitud_m", "idh_2019", "pct_pobreza", "pct_pobreza_ext",
        "densidad_2020", "vuln_alim", "macroregion",
        "lat", "lon", "superficie_km2",
    ]

    s = requests.Session(); s.headers.update(HEADERS)
    n_match = n_miss = 0
    n_inei_match = n_inei_miss = 0
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for row in fetch_live_snapshot(s, args.dep):
            b = baseline.get(row["ubigeo_dist"]) or {}
            if b: n_match += 1
            else: n_miss += 1
            row["actas_pendientes"] = max(0, row["total_actas"] - row["actas_contab"])
            row["pct_RP_2021"] = b.get("pct_RP_2021")
            row["pct_JpP_2021"] = b.get("pct_JpP_2021")
            row["total_emitidos_2021"] = b.get("total_emitidos_2021")
            # INEI join
            ic = inei.get(row["ubigeo_dist"]) or {}
            if ic: n_inei_match += 1
            else: n_inei_miss += 1
            for k in ("altitud_m","idh_2019","pct_pobreza","pct_pobreza_ext",
                      "densidad_2020","vuln_alim","macroregion","lat","lon","superficie_km2"):
                row[k] = ic.get(k)
            w.writerow(row)
    print(f"OK. match 2021={n_match}/{n_match+n_miss}, "
          f"INEI={n_inei_match}/{n_inei_match+n_inei_miss} -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
