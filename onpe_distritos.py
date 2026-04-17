#!/usr/bin/env python3
"""
ONPE — Resultados por distrito (Elecciones Generales 2026, primera vuelta).

Recorre departamento → provincia → distrito usando el backend de
`resultadoelectoral.onpe.gob.pe` (mismo que usa la SPA Angular) y escribe
un CSV con los votos por candidato en cada distrito.

Uso:
    python3 onpe_distritos.py                                   # corrida única, todo el Perú
    python3 onpe_distritos.py --dep 140000                      # solo Lima
    python3 onpe_distritos.py --watch 10                        # refrescar cada 10 min
    python3 onpe_distritos.py --watch 5 --dep 140000 --out lima.csv

En modo watch:
  - `<out>.csv`      se sobreescribe con el snapshot más reciente
  - `<out>.log.csv`  acumula todos los snapshots con columna `snapshot_ts`
  - se detiene con Ctrl+C

Requiere solo la stdlib + `requests`:
    pip install requests
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from typing import Any, Iterable

import requests

BASE = "https://resultadoelectoral.onpe.gob.pe/presentacion-backend"
ID_ELECCION = 10           # Presidencial 2026 (1ra vuelta)
ID_AMBITO = 1              # 1 = Perú, 2 = extranjero

# La WAF/CloudFront rechaza peticiones sin headers "de navegador" —
# estos son los mínimos que la dejan pasar.
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


def get_json(session: requests.Session, url: str, retries: int = 3) -> Any:
    """GET con reintentos. Devuelve el campo `data` o None si 204."""
    for attempt in range(retries):
        r = session.get(url, timeout=20)
        if r.status_code == 204:
            return None
        if r.ok and r.headers.get("content-type", "").startswith("application/json"):
            return r.json().get("data")
        time.sleep(1.5 * (attempt + 1))
    r.raise_for_status()
    raise RuntimeError(f"Respuesta inesperada: {r.status_code} {r.text[:120]}")


def list_departamentos(s: requests.Session) -> list[dict]:
    return get_json(s, f"{BASE}/ubigeos/departamentos?idEleccion={ID_ELECCION}&idAmbitoGeografico={ID_AMBITO}") or []


def list_provincias(s: requests.Session, dep: str) -> list[dict]:
    return get_json(
        s,
        f"{BASE}/ubigeos/provincias?idEleccion={ID_ELECCION}"
        f"&idAmbitoGeografico={ID_AMBITO}&idUbigeoDepartamento={dep}",
    ) or []


def list_distritos(s: requests.Session, prov: str) -> list[dict]:
    return get_json(
        s,
        f"{BASE}/ubigeos/distritos?idEleccion={ID_ELECCION}"
        f"&idAmbitoGeografico={ID_AMBITO}&idUbigeoProvincia={prov}",
    ) or []


def candidatos_distrito(s: requests.Session, dep: str, prov: str, dist: str) -> list[dict]:
    url = (
        f"{BASE}/eleccion-presidencial/participantes-ubicacion-geografica-nombre"
        f"?tipoFiltro=ubigeo_nivel_03&idAmbitoGeografico={ID_AMBITO}"
        f"&ubigeoNivel1={dep}&ubigeoNivel2={prov}&ubigeoNivel3={dist}"
        f"&idEleccion={ID_ELECCION}"
    )
    return get_json(s, url) or []


def totales_distrito(s: requests.Session, dep: str, prov: str, dist: str) -> dict:
    url = (
        f"{BASE}/resumen-general/totales?idAmbitoGeografico={ID_AMBITO}"
        f"&idEleccion={ID_ELECCION}&tipoFiltro=ubigeo_nivel_03"
        f"&idUbigeoDepartamento={dep}&idUbigeoProvincia={prov}&idUbigeoDistrito={dist}"
    )
    return get_json(s, url) or {}


def iter_rows(s: requests.Session, dep_filter: str | None) -> Iterable[dict]:
    for dep in list_departamentos(s):
        if dep_filter and dep["ubigeo"] != dep_filter:
            continue
        print(f"[dep] {dep['nombre']} ({dep['ubigeo']})", file=sys.stderr)
        for prov in list_provincias(s, dep["ubigeo"]):
            print(f"  [prov] {prov['nombre']}", file=sys.stderr)
            for dist in list_distritos(s, prov["ubigeo"]):
                cands = candidatos_distrito(s, dep["ubigeo"], prov["ubigeo"], dist["ubigeo"])
                tot = totales_distrito(s, dep["ubigeo"], prov["ubigeo"], dist["ubigeo"])
                for c in cands:
                    yield {
                        "departamento": dep["nombre"],
                        "provincia": prov["nombre"],
                        "distrito": dist["nombre"],
                        "ubigeo_dep": dep["ubigeo"],
                        "ubigeo_prov": prov["ubigeo"],
                        "ubigeo_dist": dist["ubigeo"],
                        "partido": c.get("nombreAgrupacionPolitica"),
                        "candidato": c.get("nombreCandidato"),
                        "dni": c.get("dniCandidato"),
                        "votos_validos": c.get("totalVotosValidos"),
                        "pct_validos": c.get("porcentajeVotosValidos"),
                        "pct_emitidos": c.get("porcentajeVotosEmitidos"),
                        "actas_contab_pct": tot.get("actasContabilizadas"),
                        "total_votos_emitidos": tot.get("totalVotosEmitidos"),
                        "total_votos_validos": tot.get("totalVotosValidos"),
                        "participacion_pct": tot.get("participacionCiudadana"),
                    }
                time.sleep(0.15)  # ser amable con el servidor


FIELDS = [
    "departamento", "provincia", "distrito",
    "ubigeo_dep", "ubigeo_prov", "ubigeo_dist",
    "partido", "candidato", "dni",
    "votos_validos", "pct_validos", "pct_emitidos",
    "actas_contab_pct", "total_votos_emitidos", "total_votos_validos", "participacion_pct",
]


def run_once(session: requests.Session, dep_filter: str | None, out_path: str, log_path: str | None) -> int:
    """Una pasada completa. Escribe snapshot en `out_path` y (opcional) append en `log_path`."""
    ts = datetime.now().isoformat(timespec="seconds")
    log_file = None
    log_writer = None
    if log_path:
        new_log = not os.path.exists(log_path)
        log_file = open(log_path, "a", newline="", encoding="utf-8")
        log_writer = csv.DictWriter(log_file, fieldnames=["snapshot_ts", *FIELDS])
        if new_log:
            log_writer.writeheader()

    try:
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            n = 0
            for row in iter_rows(session, dep_filter):
                w.writerow(row)
                if log_writer:
                    log_writer.writerow({"snapshot_ts": ts, **row})
                n += 1
                if n % 200 == 0:
                    f.flush()
                    if log_file:
                        log_file.flush()
                    print(f"  ... {n} filas", file=sys.stderr)
    finally:
        if log_file:
            log_file.close()
    print(f"[{ts}] snapshot: {n} filas -> {out_path}", file=sys.stderr)
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dep", help="Filtrar por código ubigeo de departamento (ej. 140000 = Lima)")
    ap.add_argument("--out", default="onpe_distritos.csv", help="Archivo CSV de salida (snapshot actual)")
    ap.add_argument("--watch", type=float, default=0,
                    help="Refrescar en bucle cada N minutos (0 = corrida única)")
    ap.add_argument("--log", help="CSV append-only con histórico (default: <out>.log.csv si --watch)")
    args = ap.parse_args()

    log_path = args.log
    if args.watch and not log_path:
        base, ext = os.path.splitext(args.out)
        log_path = f"{base}.log{ext or '.csv'}"

    s = requests.Session()
    s.headers.update(HEADERS)

    if not args.watch:
        run_once(s, args.dep, args.out, log_path)
        return

    interval = args.watch * 60
    print(f"Modo watch: refrescando cada {args.watch} min. Log: {log_path}. Ctrl+C para detener.", file=sys.stderr)
    try:
        while True:
            t0 = time.time()
            try:
                run_once(s, args.dep, args.out, log_path)
            except Exception as e:
                print(f"[warn] corrida falló: {e}", file=sys.stderr)
            elapsed = time.time() - t0
            sleep_for = max(10, interval - elapsed)
            print(f"  dormiendo {int(sleep_for)}s...", file=sys.stderr)
            time.sleep(sleep_for)
    except KeyboardInterrupt:
        print("\nDetenido por el usuario.", file=sys.stderr)


if __name__ == "__main__":
    main()
