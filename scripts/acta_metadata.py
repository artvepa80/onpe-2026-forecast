#!/usr/bin/env python3
"""
Baja metadata de todas las actas OBSERVADAS (enviadas al JEE) vía ONPE API.

Sin OCR, cero costo. Por cada acta JEE extrae:
  - codigo mesa, ubigeo, eleccion
  - razón de observación (estadoDescripcionActaResolucion) — "Acta con error aritmético", etc.
  - timeline: digitalización, digitación, contabilización (timestamps)
  - tiempo de procesamiento total
  - flags: flash acta (< 30s), night processing (2-6 am), etc.

Output: data/actas_jee_metadata.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

BASE = "https://resultadoelectoral.onpe.gob.pe/presentacion-backend"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://resultadoelectoral.onpe.gob.pe/main/actas",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Dest": "empty",
}

ID_AMBITO = 1
LIMA_TZ = timezone(timedelta(hours=-5))


def _get(s, url, retries=3):
    for i in range(retries):
        try:
            r = s.get(url, timeout=25)
            if r.status_code == 204:
                return None
            if r.ok and r.headers.get("content-type", "").startswith("application/json"):
                return r.json().get("data")
        except Exception:
            pass
        time.sleep(2 ** i)
    return None


def warm(s):
    try:
        s.get("https://resultadoelectoral.onpe.gob.pe/main/actas", timeout=20)
    except Exception:
        pass


def list_observadas(s, ubigeo: str, page_size=100):
    """Iterator sobre todas las actas observadas en un distrito (paginado)."""
    page = 0
    while True:
        url = f"{BASE}/actas/observadas?pagina={page}&tamanio={page_size}&idAmbitoGeografico={ID_AMBITO}&idUbigeo={ubigeo}"
        d = _get(s, url)
        if not d:
            return
        content = d.get("content") or []
        for a in content:
            yield a
        total_pages = d.get("totalPaginas", 0)
        if page + 1 >= total_pages:
            return
        page += 1


def get_acta_detail(s, acta_id: int) -> dict | None:
    return _get(s, f"{BASE}/actas/{acta_id}")


def iter_departamentos(s):
    return _get(s, f"{BASE}/ubigeos/departamentos?idEleccion=10&idAmbitoGeografico={ID_AMBITO}") or []


def iter_provincias(s, dep_ubigeo):
    return _get(s, f"{BASE}/ubigeos/provincias?idEleccion=10&idAmbitoGeografico={ID_AMBITO}&idUbigeoDepartamento={dep_ubigeo}") or []


def iter_distritos(s, prov_ubigeo):
    return _get(s, f"{BASE}/ubigeos/distritos?idEleccion=10&idAmbitoGeografico={ID_AMBITO}&idUbigeoProvincia={prov_ubigeo}") or []


def analyze_timeline(linea_tiempo: list) -> dict:
    """Devuelve flags y tiempos del procesamiento."""
    if not linea_tiempo:
        return {}
    events = {e["codigoEstadoActa"]: e["fechaRegistro"] for e in linea_tiempo}
    t_digitaliz = events.get("T")
    t_digit     = events.get("D")
    t_contab    = events.get("C")
    t_observada = events.get("O") or events.get("E")
    result = {
        "ts_digitalizacion_ms": t_digitaliz,
        "ts_digitacion_ms":     t_digit,
        "ts_contabilizacion_ms": t_contab,
        "ts_observada_ms":      t_observada,
    }
    # Tiempo total (digitalización → contabilización)
    if t_digitaliz and t_contab:
        delta_s = (t_contab - t_digitaliz) / 1000
        result["proc_segundos"] = int(delta_s)
        result["flag_flash_acta"] = delta_s < 30  # Procesada en <30s (sospechoso)
    # Hora Lima
    if t_digitaliz:
        dt = datetime.fromtimestamp(t_digitaliz/1000, tz=LIMA_TZ)
        result["hora_digitaliz_lima"] = dt.strftime("%Y-%m-%d %H:%M")
        h = dt.hour
        result["flag_hora_anomala"] = h in (2, 3, 4, 5)   # 2am-5am Lima
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DATA / "actas_jee_metadata.csv"))
    ap.add_argument("--dep", help="Filtrar departamento (ubigeo 6 dígitos)")
    ap.add_argument("--eleccion", type=int, default=10,
                    help="Filtrar por idEleccion (10=Presidencial). Si no, incluye todas.")
    ap.add_argument("--sample", type=int, default=0, help="Detener tras N actas (0=sin límite)")
    args = ap.parse_args()

    s = requests.Session(); s.headers.update(HEADERS)
    warm(s)

    fields = [
        "id", "idEleccion", "codigoMesa", "idUbigeo",
        "ubigeoNivel01", "ubigeoNivel02", "ubigeoNivel03",
        "nombreLocalVotacion", "totalElectoresHabiles",
        "descripcionEstadoActa", "estadoDescripcionActaResolucion",
        "ts_digitalizacion_ms", "ts_digitacion_ms", "ts_contabilizacion_ms", "ts_observada_ms",
        "hora_digitaliz_lima", "proc_segundos",
        "flag_flash_acta", "flag_hora_anomala",
    ]

    n = 0
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for dep in iter_departamentos(s):
            if args.dep and dep["ubigeo"] != args.dep:
                continue
            print(f"[dep] {dep['nombre']}", file=sys.stderr)
            for prov in iter_provincias(s, dep["ubigeo"]):
                for dist in iter_distritos(s, prov["ubigeo"]):
                    # idUbigeo sin ceros padding — e.g. 010101 → 10101
                    ubigeo_num = str(int(dist["ubigeo"]))
                    for a in list_observadas(s, ubigeo_num):
                        if args.eleccion and a.get("idEleccion") != args.eleccion:
                            continue
                        detail = get_acta_detail(s, a["id"])
                        if not detail:
                            continue
                        tl = analyze_timeline(detail.get("lineaTiempo") or [])
                        row = {
                            "id": detail.get("id"),
                            "idEleccion": detail.get("idEleccion"),
                            "codigoMesa": detail.get("codigoMesa"),
                            "idUbigeo": ubigeo_num,
                            "ubigeoNivel01": detail.get("ubigeoNivel01"),
                            "ubigeoNivel02": detail.get("ubigeoNivel02"),
                            "ubigeoNivel03": detail.get("ubigeoNivel03"),
                            "nombreLocalVotacion": detail.get("nombreLocalVotacion"),
                            "totalElectoresHabiles": detail.get("totalElectoresHabiles"),
                            "descripcionEstadoActa": detail.get("descripcionEstadoActa"),
                            "estadoDescripcionActaResolucion": detail.get("estadoDescripcionActaResolucion"),
                        }
                        row.update(tl)
                        w.writerow(row)
                        n += 1
                        if n % 50 == 0:
                            f.flush()
                            print(f"  {n} actas procesadas", file=sys.stderr)
                        if args.sample and n >= args.sample:
                            print(f"OK. Sample {n} actas -> {args.out}", file=sys.stderr)
                            return
                        time.sleep(0.1)
    print(f"OK. Total {n} actas JEE -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
