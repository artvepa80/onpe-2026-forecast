#!/usr/bin/env python3
"""
Baja las actas OCR'd de archive.org y cruza con nuestro metadata JEE.

Fuente: https://archive.org/details/actas-observadas-peru-eg2026
  (uploader: Luis Rivera583, 241 PDFs, 41 con OCR Tesseract)

Extrae de cada OCR .txt:
  - número de mesa (6 dígitos)
  - hora de instalación manuscrita ("Siendo las X del 12 de abril...")
Cruza con data/actas_jee_metadata.csv vía codigoMesa.

Hallazgos esperados:
  - Horas 2-5 AM (bug AM/PM del tweet de Tanaka)
  - Actas cuya razón en JEE no matches con lo visible en el acta
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ARCH_DIR = DATA / "archive_actas"
ARCH_DIR.mkdir(exist_ok=True, parents=True)

ARCHIVE_META = "https://archive.org/metadata/actas-observadas-peru-eg2026"
ARCHIVE_DL   = "https://archive.org/download/actas-observadas-peru-eg2026"

# Regex para extraer hora: "Siendo las [hora] del 12 de abril"
RE_HORA = re.compile(
    r"[Ss]iendo\s+las?\s+([^,\.]{1,40}?)\s+del?\s+[\"']?1[23]\s+de\s+abril",
    re.IGNORECASE
)
# Mesa: en header "MESA DE SUFRAGIO N.? 000033"
RE_MESA = re.compile(r"MESA\s+DE\s+SUFRAGIO\s+N\.?\s*\??\s*(\d{5,6})", re.IGNORECASE)


def download_ocr_files():
    """Lista metadata, baja solo los .txt de OCR."""
    print("Fetching archive.org metadata...", file=sys.stderr)
    m = requests.get(ARCHIVE_META, timeout=30).json()
    files = m.get("files", [])
    txts = [f for f in files if f.get("format") == "DjVuTXT"]
    print(f"Archivos OCR disponibles: {len(txts)}", file=sys.stderr)
    for f in txts:
        name = f["name"]
        path = ARCH_DIR / name
        if path.exists():
            continue
        url = f"{ARCHIVE_DL}/{name}"
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        path.write_text(r.text, encoding="utf-8")
    return list(ARCH_DIR.glob("ESCRUTINIO_*_djvu.txt"))


def parse_acta(path: Path) -> dict:
    txt = path.read_text(encoding="utf-8", errors="replace")
    # Número de mesa del filename como fallback
    fname_mesa = re.search(r"ESCRUTINIO_(\d+)", path.name).group(1)
    mesa_match = RE_MESA.search(txt)
    mesa = (mesa_match.group(1) if mesa_match else fname_mesa).zfill(6)
    hora_match = RE_HORA.search(txt)
    hora_raw = hora_match.group(1).strip() if hora_match else None
    return {
        "file": path.name,
        "codigoMesa": mesa,
        "hora_texto_raw": hora_raw,
        "texto_completo": txt[:2000],   # truncado para análisis
    }


def load_metadata() -> dict:
    out = {}
    p = DATA / "actas_jee_metadata.csv"
    if not p.exists():
        return out
    with p.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("idEleccion") != "10":
                continue
            # Usar codigoMesa (6 dígitos) como clave
            mesa = (row.get("codigoMesa") or "").zfill(6)
            # Si hay múltiples (una mesa puede tener observadas en varias elecciones)
            if mesa not in out:
                out[mesa] = row
    return out


def main():
    # 1) Download OCR files
    files = download_ocr_files()
    print(f"OCR txt files en disco: {len(files)}", file=sys.stderr)

    # 2) Parse cada uno
    parsed = [parse_acta(f) for f in sorted(files)]

    # 3) Cross-reference con metadata JEE
    jee_meta = load_metadata()
    print(f"Metadata JEE cargado: {len(jee_meta)} actas", file=sys.stderr)

    matches = []
    for p in parsed:
        m = jee_meta.get(p["codigoMesa"])
        row = {
            "archive_file": p["file"],
            "codigoMesa": p["codigoMesa"],
            "hora_texto_ocr": p["hora_texto_raw"],
            "match_en_jee": m is not None,
        }
        if m:
            row.update({
                "razon_jee": m.get("estadoDescripcionActaResolucion"),
                "ubigeo": m.get("idUbigeo"),
                "departamento": m.get("ubigeoNivel01"),
                "distrito": m.get("ubigeoNivel03"),
                "hora_digitaliz_lima": m.get("hora_digitaliz_lima"),
            })
        matches.append(row)

    # 4) Guardar comparación
    out_path = DATA / "archive_vs_jee.csv"
    fields = ["archive_file","codigoMesa","hora_texto_ocr","match_en_jee",
              "razon_jee","ubigeo","departamento","distrito","hora_digitaliz_lima"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in matches: w.writerow({k: r.get(k, "") for k in fields})

    # 5) Stats
    n = len(matches)
    n_match = sum(1 for r in matches if r["match_en_jee"])
    n_hora  = sum(1 for r in matches if r.get("hora_texto_ocr"))
    razones = {}
    for r in matches:
        if r.get("razon_jee"):
            razones[r["razon_jee"]] = razones.get(r["razon_jee"], 0) + 1

    print(f"\n=== RESULTADOS")
    print(f"Actas OCR analizadas:       {n}")
    print(f"Match en nuestro JEE meta:  {n_match}/{n}  ({100*n_match/n:.0f}%)")
    print(f"Hora extraída por regex:    {n_hora}/{n}  ({100*n_hora/n:.0f}%)")
    print(f"\nRazones JEE de las matches:")
    for raz, c in sorted(razones.items(), key=lambda x: -x[1]):
        print(f"  {c:>3}  {raz}")

    print(f"\n=== MUESTRA de horas OCR (primeras 15)")
    for r in matches[:15]:
        if r.get("hora_texto_ocr"):
            print(f"  mesa {r['codigoMesa']}: '{r['hora_texto_ocr']}' "
                  f"| {r.get('departamento','?')}/{r.get('distrito','?')}")

    print(f"\nOutput: {out_path}")


if __name__ == "__main__":
    main()
