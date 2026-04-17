# ONPE 2026 — Forecast del 2° puesto (Sánchez vs López Aliaga)

Pipeline reproducible para (1) scrapear el portal en vivo de ONPE, (2) modelar
Bayesianamente quién termina segundo en la 1ra vuelta presidencial 2026, y
(3) comparar dos stacks (PyMC en Python vs brms en R) contra los resultados
finales.

## Flujo

```
onpe_distritos.py  ──► data/*.csv snapshots en vivo (opcional --watch)
                          │
prep.py            ──► data/district_panel.csv  (join 2026 + baseline 2021)
                          │
            ┌─────────────┼─────────────┐
            ▼                           ▼
models/model_pymc.py         models/model_brms.R
            │                           │
forecasts/pymc_<ts>.json     forecasts/brms_<ts>.json
            └─────────────┬─────────────┘
                          ▼
     models/compare.py  (al cierre de la 1ra vuelta)
                          │
                          ▼
           forecasts/comparison_report.md
```

## Setup

### Python (scraper + PyMC)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### R (brms)
```r
install.packages(c("brms", "tidyverse", "posterior", "jsonlite", "cmdstanr"))
cmdstanr::install_cmdstan()
```

## Correr

```bash
# 1) Snapshot + baseline + join → panel
python3 prep.py --out data/district_panel.csv

# 2a) Forecast PyMC
python3 models/model_pymc.py --panel data/district_panel.csv

# 2b) Forecast brms
Rscript models/model_brms.R data/district_panel.csv forecasts

# Automatizado en vivo (refresca cada 15 min, re-ajusta ambos modelos)
python3 onpe_distritos.py --watch 15    # en una terminal
# y en otra, cron/loop manual:
while true; do
  python3 prep.py && python3 models/model_pymc.py && \
  Rscript models/model_brms.R && sleep 900
done

# 3) Al cerrar la primera vuelta
python3 models/compare.py --final data/final_resultados.json
```

## Modelo

Para cada candidato *k* ∈ {Sánchez, LA}, por distrito *d*:

```
y_{k,d} | n_d ~ Binomial(n_d, p_{k,d})
logit(p_{k,d}) = α_{k, dep[d]} + β_k · logit(pct_baseline_{k,d}) + ε_{k,d}
α_{k,dep} ~ Normal(μ_k, σ_{k,dep})
ε_{k,d}   ~ Normal(0, σ_{k,dist})
```

**Baselines 2021:**
- Sánchez → Juntos por el Perú (Mendoza 2021) — proxy razonable, misma coalición.
- López Aliaga → Renovación Popular 2021 (mismo candidato).

**Actas pendientes:** se proyecta `n_pend_d ≈ actas_pendientes × (votos/acta vistos)`.
Muestreo binomial posterior → distribución de votos finales nacionales.

## Qué mide cada modelo

| Output | PyMC | brms |
|---|---|---|
| Mediana + CI 95% de votos finales | ✅ | ✅ |
| P(Sánchez > LA) | ✅ | ✅ |
| R̂ del sampler | ✅ | ✅ |
| Backend | NUTS propio | Stan (cmdstanr) |
| Tiempo típico nacional | ~2 min | ~3 min |

Ambos modelos son matemáticamente equivalentes; las diferencias que aparezcan
serán por priors, parametrización interna y sampler. Al comparar contra el
resultado final (`compare.py`) vemos cuál caló mejor la incertidumbre real.

## Limitaciones conocidas

1. **Baseline 2021 ≠ 2026 perfecto.** Sánchez no es Mendoza; usarlo como
   covariable asume persistencia geográfica del voto de izquierda. Útil pero
   imperfecto.
2. **Sesgo de llegada de actas pendientes:** el modelo asume que las actas
   pendientes del distrito *d* se comportan como las contadas de *d*. Si
   ONPE procesa primero zonas urbanas y deja rurales al final, aparece sesgo
   residual.
3. **Votos en el extranjero** (`idAmbitoGeografico=2`) no están en el panel;
   son ~3-4% del total y habría que modelarlos aparte.
4. **Actas observadas por el JEE** pueden anularse o validarse — aumenta
   varianza que el modelo no captura explícitamente.

## Licencia
MIT (código). Los datos de ONPE son públicos.
