# ONPE 2026 — Forecast del 2° puesto (Sánchez vs López Aliaga)

Pipeline reproducible para (1) scrapear el portal en vivo de ONPE, (2) modelar
Bayesianamente quién termina segundo en la 1ra vuelta presidencial 2026 con
covariables socioeconómicas, y (3) comparar dos stacks (PyMC en Python vs brms
en R) contra los resultados finales.

**Dashboard en vivo:** https://artvepa80.github.io/onpe-2026-forecast/

## Flujo

```
onpe_distritos.py  ──► data/*.csv snapshots en vivo (opcional --watch)
                          │
prep.py            ──► data/district_panel.csv
   ├─ ONPE en vivo (26 deps × ~1874 distritos)
   ├─ Baseline 2021 (ONPE 1ra vuelta, RP y JpP)
   └─ Covariables INEI (altitud, IDH, pobreza, densidad, vulnerabilidad)
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

GitHub Actions: workflow "Update forecast" corre prep+PyMC cada 30 min,
commitea docs/data/pymc_latest.json → GitHub Pages autosirve el dashboard.
```

## Setup

### Python (scraper + PyMC)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### R (brms) — opcional, solo local
```r
install.packages(c("brms", "tidyverse", "posterior", "jsonlite", "cmdstanr"))
cmdstanr::install_cmdstan()
```

## Correr

```bash
# 1) Snapshot + baseline + covariables → panel
python3 prep.py --out data/district_panel.csv

# 2a) Forecast PyMC (nacional)
python3 models/model_pymc.py --panel data/district_panel.csv

# 2b) Forecast brms (nacional)
Rscript models/model_brms.R data/district_panel.csv forecasts

# Scraper en vivo (refresca snapshots cada N min)
python3 onpe_distritos.py --watch 15

# 3) Al cerrar la primera vuelta
python3 models/compare.py --final data/final_resultados.json
```

## Modelo

Para cada candidato *k* ∈ {Sánchez, LA}, por distrito *d*:

```
y_{k,d} | n_d ~ BetaBinomial(n_d, α=p·κ, β=(1-p)·κ)         ← overdispersion κ
logit(p_{k,d}) = α_{dep[d]} + β · logit(baseline_{k,d})      ← efecto 2021
                 + γ₁·altitud + γ₂·IDH + γ₃·pobreza           ← covariables
                 + γ₄·log(densidad) + γ₅·vuln_alim               económicas
                 + ε_d                                        ← ruido distrital

α_dep = μ + σ_dep · α_raw       (non-centered)
ε_d   = σ_dist · eps_raw        (non-centered)
κ ~ HalfNormal(15) + 1
γ ~ Normal(0, 0.5)               (covariables estandarizadas → coefs comparables)
```

**Baselines 2021:**
- Sánchez → Juntos por el Perú (Mendoza 2021) — proxy razonable, misma coalición.
- López Aliaga → Renovación Popular 2021 (él mismo).

**Actas pendientes:** se proyecta `n_pend_d ≈ actas_pendientes × (votos/acta vistos)`.
Posterior predictive **BetaBinomial** + un **swing nacional** `N(0, 0.08)` en logit
capturan incertidumbre de llegada (zonas tardías rinden distinto que las tempranas).

## Covariables econométricas (Capa 1)

Datos INEI por distrito (via `jmcastagnetto/ubigeo-peru-aumentado`):

| Covariable | Fuente | Transform |
|---|---|---|
| `altitud_m` | INEI | ÷ 1000 (km) + z-score |
| `idh_2019` | PNUD | z-score |
| `pct_pobreza` | INEI 2018 | ÷100 + z-score |
| `densidad_2020` | INEI | log1p + z-score |
| `vuln_alim` | INEI | z-score |

Coeficientes γ están **estandarizados** → su magnitud se interpreta como
"efecto sobre logit(voto) por 1 desviación estándar de la covariable".

## Qué mide cada modelo

| Output | PyMC | brms |
|---|---|---|
| Mediana + CI 95% de votos finales | ✅ | ✅ |
| P(Sánchez > LA) | ✅ | ✅ |
| R̂ + ESS del sampler | ✅ | ✅ |
| Coeficientes γ (covariables) | ✅ | _(falta portar)_ |
| Backend | NUTS (pytensor) | Stan (cmdstanr) |
| Tiempo típico nacional | ~3 min | ~5 min |

Ambos modelos son matemáticamente equivalentes. Al cierre de 1ra vuelta
`compare.py` evalúa: MAE en votos, cobertura del CI 95%, Brier score,
dirección correcta.

## GitHub Actions — flujo híbrido

`.github/workflows/update-forecast.yml` corre cada 30 min:
1. Checkout + Python 3.12 + cache pip
2. `pip install -r requirements.txt`
3. Intenta scrape ONPE (3 intentos con backoff 60/120/180s)
4. **Si el WAF de ONPE bloquea las IPs de GitHub runners**, cae a
   `data/district_panel.csv` committeado (modo fallback — modelo corre con
   el último panel que el usuario subió localmente)
5. `python models/model_pymc.py --draws 500 --tune 500 --chains 2`
6. Copia `forecasts/pymc_latest.json` a `docs/data/`
7. `git pull --rebase + commit + push` con retry

GitHub Pages autosirve `docs/` — el dashboard se refresca solo.

**Forzar ejecución manual:**
```bash
gh workflow run "Update forecast" --repo artvepa80/onpe-2026-forecast --ref main
```

### Refrescar panel desde tu máquina

Si el WAF de ONPE está bloqueando al runner, scrape local y pushea:
```bash
.venv/bin/python prep.py --out data/district_panel.csv
git add data/district_panel.csv
git commit -m "data: refresh panel ONPE $(date +%Y-%m-%d_%H:%M)"
git push
# El workflow programado recoge el panel fresco en el próximo run
```

## Limitaciones conocidas

1. **Baselines 2021 imperfectos.** Sánchez no es Mendoza; el β estimado en
   el modelo captura el grado de persistencia geográfica.
2. **Sesgo de llegada mitigado, no eliminado.** El national swing `N(0, 0.08)`
   añade ±1.5pp de incertidumbre, pero una llegada sistemática de zonas
   extremas (p.ej., todo el extranjero al final) no se modela explícitamente.
3. **Votos en el extranjero** (`idAmbitoGeografico=2`) no están en el panel;
   son ~3-4% del total y habría que modelarlos aparte.
4. **Actas observadas JEE** (~0.5%) pueden anularse — no modelado.
5. **Convergencia del sampler Sánchez:** rhat puede flotar en 1.10-1.20 con
   500 draws; para rhat ≤1.01 se recomienda `--draws 2000 --chains 4`
   (workflow lo hace en modo reducido por tiempo de CI).

## Próximos pasos

- **Capa 2** — econometría espacial (SAR/SEM con matriz de contigüidad)
- **Capa 3** — multinomial logit para forecast de 2da vuelta (7 junio) con
  trasvase de votos LA/Nieto/Belmont → Fujimori vs Sánchez

## Licencia
MIT (código). Los datos de ONPE y INEI son públicos.
