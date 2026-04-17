#!/usr/bin/env Rscript
# Modelo Bayesiano jerárquico en brms — réplica del modelo PyMC para validación cruzada.
#
# Para cada candidato k ∈ {Sánchez, LA}:
#   votos_k_d | n_d ~ Binomial(n_d, p_{k,d})
#   logit(p_{k,d}) = α_{k, dep[d]} + β_k · logit(pct_baseline_{k,d}) + ε_{k,d}
#
# Uso:
#   Rscript models/model_brms.R [panel.csv] [out_dir]
#
# Requiere: brms, tidyverse, posterior, jsonlite
#
suppressPackageStartupMessages({
  library(tidyverse)
  library(brms)
  library(posterior)
  library(jsonlite)
})

args     <- commandArgs(trailingOnly = TRUE)
panel    <- if (length(args) >= 1) args[1] else "data/district_panel.csv"
out_dir  <- if (length(args) >= 2) args[2] else "forecasts"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

logit_safe <- function(p, eps = 1e-3) {
  p <- pmin(pmax(p, eps), 1 - eps)
  log(p / (1 - p))
}

df <- read_csv(panel, show_col_types = FALSE) |>
  filter(!is.na(pct_RP_2021), !is.na(pct_JpP_2021),
         votos_validos_contados > 0) |>
  mutate(
    pct_RP_2021   = pmax(pct_RP_2021, 0.01) / 100,
    pct_JpP_2021  = pmax(pct_JpP_2021, 0.01) / 100,
    logit_RP      = logit_safe(pct_RP_2021),
    logit_JpP     = logit_safe(pct_JpP_2021),
    votos_por_acta = votos_validos_contados / pmax(actas_contab, 1),
    n_pend_est    = round(actas_pendientes * votos_por_acta)
  )

cat(sprintf("Panel: %d distritos\n", nrow(df)))

fit_candidate <- function(df, y_col, x_col, seed = 42) {
  data_fit <- df |>
    transmute(y = !!sym(y_col),
              n = votos_validos_contados,
              x = !!sym(x_col),
              dep = factor(ubigeo_dep))
  fit <- brm(
    bf(y | trials(n) ~ x + (1 | dep)),
    data = data_fit, family = binomial("logit"),
    prior = c(prior(normal(0, 2.5), class = "Intercept"),
              prior(normal(1, 1),   class = "b"),
              prior(exponential(1), class = "sd")),
    chains = 2, iter = 2000, warmup = 1000,
    seed = seed, refresh = 0,
    backend = "cmdstanr", silent = 2
  )
  list(fit = fit, data = data_fit)
}

posterior_final <- function(mod, df, y_col, seed = 123) {
  # posterior_epred: expected p (ya en probabilidad) por fila x draw
  p_mat <- posterior_epred(mod$fit, newdata = mod$data |> mutate(n = 1L), ndraws = 2000)
  # p_mat: draws x rows
  set.seed(seed)
  n_pend <- df$n_pend_est
  y_counted <- df[[y_col]]
  # Sample binomial draws for pending actas, per district per draw
  y_pend <- matrix(rbinom(length(p_mat), size = rep(n_pend, each = nrow(p_mat)), prob = as.vector(p_mat)),
                   nrow = nrow(p_mat))
  total <- rowSums(sweep(y_pend, 2, y_counted, "+"))
  total
}

cat("Ajustando Sánchez...\n")
mod_s  <- fit_candidate(df, "votos_Sanchez", "logit_JpP", seed = 42)
cat("Ajustando López Aliaga...\n")
mod_la <- fit_candidate(df, "votos_LA",      "logit_RP",  seed = 43)

votes_s  <- posterior_final(mod_s,  df, "votos_Sanchez", seed = 111)
votes_la <- posterior_final(mod_la, df, "votos_LA",      seed = 222)

q <- function(v) {
  qq <- quantile(v, c(0.025, 0.5, 0.975))
  list(media = mean(v), mediana = qq[["50%"]], ci_low = qq[["2.5%"]], ci_high = qq[["97.5%"]])
}

lead <- votes_s - votes_la
lead_q <- quantile(lead, c(0.025, 0.5, 0.975))

report <- list(
  modelo = "brms_binomial_jerarquico",
  timestamp = format(Sys.time(), "%Y-%m-%dT%H:%M:%S"),
  snapshot_panel = basename(panel),
  n_distritos_usados = nrow(df),
  votos_contados_Sanchez = sum(df$votos_Sanchez),
  votos_contados_LA = sum(df$votos_LA),
  votos_pendientes_estimados = sum(df$n_pend_est),
  prediccion = list(
    Sanchez = c(candidato = "Sanchez", q(votes_s)),
    LA      = c(candidato = "LA",      q(votes_la)),
    diferencia_Sanchez_minus_LA = list(
      mediana = lead_q[["50%"]], ci_low = lead_q[["2.5%"]], ci_high = lead_q[["97.5%"]]
    ),
    `P(Sanchez > LA)` = mean(votes_s > votes_la)
  )
)

ts <- format(Sys.time(), "%Y%m%d_%H%M")
out_path <- file.path(out_dir, paste0("brms_", ts, ".json"))
write_json(report, out_path, pretty = TRUE, auto_unbox = TRUE)
write_json(report, file.path(out_dir, "brms_latest.json"), pretty = TRUE, auto_unbox = TRUE)
cat(sprintf("\nGuardado: %s\n", out_path))
print(toJSON(report$prediccion, pretty = TRUE, auto_unbox = TRUE))
