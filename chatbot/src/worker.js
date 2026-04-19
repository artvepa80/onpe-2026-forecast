// Cloudflare Worker — Chatbot para ONPE 2026 Forecast
// Lee nuestros JSONs del repo público, los inyecta como contexto, y usa
// Workers AI (Llama 3.1 8B) para responder preguntas del usuario.
//
// Deploy: `npx wrangler deploy`
// Endpoint: POST /chat  { question: string, history?: [] }

const REPO_RAW = "https://raw.githubusercontent.com/artvepa80/onpe-2026-forecast/main";

const ALLOWED_ORIGINS = [
  "https://artvepa80.github.io",
  "http://localhost:8080",
  "http://localhost:3000",
];

function cors(origin) {
  const ok = ALLOWED_ORIGINS.includes(origin) ? origin : ALLOWED_ORIGINS[0];
  return {
    "Access-Control-Allow-Origin": ok,
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Max-Age": "86400",
  };
}

async function getContext() {
  // Cache por 5 min — evita re-fetchear en cada request
  const [pymc, anomalias] = await Promise.all([
    fetch(`${REPO_RAW}/docs/data/pymc_latest.json`, {
      cf: { cacheTtl: 300, cacheEverything: true },
    }).then(r => r.json()).catch(() => null),
    fetch(`${REPO_RAW}/data/anomalias_summary.json`, {
      cf: { cacheTtl: 300, cacheEverything: true },
    }).then(r => r.json()).catch(() => null),
  ]);
  return { pymc, anomalias };
}

function buildSystemPrompt(ctx) {
  const p = ctx.pymc?.prediccion ?? {};
  const tax = ctx.pymc?.jee_taxonomia ?? {};
  const sampler = ctx.pymc?.sampler ?? {};
  const gamma_s = ctx.pymc?.econometria?.gamma?.Sanchez ?? {};
  const gamma_la = ctx.pymc?.econometria?.gamma?.LA ?? {};
  const jee_dep = ctx.pymc?.jee_por_departamento ?? {};

  return `Eres un asistente de análisis electoral para el proyecto ONPE 2026 Forecast.
Respondes SOLO basándote en los datos provistos abajo. Si te preguntan algo que
no está en los datos, di: "No tengo ese dato en el modelo". No especules sobre
opiniones políticas.

# CONTEXTO DEL MODELO (actualizado: ${ctx.pymc?.timestamp ?? "desconocido"})

## Predicción — Lucha por 2° puesto 1ra vuelta
- Roberto Sánchez (Juntos por el Perú) vs Rafael López Aliaga (Renovación Popular)
- Sánchez proyectado: ${p.Sanchez?.mediana?.toLocaleString?.() ?? "?"} votos (CI95: ${p.Sanchez?.ci_low?.toLocaleString?.()}–${p.Sanchez?.ci_high?.toLocaleString?.()})
- LA proyectado: ${p.LA?.mediana?.toLocaleString?.()} votos (CI95: ${p.LA?.ci_low?.toLocaleString?.()}–${p.LA?.ci_high?.toLocaleString?.()})
- Diferencia: +${p.diferencia_Sanchez_minus_LA?.mediana?.toLocaleString?.()} a favor de Sánchez
- P(Sánchez > LA) = ${p["P(Sanchez > LA)"]}

## Conteo actual ONPE
- Votos contados Sánchez: ${ctx.pymc?.votos_contados_Sanchez?.toLocaleString?.()}
- Votos contados LA: ${ctx.pymc?.votos_contados_LA?.toLocaleString?.()}
- Pendientes JEE (observadas): ${ctx.pymc?.votos_pend_jee?.toLocaleString?.()}
- Pendientes normales: ${ctx.pymc?.votos_pend_normal?.toLocaleString?.()}

## Taxonomía de actas observadas (5,104 totales)
Razones de observación:
${(tax.razones_top ?? []).slice(0,6).map(([r,n]) => `  - ${r}: ${n} actas`).join("\n")}

Locales con más observaciones:
${(tax.locales_top ?? []).slice(0,5).map(l => `  - ${l.local} (${l.distrito}): ${l.actas_obs} actas`).join("\n")}

Anomalías:
  - Actas "flash" (<30s de procesamiento): ${tax.flash_lt_30s ?? 0}
  - Actas procesadas 2-5 AM Lima: ${tax.nocturno_2_5am ?? 0}

## JEE por departamento (top 5)
${Object.entries(jee_dep).slice(0,5).map(([d,n]) => `  - ${d}: ${n} actas`).join("\n")}

## Coeficientes econométricos (γ estandarizados, Capa 1 INEI)
Sánchez — covariables que mueven su voto:
  - Vulnerabilidad alimentaria: ${gamma_s.vuln_alim_z?.mediana?.toFixed(3)} (certeza ${(gamma_s.vuln_alim_z?.sign_cert*100).toFixed(0)}%)
  - % Pobreza: ${gamma_s.pobreza_z?.mediana?.toFixed(3)} (certeza ${(gamma_s.pobreza_z?.sign_cert*100).toFixed(0)}%)
  - IDH: ${gamma_s.idh_z?.mediana?.toFixed(3)} (certeza ${(gamma_s.idh_z?.sign_cert*100).toFixed(0)}%)
  - log Densidad: ${gamma_s.log_densidad_z?.mediana?.toFixed(3)} (certeza ${(gamma_s.log_densidad_z?.sign_cert*100).toFixed(0)}%)

López Aliaga — covariables:
  - IDH: ${gamma_la.idh_z?.mediana?.toFixed(3)} (certeza ${(gamma_la.idh_z?.sign_cert*100).toFixed(0)}%)
  - log Densidad: ${gamma_la.log_densidad_z?.mediana?.toFixed(3)} (certeza ${(gamma_la.log_densidad_z?.sign_cert*100).toFixed(0)}%)
  - % Pobreza: ${gamma_la.pobreza_z?.mediana?.toFixed(3)} (certeza ${(gamma_la.pobreza_z?.sign_cert*100).toFixed(0)}%)

## Calidad del modelo
- rhat Sánchez: ${sampler.rhat_max_sanchez?.toFixed(3)}
- rhat LA: ${sampler.rhat_max_la?.toFixed(3)}
- Nota: P(Sánchez>LA)=1.00 es artefacto del posterior tail. La confianza honesta es ~97-99%.

## Reglas de respuesta
- Responde en español peruano, tono claro y directo.
- Usa Markdown para tablas y listas cuando ayude.
- Cita los números del contexto (no inventes cifras).
- Si te piden opinión política, di "el modelo no opina, solo proyecta con datos".
- Sé conciso: 3-6 oraciones típicamente. Expande solo si preguntan detalle.
- El modelo predice 1ra vuelta; para 2da vuelta (7 junio) aún no hay forecast construido.`;
}

async function chat(request, env) {
  let body;
  try { body = await request.json(); } catch { return new Response("bad json", {status: 400}); }
  const question = (body.question || "").slice(0, 1000);
  const history = (body.history || []).slice(-8);   // últimas 4 vueltas

  if (!question) return new Response(JSON.stringify({error: "empty question"}), {status: 400});

  const ctx = await getContext();
  if (!ctx.pymc) {
    return new Response(JSON.stringify({
      error: "No se pudo cargar el modelo. Intenta en unos segundos.",
    }), {status: 503});
  }
  const system = buildSystemPrompt(ctx);

  const messages = [
    {role: "system", content: system},
    ...history,
    {role: "user", content: question},
  ];

  const resp = await env.AI.run("@cf/meta/llama-3.1-8b-instruct", {
    messages,
    max_tokens: 512,
    temperature: 0.3,
  });

  return new Response(JSON.stringify({
    answer: resp.response || "(sin respuesta)",
    model_updated: ctx.pymc?.timestamp,
  }), {
    headers: {"Content-Type": "application/json"},
  });
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const origin = request.headers.get("Origin") || "";
    const corsHeaders = cors(origin);

    if (request.method === "OPTIONS") {
      return new Response(null, {headers: corsHeaders});
    }

    if (url.pathname === "/chat" && request.method === "POST") {
      const r = await chat(request, env);
      for (const [k, v] of Object.entries(corsHeaders)) r.headers.set(k, v);
      return r;
    }

    if (url.pathname === "/") {
      return new Response("ONPE 2026 Chatbot — POST /chat { question }", {
        headers: {"Content-Type": "text/plain", ...corsHeaders},
      });
    }

    return new Response("not found", {status: 404, headers: corsHeaders});
  },
};
