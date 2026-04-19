# Chatbot ONPE 2026 — Cloudflare Worker

Chatbot sobre los datos del forecast, usando **Cloudflare Workers AI** (Llama 3.1 8B) —
gratis hasta 10K invocaciones/día.

## Arquitectura

```
Browser (docs/index.html)  ─→  POST /chat
                                   │
                                   ▼
                         onpe-forecast-chat.workers.dev
                                   │
                                   ├─ fetch forecasts/pymc_latest.json
                                   ├─ fetch data/anomalias_summary.json
                                   └─ llama-3.1-8b-instruct (Workers AI)
```

Contexto inyectado en cada pregunta:
- Predicción actual + CI 95%
- Conteo ONPE vivo
- Taxonomía JEE (razones, locales, anomalías)
- Coeficientes econométricos
- Top 5 departamentos JEE

## Deploy local (primera vez)

```bash
cd chatbot
npm install
npx wrangler login           # abre browser, login con tu cuenta Cloudflare
npx wrangler deploy           # publica el worker
```

Tras deploy el Worker queda en `https://onpe-forecast-chat.<tu-subdomain>.workers.dev`.

Copiar ese URL al frontend en `docs/index.html` (constante `CHAT_API`).

## Costos

- Workers: free tier **100,000 requests/día**
- Workers AI: free tier **10,000 invocaciones/día** (Llama 3.1 8B)
- Para ~500 chats/mes: **$0**

Si escalás mucho:
- Workers AI paid tier: ~$0.011 por 1K requests (Llama 8B)
- Haiku via API externa (mejor calidad): ~$0.001 por mensaje

## Test

```bash
curl -X POST https://onpe-forecast-chat.<tu-subdomain>.workers.dev/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"¿Quién gana el 2° puesto?"}'
```

## Prompt del sistema

El prompt se reconstruye en **cada request** leyendo el repo público. Ventajas:
- Sin necesidad de redeploy cuando cambian los datos
- Siempre refleja el último forecast
- Cacheo de 5 min para eficiencia

## Límites

- Llama 3.1 8B tiene calidad menor que Claude/GPT-4o
- Contexto total ~8K tokens (suficiente para nuestros JSONs)
- Rate limit del usuario depende del free tier de Cloudflare
