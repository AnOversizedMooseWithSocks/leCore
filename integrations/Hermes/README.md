# Hermes Agent × openzoo

Hermes Agent (Nous Research's open-source, model-agnostic agent harness — CLI plus
messaging-platform gateway, persistent memory, skills) works with any
OpenAI-compatible provider via its `custom` provider type.

## Setup

Easiest: run `npx openzoo` (the paying proxy), then `hermes model`, choose
**Custom endpoint**, and enter:

- Base URL: `http://localhost:8402/v1` (stop at `/v1` — Hermes appends the route)
- API key: `sk-openzoo` (any value for the local proxy; real key for hosted)
- Model: any zoo id (`curl localhost:8402/v1/models` lists them)

Hermes saves the selection to `~/.hermes/config.yaml`. `config.openzoo.yaml` in
this folder is the equivalent manual merge if you prefer editing config directly.

## Notes

- Hermes is a long-running agent with big context accumulation (memory, skills,
  tool loops) — exactly the workload where openzoo's leCore large-context handling
  and per-call receipts pay off.
- `provider: custom` is for the main model. Auxiliary tasks (vision, compression)
  are configured separately in Hermes and can point at different providers.
- 402 error = burner wallet unfunded: `npx openzoo address` / `npx openzoo balance`.
- Grok note: Hermes also ships native xAI/SuperGrok OAuth support. openzoo and
  Grok aren't exclusive — Hermes users can set Grok as main and a zoo model as a
  fallback provider, or vice versa, via `hermes fallback`.
