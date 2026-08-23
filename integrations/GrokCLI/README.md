# Grok CLI × openzoo

Grok CLI (superagent-ai/grok-cli — the open-source terminal coding agent built
for the Grok API) speaks the OpenAI chat-completions format and accepts a custom
base URL, so it can route through openzoo. This covers "Grok-ecosystem" users:
they keep their harness and gain the zoo's 480+ models and leCore large-context
handling.

## Setup

Quickest — flags or env vars, no file edits:

```bash
npx openzoo                                   # start the paying proxy (once)
grok --base-url http://localhost:8402/v1 \
     --api-key sk-openzoo \
     --model nvidia/nemotron-3.5-lightning
# or:
export GROK_BASE_URL=http://localhost:8402/v1
export GROK_API_KEY=sk-openzoo
grok --model nvidia/nemotron-3.5-lightning
```

Persistent — add the provider entry from `models.openzoo.json` in this folder to
`~/.grok/models.json` (append to the existing array; add more zoo ids to
`models` as desired — `curl localhost:8402/v1/models` lists them). Per-project model
pinning lives in `.grok/settings.json`.

## Notes

- Grok CLI's settings resolution: env vars > project settings > user settings.
  If `GROK_API_KEY`/`GROK_BASE_URL` are exported, they win over models.json.
- Some grok-cli versions have had models.json quirks (see upstream issue #73);
  the env-var route is the dependable fallback.
- Hosted endpoint: swap the base URL and use a real key.
- 402 = burner wallet unfunded: `npx openzoo address` / `npx openzoo balance`.

## Related Grok-ecosystem paths

- **Hermes Agent** (see `../Hermes/`) ships native xAI/SuperGrok OAuth alongside
  its custom-endpoint support — the natural home for users who want Grok *and*
  zoo models in one harness with fallback between them.
- Grok's own apps (grok.com, the X integration) are closed surfaces with no
  custom-endpoint hook; there is nothing to integrate there.
