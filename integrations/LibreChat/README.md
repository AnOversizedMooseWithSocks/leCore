# LibreChat × openzoo

LibreChat supports any OpenAI-compatible service as a custom endpoint configured in
`librechat.yaml`. `librechat.openzoo.yaml` in this folder is a ready-to-merge snippet
that adds openzoo to the endpoint dropdown with live model fetching.

## Setup

1. Start the openzoo proxy: `npx openzoo` (funds via x402 burner wallet; see
   `../OpenWebUI/README.md` for wallet funding). Skip this if using a hosted endpoint.
2. Merge the `endpoints.custom` entry from `librechat.openzoo.yaml` into your
   `librechat.yaml` (project root, same directory as `.env`).
3. Docker installs: ensure `librechat.yaml` is mounted via
   `docker-compose.override.yml` (bind `./librechat.yaml` → `/app/librechat.yaml`).
4. Restart LibreChat. "openzoo" appears in the endpoint selector; models are fetched
   live from `/v1/models`.

## Hosted endpoint

Change `baseURL` to the hosted URL, put your key in `.env` as `OPENZOO_KEY=...`,
and set `apiKey: "${OPENZOO_KEY}"`.

## Notes

- LibreChat requires `apiKey` to be non-empty even when the endpoint ignores it
  (the local proxy pays via x402, not keys) — hence the `sk-openzoo` placeholder.
- A 402 response means the request reached the proxy but the burner wallet is
  unfunded: `npx openzoo address` / `npx openzoo balance`.
- Validate your merged YAML (yamlchecker.com) — YAML syntax errors are the most
  common failure and LibreChat will log a validation error at startup.
