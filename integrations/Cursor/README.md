# Cursor × openzoo

Cursor (AI code editor) supports custom OpenAI-compatible endpoints through its
"Override OpenAI Base URL" setting. Pure UI configuration — no file to install,
so this folder is documentation only.

## Setup

1. Start the openzoo proxy: `npx openzoo` (or use a hosted endpoint).
2. Cursor Settings (`Cmd/Ctrl+Shift+J`) → **Models**.
3. Disable pre-enabled Cursor-hosted models to avoid routing conflicts.
4. In the OpenAI section, enable **Override OpenAI Base URL** and enter:
   `http://localhost:8402/v1`
   (hosted: `https://x402-tokens.fly.dev/v1`) — include the `/v1` suffix; Cursor
   appends `/chat/completions` itself.
5. **OpenAI API Key**: `sk-openzoo` (any value for local; real key for hosted).
   Despite the "OpenAI" label, this key is sent to the overridden endpoint.
6. Click **+ Add Model** and type a zoo model id, e.g. `nvidia/nemotron-3.5-lightning`
   (`curl localhost:8402/v1/models` lists ids). Verify/Save — Cursor sends a test request.
7. Pick the zoo model in the chat sidebar (`Cmd/Ctrl+L`).

## Known limitations (Cursor-side, not openzoo-side)

- **BYOK requires Cursor Pro.** The Hobby tier can't override the base URL at all
  (confirmed by openzoo's own docs).
- The override applies to the AI panel (plan + agent mode). **Tab autocomplete
  and inline edit (`Cmd/Ctrl+K`) stay on Cursor's own backend** and will not
  route through openzoo. For full routing use a terminal agent (aider, Cline,
  Continue, Hermes — see sibling folders).
- Only one override URL can be active at a time.
- Your endpoint must support `stream: true` SSE responses (the openzoo proxy does).
- Localhost overrides can be finicky in some Cursor versions (requests may
  originate from Cursor's servers rather than your machine). If Verify fails
  against localhost, the hosted endpoint is the reliable path for Cursor.

## Full platform access: add the MCP server too

The chat override above covers ordinary completions. openzoo's biggest capability —
`zoo_ask`, which takes a corpus of up to ~9.8M tokens per call (a body models refuse directly)
and answers via leCore memory spill — is an **MCP tool**, and Cursor supports MCP.
Settings → MCP → *Add new global MCP server*, or drop into `~/.cursor/mcp.json`
(per-project: `.cursor/mcp.json`):

```json
{ "mcpServers": { "openzoo": { "command": "npx", "args": ["-y", "openzoo", "mcp"] } } }
```

This exposes `zoo_ask` (giant-corpus Q&A with per-call receipt), `zoo_models`
(model list + pricing, free), and `zoo_wallet` (funding address, balances,
session receipts) to Cursor's agent. Same wallet as the proxy — fund once.

## Troubleshooting

- "Invalid API Key" → key field empty; any non-empty value works for the proxy.
- "Model not found" → model id must exactly match an id from `/v1/models`.
- 402 → burner wallet unfunded: `npx openzoo address` / `npx openzoo balance`.
