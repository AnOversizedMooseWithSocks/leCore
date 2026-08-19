# Cline × openzoo

Cline (autonomous coding agent for VS Code / JetBrains) has a first-class
"OpenAI Compatible" provider in its settings. Pure UI configuration — no file to
install, so this folder is documentation only.

## Setup

1. Start the openzoo proxy: `npx openzoo` (or use a hosted endpoint).
2. In Cline, click the ⚙️ settings icon.
3. **API Provider**: select `OpenAI Compatible`.
4. **Base URL**: `http://localhost:8402/v1`
   (hosted: `https://x402-tokens.fly.dev/v1`) — do NOT paste `/chat/completions`.
5. **API Key**: `sk-openzoo` (any value for local; real key for hosted).
6. **Model ID**: a zoo model id, e.g. `nvidia/nemotron-3.5-lightning`
   — must exactly match an id returned by `/v1/models` (`curl localhost:8402/v1/models`).
7. Fill the **Model Configuration** block (context window, output limits,
   whether the model supports images/computer-use). People skip this; for
   custom endpoints it matters — set the context window generously, since the
   zoo's leCore handling accepts oversized context that other providers refuse.
8. Click **Verify** to confirm the connection.

## Troubleshooting

Isolate failures by hitting the proxy directly:

```bash
curl http://localhost:8402/v1/models -H "Authorization: Bearer sk-openzoo"
```

Clean JSON back = endpoint and key are fine, the issue is in the Cline form
(usually the Model ID not matching exactly). 401/404 = re-check Base URL ends
at `/v1`. 402 = burner wallet unfunded: `npx openzoo address`.

## Full platform access: add the MCP server too

Cline is an MCP host, and openzoo's flagship capability — `zoo_ask`, giant-corpus
(up to ~1M token) Q&A via leCore memory spill — is an MCP tool, not a chat
completion. Add the server (Cline → MCP Servers → configure, stdio transport):

```json
{ "mcpServers": { "openzoo": { "command": "npx", "args": ["-y", "openzoo", "mcp"] } } }
```

Tools exposed: `zoo_ask` (corpus + question → answer + receipt: billedUsd,
savesVsDirect, tokens actually read), `zoo_models` (free), `zoo_wallet`. Same
burner wallet as the chat proxy — fund once, both surfaces pay from it.

## Notes

- Cline is agentic (reads files, runs commands) and burns context fast —
  per-call receipts from the zoo show what each tool loop actually cost.
- Roo Code (the Cline fork) has the same OpenAI Compatible provider; these
  steps apply unchanged.
