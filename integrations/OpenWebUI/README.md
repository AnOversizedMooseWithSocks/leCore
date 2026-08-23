# OpenWebUI × openzoo

Surfaces **openzoo.fun** as a full model provider inside [OpenWebUI](https://openwebui.com):
every zoo model appears in the model dropdown, chats route through openzoo, and billing
receipts (cost per call, savings vs direct) show up in the chat.

leCore is the engine underneath openzoo's large-corpus handling: oversized context is
bound into holographic memory server-side, so the upstream model reads only a few
thousand tokens. From OpenWebUI's point of view this is invisible — models just accept
inputs that would otherwise be refused.

## Files

| File | What it is |
|---|---|
| `openzoo_pipe.py` | A "manifold" Pipe function. Install into OpenWebUI (Admin Panel → Functions → `+` → paste). Lists zoo models via `/v1/models`; forwards chats to `/v1/chat/completions`; handles streaming, 402/401 errors, and receipts. |

## Setup (local proxy — works today)

1. Start the openzoo proxy: `npx openzoo`
   First run creates a burner wallet at `~/.openzoo/wallet.json` and prints a funding
   address. Fund it (USDC) — `npx openzoo balance` to check.
2. In OpenWebUI: **Admin Panel → Functions → `+`**, paste `openzoo_pipe.py`, save, enable.
3. Zoo models appear in the model dropdown as `zoo/<model>`. Chat normally.

Defaults assume the proxy at `http://localhost:8402/v1`. No API key needed — the proxy
pays per call via x402; the key field is sent but ignored.

## Setup (hosted endpoint — when available)

Open the function's **Valves** and set:

- `OPENZOO_BASE_URL` → the hosted URL (e.g. `https://api.openzoo.fun/v1`)
- `OPENZOO_API_KEY` → your real key

Nothing else changes; the same file serves both rails.

## Notes

- Model ids are provider-prefixed (e.g. `nvidia/nemotron-3.5-lightning`); the pipe
  fetches them live from `/v1/models` so the dropdown is always correct.
- Receipts: the proxy prints a one-line receipt per payment on its own console.
  The pipe additionally appends billedUsd/savings to the chat reply *when* the
  response body carries them in `usage`; if it doesn't, the chat shows nothing
  extra and the console receipt is the record. This is graceful either way.

## Troubleshooting

- **"openzoo unreachable" in the dropdown** — the proxy isn't running; start `npx openzoo`.
- **402 Payment Required** — burner wallet is unfunded; `npx openzoo address` for the funding address.
- **Timeouts on huge documents** — raise `REQUEST_TIMEOUT` in Valves.

## Verify

The pure logic (no network) self-tests:

```bash
python3 integrations/OpenWebUI/openzoo_pipe.py
```
