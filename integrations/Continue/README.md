# Continue.dev × openzoo

Continue (the open-source AI coding assistant for VS Code/JetBrains) talks to any
OpenAI-compatible endpoint via its built-in `provider: openai` — no plugin required.
`config.openzoo.yaml` is a ready-to-merge snippet.

## Setup

1. Start the openzoo proxy: `npx openzoo` (or use a hosted endpoint URL).
2. Merge the `models` entries from `config.openzoo.yaml` into `~/.continue/config.yaml`.
3. Reload Continue. Zoo models appear in the model picker.

## Notes

- Continue does not fetch a live model list; each zoo model you want is one entry.
  `curl localhost:8402/v1/models` prints available model ids and pricing.
- `apiKey` is ignored by the local proxy (x402 pays); required-but-arbitrary. For a
  hosted endpoint, use your real key.
- Coding sessions can push large repo context; the zoo's leCore corpus handling is
  exactly the case where per-call receipts show real savings vs direct.
