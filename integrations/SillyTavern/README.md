# SillyTavern × openzoo

SillyTavern connects to any OpenAI-compatible server through its built-in
"Custom (OpenAI-compatible)" chat completion source. The integration is pure UI
configuration — there is no file to install, so this folder is documentation only.

## Setup

1. Start the openzoo proxy: `npx openzoo`.
2. In SillyTavern, click the plug icon (API Connections).
3. Set **API** → `Chat Completion`.
4. Set **Chat Completion Source** → `Custom (OpenAI-compatible)`.
5. **Custom Endpoint**: `http://localhost:8402/v1`
   (hosted: `https://api.openzoo.fun/v1`)
   — stop at `/v1`; do NOT append `/chat/completions`, SillyTavern adds the route.
6. **API Key**: `sk-openzoo` (any value for the local proxy; real key for hosted).
7. Connect. Because the proxy implements `/v1/models`, the model dropdown fills
   with the live zoo list — pick one and chat.

## Notes

- Roleplay chats grow long; the zoo's leCore large-context handling means old
  history keeps working where direct APIs would truncate or refuse.
- 402 error = burner wallet unfunded: `npx openzoo address` / `npx openzoo balance`.
