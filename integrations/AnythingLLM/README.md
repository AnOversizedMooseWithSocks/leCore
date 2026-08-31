# AnythingLLM × openzoo

AnythingLLM (all-in-one document chat / RAG workspace app) supports a
"Generic OpenAI" LLM provider in its settings UI. The integration is pure
configuration — no file to install, so this folder is documentation only.

## Setup

1. Start the openzoo proxy: `npx openzoo`.
2. In AnythingLLM: **Settings → AI Providers → LLM** → choose **Generic OpenAI**.
3. **Base URL**: `http://localhost:8402/v1` (hosted: `https://api.openzoo.fun/v1`)
4. **API Key**: `sk-openzoo` (any value for local; real key for hosted)
5. **Chat Model Name**: a zoo model id, e.g. `nvidia/nemotron-3.5-lightning`
   (`npx openzoo models` lists ids and pricing)
6. Set token context window / max tokens as desired, save, and chat.

## Notes

- AnythingLLM's own RAG chunks documents before the model sees them; the zoo's
  leCore large-context path is complementary — oversized prompts that slip
  through still work instead of erroring.
- 402 error = burner wallet unfunded: `npx openzoo address` / `npx openzoo balance`.
