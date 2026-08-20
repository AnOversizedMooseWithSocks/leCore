# aider × openzoo

aider reads the standard `OPENAI_API_BASE` / `OPENAI_API_KEY` environment variables
and routes any `openai/`-prefixed model through them. No plugin exists or is needed.

## Setup

```bash
npx openzoo                                    # start the paying proxy (once)
source integrations/aider/openzoo.env          # sets the two variables
aider --model openai/nvidia/nemotron-3.5-lightning         # any zoo model id works
```

`npx openzoo models` lists model ids and pricing.

## Persistent config (optional)

Instead of env vars, add to `~/.aider.conf.yml`:

```yaml
openai-api-base: http://localhost:8402/v1
openai-api-key: sk-openzoo
model: openai/nvidia/nemotron-3.5-lightning
```

## Notes

- aider may warn about unknown model metadata (context window, cost) for zoo ids;
  it still works. Silence it with a `.aider.model.metadata.json` if it bothers you.
- Hosted endpoint: swap the base URL and use a real key.
