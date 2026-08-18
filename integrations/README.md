# integrations/

Plugins, adapters, and glue code that surface leCore (or leCore-backed services such
as openzoo) inside third-party applications. This directory is the home for anything
whose *runtime host is another app* — code here is loaded by that app, not imported
by the leCore engine.

## Structure

One subfolder per target application, named after the app:

```
integrations/
├── README.md                      ← this file
├── OpenWebUI/                     ← self-hosted AI chat harness (plugin: Pipe function)
│   ├── README.md
│   └── openzoo_pipe.py            ← manifold Pipe: openzoo as a model provider
├── LibreChat/                     ← multi-user AI chat platform (config: yaml)
│   ├── README.md
│   └── librechat.openzoo.yaml     ← ready-to-merge custom endpoint block
├── Continue/                      ← AI coding assistant, VS Code/JetBrains (config: yaml)
│   ├── README.md
│   └── config.openzoo.yaml        ← ready-to-merge models entries
├── aider/                         ← terminal AI pair programmer (config: env)
│   ├── README.md
│   └── openzoo.env                ← source-able OPENAI_API_BASE/KEY exports
├── SillyTavern/                   ← LLM chat frontend (UI-config only; README documents it)
│   └── README.md
├── AnythingLLM/                   ← document chat / RAG app (UI-config only; README documents it)
│   └── README.md
├── Hermes/                        ← Nous Research agent harness (config: yaml)
│   ├── README.md
│   └── config.openzoo.yaml        ← provider: custom block; `hermes model` interactive alt
├── Cursor/                        ← AI code editor (UI-config only; README documents it)
│   └── README.md                  ← incl. known limits: autocomplete stays on Cursor's backend
├── Cline/                         ← VS Code/JetBrains coding agent (UI-config only; README)
│   └── README.md                  ← also covers Roo Code (same provider form)
└── GrokCLI/                       ← superagent-ai grok-cli terminal agent (config: json/env)
    ├── README.md
    └── models.openzoo.json        ← provider entry for ~/.grok/models.json
```

An integration takes whatever the smallest sufficient form is: a real plugin file
(OpenWebUI), a mergeable config snippet (LibreChat, Continue, aider), or — when the
host app is configured entirely through its UI — a README alone (SillyTavern,
AnythingLLM). A README-only folder is a legitimate integration; the folder existing
is what makes it discoverable.

Future subfolders that belong here: `ComfyUI/` (if the node pack ever moves
in-repo), `Cursor/`, `Cline/`, `Slack/` — always the app's own name, matching its
official capitalization.

## The openzoo platform: two surfaces, one wallet

Every integration here targets one or both of openzoo's surfaces:

1. **OpenAI-compatible chat proxy** — `npx openzoo` → `http://localhost:8402/v1`.
   Model ids are provider-prefixed (e.g. `nvidia/nemotron-3.5-lightning`) and come
   from `GET /v1/models` (free, no payment). Streaming (SSE) is passed through
   unbuffered. Oversized bodies are priced at a counterfactual discount because
   the zoo's leCore memory spills them server-side (~10× cheaper than direct);
   short prompts price at a 3× passthrough markup — receipts name which base
   applied, and print on the proxy console per call.
2. **MCP server** — `npx openzoo mcp` (stdio). Tools: `zoo_ask` (corpus up to
   ~9.8M tokens per call (~128M bound ceiling) + question → answer + receipt), `zoo_models`, `zoo_wallet`. MCP
   hosts (Cursor, Cline, Claude Desktop, Windsurf) should wire BOTH surfaces —
   chat for ordinary completions, MCP for the giant-corpus flagship.

Both surfaces share the burner wallet at `~/.openzoo/wallet.json`. Spend safety:
the proxy refuses any single quote above `OPENZOO_MAX_USD_PER_CALL` (default
$0.50). Fund with plain USDC; `npx openzoo address` / `npx openzoo balance`.

## Rules for this directory

These differ deliberately from the core engine rules, because the host app — not
leCore — dictates the environment:

1. **Host conventions win.** A file here follows the target app's plugin format
   (frontmatter, class shapes, required method names) even where that conflicts
   with leCore style. E.g. OpenWebUI functions require `pydantic` — acceptable
   here, never in core.
2. **Core constraints still bleed through where possible.** WHY-comments, a
   `_selftest()` that runs the pure logic without a network, and kept negatives
   recorded in comments. An integration file should still read like leCore code.
3. **No engine imports.** Integrations talk to leCore over HTTP (`/invoke`, the
   OpenAI facade) or to openzoo's endpoint — never `import lecore`. This keeps
   them installable inside the host app with zero leCore install.
4. **Each subfolder is self-documenting.** Every app folder carries its own
   `README.md` with install steps, configuration, and troubleshooting. A user
   should be able to land in `integrations/<App>/` and succeed without reading
   anything else in the repo.
5. **Not part of the test suite or the wheel.** Nothing here ships in the
   `leos-core` PyPI package, and CI does not import it (host-app dependencies
   aren't installed there). Each file's `python3 <file>` selftest is the
   verification contract.

## Why a separate top-level folder

`tools/` is for scripts *we* run against the repo; `holographic/` is the engine;
`integrations/` is code *other apps* run. Keeping the boundary hard prevents the
engine from ever growing a dependency on a host app's SDK, and makes it obvious at
a glance what is safe to vendor elsewhere.
