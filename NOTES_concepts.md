
## integrations/ folder: openzoo harness integrations + audit sweep (session)
Built integrations/ (10 harnesses: OpenWebUI pipe function, LibreChat yaml, Continue yaml,
aider env, SillyTavern/AnythingLLM/Cursor/Cline README-only, Hermes yaml, GrokCLI json).
Registered pointer capability "Harness integrations for openzoo" in catalog p06 (import-only
declared negative: integrations never import lecore; HTTP to localhost:8402/v1 only). Battery 5/5.
BUGS FOUND & FIXED in sweep:
- openzoo model ids are provider-prefixed AND contain dots (nvidia/nemotron-3.5-lightning);
  original placeholder deepseek-v4-flash would 404 everywhere. Fixed in 10 files.
- OpenWebUI pipe _strip_owui_prefix split on first "." -- mangled bare dotted ids to
  "5-lightning". Fixed with dot-before-slash + digitless-head guards; pinned in selftest
  as a kept negative.
- Pipe forwarded OpenWebUI bookkeeping keys (chat_id/metadata/...) upstream; now allowlisted
  to OpenAI chat fields. Round-trip tested against a mock server (models fetch, non-stream +
  receipt, SSE stream, 402 guidance) -- all pass.
- Literal "{a,b}" junk dirs from failed brace expansion caught BY RUNNING the catalog example.
KEPT NEGATIVE: no client-side corpus spill in any integration -- spill is server-side at the
zoo so all harnesses benefit and nothing double-bills. NOT LIVE-VERIFIED: a settled paid call
(no funded wallet in this environment); receipt-in-usage shape unconfirmed -- pipe degrades
gracefully either way.
