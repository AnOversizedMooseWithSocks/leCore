# x402 API Publishing

Yes: leCore can be published as a paid API with x402.

The implementation lives in `holographic_x402_api.py`. It wraps
`LocalAgentCore` with a small FastAPI app and applies x402 middleware only to
the public read/compute routes:

- `POST /v1/recall`
- `POST /v1/route`
- `GET /v1/dashboard`
- `POST /leos/v1/recall`, at the credential-gated leOS offer price
- `POST /leos/v1/route`, at the credential-gated leOS offer price
- `GET /leos/v1/dashboard`, at the credential-gated leOS offer price

Free routes:

- `GET /health`
- `GET /pricing`

Admin route:

- `POST /admin/remember`, guarded by `X-Admin-Token`
- `POST /admin/tenant-token`, guarded by `X-Admin-Token`

This split is deliberate. Paid customers can use the memory/router/dashboard,
but they cannot mutate memory unless they also hold the admin token. Private
tenant memory also requires a tenant token; x402 proves payment, not tenant
authorization.

The leOS CA identifies the discounted offer, but does not itself prove buyer
eligibility because it is public. Discounted calls must also include the
operator-issued `X-leCore-leOS-Access` credential. Failed authorization responses
are not settled by the x402 middleware.

## Install

```bash
pip install ".[x402]"
```

The core package still needs only NumPy. The `x402` extra pulls in the optional
FastAPI/x402/uvicorn stack.

## Testnet Run

The default network is Base Sepolia (`eip155:84532`) and the default facilitator
is the signup-free x402.org testnet facilitator.

```bash
export LECORE_X402_PAY_TO="0xYourReceivingWallet"
export LECORE_X402_PRICE="$0.0011"
export LECORE_X402_ADMIN_TOKEN="local-admin-secret"
export LECORE_X402_TENANT_SECRET="local-tenant-secret"
export LECORE_X402_LEOS_ACCESS_TOKEN="local-leos-buyer-secret"
export LECORE_X402_TENANT_STATE_DIR="./tenant-state"

python holographic_x402_api.py --host 127.0.0.1 --port 4021
```

Inspect pricing:

```bash
curl http://127.0.0.1:4021/pricing
```

Add memories locally as the seller:

```bash
curl -X POST http://127.0.0.1:4021/admin/remember \
  -H "Content-Type: application/json" \
  -H "X-Admin-Token: local-admin-secret" \
  -H "Idempotency-Key: initial-memory-001" \
  -d '{"text":"local agents need deterministic durable memory","label":"memory"}'
```

When `LECORE_X402_TENANT_STATE_DIR` is configured, admin writes use a small
durable transaction journal. Reuse the same `Idempotency-Key` after a timeout:
the API returns the original memory rather than creating another entry. Reusing
one key with a different request is rejected with `409 Conflict`.

For an enabled NoSQLite mirror, the journal records the core commit before
projecting the same stable memory id to NoSQLite. A temporary NoSQLite failure
leaves that projection pending; the same idempotent retry, or the next app
startup, resumes it without duplicating core memory. The implementation does
not advertise cross-store rollback it cannot provide.

Issue a private tenant token:

```bash
curl -X POST http://127.0.0.1:4021/admin/tenant-token \
  -H "Content-Type: application/json" \
  -H "X-Admin-Token: local-admin-secret" \
  -d '{"tenant":"acme"}'
```

Use that token with paid calls for private tenant memory:

```bash
curl -X POST http://127.0.0.1:4021/v1/recall \
  -H "Content-Type: application/json" \
  -H "X-leCore-Tenant: acme" \
  -H "X-leCore-Tenant-Token: <tenant token>" \
  -d '{"query":"deterministic local memory"}'
```

Use the discounted leOS route only with an issued offer credential:

```bash
curl -X POST http://127.0.0.1:4021/leos/v1/recall \
  -H "Content-Type: application/json" \
  -H "X-leCore-leOS-Access: local-leos-buyer-secret" \
  -d '{"query":"deterministic local memory"}'
```

Requests to paid routes return `402 Payment Required` unless the client retries
with a valid x402 payment payload:

```bash
curl -X POST http://127.0.0.1:4021/v1/recall \
  -H "Content-Type: application/json" \
  -d '{"query":"deterministic local memory"}'
```

## Local Unpaid Smoke Test

Use this only for development:

```bash
python holographic_x402_api.py --unpaid-dev --host 127.0.0.1 --port 4021
```

## Optional NoSQLite Memory Backend

`Dockerfile.x402` builds the vendored NoSQLite source snapshot pinned at
`8964da2` into the service image. The default remains `core`:
`LocalAgentCore` is the serving backend and the existing per-tenant JSON state
remains the durable control-plane mirror.

To cut semantic recall over to NoSQLite, configure a durable mounted directory:

```bash
export LECORE_X402_MEMORY_BACKEND=nosqlite
export LECORE_X402_NOSQLITE_BIN=/usr/local/bin/nosqlite
export LECORE_X402_NOSQLITE_DATA_DIR=/data/nosqlite
export LECORE_X402_NOSQLITE_DURABILITY=sync
export LECORE_X402_TENANT_STATE_DIR=/data/tenants
```

The API keeps each tenant in a separate hashed collection, writes the same
admin-created entry to `LocalAgentCore` for routing/dashboard continuity, and
uses NoSQLite's deterministic `holographic-hash-v1` encoder plus neural
candidate routing and cosine reranking for `/v1/recall`. Responses retain the
existing `id`, `text`, `label`, `metadata`, and `score` shape.

Before cutover, set `LECORE_X402_NOSQLITE_SHADOW=1` while leaving
`LECORE_X402_MEMORY_BACKEND=core`. Admin writes are mirrored; recall continues
to serve from the core while differences are logged without query text or
tenant identifiers.

NoSQLite-enabled writes require `LECORE_X402_TENANT_STATE_DIR`, which is also
where the transaction journal lives. Keep that directory on durable shared
storage with the tenant state; do not delete `.x402-memory-transactions` during
normal deployment cleanup.

NoSQLite filesystem mode holds one nonblocking exclusive writer lock for the
life of its process. Run exactly one active writer against a given data
directory. A rolling ECS replacement must drain the old writer before enabling
the new one, so the initial deployed configuration keeps this feature disabled
until that maintenance window is scheduled.

## Production Notes

- Use a real receiving wallet and a production facilitator.
- Put the API behind HTTPS.
- Keep route prices explicit; avoid wildcard paid route configs for this first
  product surface.
- Keep writes admin-only. Use `LECORE_X402_TENANT_SECRET` and
  `LECORE_X402_TENANT_STATE_DIR` for durable public and private memory. Writes
  use per-tenant process locks plus atomic replacement on shared storage.
- If NoSQLite is enabled, mount `LECORE_X402_NOSQLITE_DATA_DIR` on the same
  durable storage and keep the service at a single active writer for that path.
- Store `LECORE_X402_LEOS_ACCESS_TOKEN` as a secret and distribute it only to
  buyers eligible for the discounted routes.
- Treat x402 payment metadata as public enough to avoid putting secrets or PII
  in route descriptions.

The implementation follows the current x402 seller shape: FastAPI middleware,
`RouteConfig`, `PaymentOption`, an `exact` EVM scheme, and a facilitator-backed
resource server.

For AWS hosting, see [`AWS_X402_DEPLOY.md`](AWS_X402_DEPLOY.md).
