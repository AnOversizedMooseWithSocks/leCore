# leCore Agent Memory & Routing API

leCore is available as a hosted memory and capability-routing API with x402
payment on the protected routes.

Public reference:

- [Swagger UI](https://lecore.rati.foundation/docs)
- [ReDoc reference](https://lecore.rati.foundation/redoc)
- [OpenAPI 3.1 schema](https://lecore.rati.foundation/openapi.json)
- [Pricing and route manifest](https://lecore.rati.foundation/pricing)

The implementation lives in `holographic_x402_api.py`. It exposes
tenant-scoped agent memory and routing through FastAPI, backed internally by
`LocalAgentCore`, and applies x402 middleware only to the public memory/compute
routes:

- `POST /v1/memory` — encrypted, idempotent private-tenant writes
- `GET /v1/memory` — bounded listing or exact-id retrieval
- `PATCH /v1/memory` — atomic selected-field updates
- `DELETE /v1/memory` — idempotent deletion with durable tombstones
- `POST /v1/recall`
- `POST /v1/route`
- `GET /v1/dashboard`

Free routes:

- `GET /health`
- `GET /pricing`

Admin route:

- `POST /admin/remember`, guarded by `X-Admin-Token`
- `POST /admin/tenant-token`, guarded by `X-Admin-Token`

This split is deliberate. Paid customers can store and recall their own private
tenant memory, but cannot mutate the shared public dataset. A private write
requires both a tenant token and a stable `Idempotency-Key`; x402 proves payment,
not tenant authorization. Admin routes remain available for provisioning and
are not included in the public OpenAPI schema.

Durable core files and write journals are compressed before authenticated
encryption with AES-256-GCM. HKDF-SHA256 derives a distinct data key for each
tenant/file context from a versioned service keyring. The tenant/file identity
is authenticated as associated data, files are replaced atomically with mode
`0600`, altered ciphertext is rejected, and paid durable mode fails closed when
the keyring is absent. AWS volume encryption remains a second, independent
layer rather than the only protection.

## Public Preview Quickstart

Read the free discovery manifest before signing anything:

```bash
curl -sS https://lecore.rati.foundation/pricing
```

Make an unsigned request to see the exact x402 contract without moving testnet
funds:

```bash
curl -i https://lecore.rati.foundation/v1/dashboard
```

The response is `402 Payment Required` with a base64 `Payment-Required` header.
Configure an x402 v2 client using the
[official buyer quickstart](https://docs.x402.org/getting-started/quickstart-for-buyers),
sign one accepted option, and retry with `Payment-Signature`. A successful paid
response includes `Payment-Response` with the settlement result.

The public OpenAPI contract documents request bodies, successful response
shapes, payment headers, tenant authorization failures, facilitator errors, and
the recall backend's availability response.

## Install

```bash
pip install ".[x402]"
```

The core package still needs only NumPy. The `x402` extra pulls in the optional
FastAPI/x402/uvicorn stack.

## Testnet Run

The default network is Base Sepolia (`eip155:84532`) and the default facilitator
is the signup-free x402.org testnet facilitator. This is a **developer preview**:
the listed `$0.0011` request price is displayed as `$1.10 per 1,000 requests`,
uses testnet USDC, and does not accept production payments.

```bash
export LECORE_X402_PAY_TO="0xYourReceivingWallet"
export LECORE_X402_PRICE="$0.0011"
export LECORE_X402_PUBLIC_URL="http://127.0.0.1:4021"
export LECORE_X402_ADMIN_TOKEN="dev-admin-secret"
export LECORE_X402_TENANT_SECRET="dev-tenant-secret"
export LECORE_X402_TENANT_STATE_DIR="./tenant-state"
export LECORE_X402_MEMORY_KEYS="$(python -c 'import base64,json,secrets; print(json.dumps({"active":"v1","keys":{"v1":base64.urlsafe_b64encode(secrets.token_bytes(32)).decode()}}))')"

python holographic_x402_api.py --host 127.0.0.1 --port 4021
```

Inspect pricing:

```bash
curl http://127.0.0.1:4021/pricing
```

Add memories through the operator endpoint:

```bash
curl -X POST http://127.0.0.1:4021/admin/remember \
  -H "Content-Type: application/json" \
  -H "X-Admin-Token: dev-admin-secret" \
  -H "Idempotency-Key: initial-memory-001" \
  -d '{"text":"agents need deterministic durable memory","label":"memory"}'
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
  -H "X-Admin-Token: dev-admin-secret" \
  -d '{"tenant":"acme"}'
```

Use that token with paid calls for private tenant memory:

```bash
curl -X POST http://127.0.0.1:4021/v1/memory \
  -H "Content-Type: application/json" \
  -H "X-leCore-Tenant: acme" \
  -H "X-leCore-Tenant-Token: <tenant token>" \
  -H "Idempotency-Key: session-42-preference-001" \
  -d '{"text":"the user prefers concise answers","label":"preference"}'
```

In paid mode that unsigned request first returns `402`; an x402 client signs
and retries the same body and idempotency key. Repeating the completed request
returns the original memory. Reusing the key for different content returns
`409 Conflict`.

Recall it:

```bash
curl -X POST http://127.0.0.1:4021/v1/recall \
  -H "Content-Type: application/json" \
  -H "X-leCore-Tenant: acme" \
  -H "X-leCore-Tenant-Token: <tenant token>" \
  -d '{"query":"deterministic agent memory"}'
```

List the tenant's memories in insertion order, at most 100 per page:

```bash
curl "http://127.0.0.1:4021/v1/memory?limit=50" \
  -H "X-leCore-Tenant: acme" \
  -H "X-leCore-Tenant-Token: <tenant token>"
```

Pass the returned `next_cursor` to continue, or retrieve one exact record:

```bash
curl "http://127.0.0.1:4021/v1/memory?memory_id=<memory id>" \
  -H "X-leCore-Tenant: acme" \
  -H "X-leCore-Tenant-Token: <tenant token>"
```

Update any combination of text, label, and metadata. Omitted fields are
preserved, `label: null` clears the label, and `{}` clears metadata:

```bash
curl -X PATCH "http://127.0.0.1:4021/v1/memory?memory_id=<memory id>" \
  -H "Content-Type: application/json" \
  -H "X-leCore-Tenant: acme" \
  -H "X-leCore-Tenant-Token: <tenant token>" \
  -d '{"text":"the user prefers concise release notes","metadata":{"confirmed":true}}'
```

An update is atomic with the encrypted tenant snapshot. A no-op update does
not rewrite the file. Retrying the original create request after changing the
record returns `409` instead of silently overwriting the newer value.

Delete one record idempotently:

```bash
curl -X DELETE "http://127.0.0.1:4021/v1/memory?memory_id=<memory id>" \
  -H "X-leCore-Tenant: acme" \
  -H "X-leCore-Tenant-Token: <tenant token>"
```

Deletion writes an encrypted tombstone into the originating retry journal
before removing the core entry. Retrying the old store request cannot resurrect
the deleted memory; it returns `409` and requires a new idempotency key.

Requests to paid routes return `402 Payment Required` unless the client retries
with a valid x402 payment payload:

```bash
curl -X POST http://127.0.0.1:4021/v1/recall \
  -H "Content-Type: application/json" \
  -d '{"query":"deterministic agent memory"}'
```

## Unpaid Development Smoke Test

Use this only for development:

```bash
python holographic_x402_api.py --unpaid-dev --host 127.0.0.1 --port 4021
```

Unpaid development may omit the keyring and use plaintext files. That fallback
is intentionally unavailable to a paid app with durable state.

## Storage Performance

The current v1 envelope is tuned for small and medium tenant snapshots:

- zlib level 6 balances compression ratio and CPU time;
- current in-process tenants are not decrypted and rebuilt before every write;
- cross-task file-version changes still force a safe reload under the lock;
- no-op updates and missing/idempotent deletes do not rewrite the tenant file;
- listing is bounded to 100 records per response and uses stable cursors.

When NoSQLite semantic indexing is enabled in an unpaid deployment, updates
replace the document and embedding and deletes remove the projection. The
durable core snapshot remains authoritative; a fresh process or an interrupted
projection reconciles the tenant collection from that snapshot before serving
semantic recall. Application-encrypted paid memory continues to reject the
plaintext NoSQLite backend and shadow instead of silently weakening storage.

An actual mutation still atomically rewrites one compressed tenant snapshot.
That is simple and crash-safe, but it is `O(tenant memory size)`. Before very
large tenants, introduce a version-2 encrypted append log or fixed-size
encrypted segments with background compaction. Keep v1 as the migration and
recovery format rather than adding an unauthenticated side index.

## Key Rotation And Legacy Migration

`LECORE_X402_MEMORY_KEYS` is a Secrets Manager JSON value with one active key
and up to seven retained decryption keys:

```json
{"active":"v2","keys":{"v1":"<old 32-byte base64 key>","v2":"<new 32-byte base64 key>"}}
```

Deploy the expanded keyring first. Startup authenticates every tenant and
journal file and atomically re-encrypts records not using `active`. After every
record has been verified under `v2`, remove `v1` and launch fresh tasks.

Existing plaintext state is refused by default. For a controlled one-time
migration, deploy the keyring with
`LECORE_X402_ALLOW_PLAINTEXT_MIGRATION=1`, drain old writers, start exactly one
new task, and verify every durable file now begins with the `LECMEM01` envelope
magic. Then remove the migration flag immediately; leaving it enabled would
allow plaintext to bypass ciphertext authentication.

## Optional NoSQLite Memory Backend (Unencrypted Development Only)

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

NoSQLite does not yet use the application encryption envelope. The service
therefore refuses NoSQLite serving or shadow mode whenever memory encryption is
configured. Do not use it for the hosted paid API until NoSQLite gains an
equivalent authenticated-encryption boundary.

In an explicitly unencrypted development deployment, the API keeps each tenant in a separate hashed collection, writes the same
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
- Put the API behind HTTPS and set `LECORE_X402_PUBLIC_URL` to its canonical
  public base URL, for example `https://lecore.rati.foundation`. Each payment
  challenge advertises that configured URL rather than trusting forwarded
  request headers.
- Keep route prices explicit; avoid wildcard paid route configs for this first
  product surface.
- Keep writes admin-only. Use `LECORE_X402_TENANT_SECRET` and
  `LECORE_X402_TENANT_STATE_DIR` for durable public and private memory. Writes
  use per-tenant process locks plus atomic replacement on shared storage.
- If NoSQLite is enabled, mount `LECORE_X402_NOSQLITE_DATA_DIR` on the same
  durable storage and keep the service at a single active writer for that path.
- Treat x402 payment metadata as public enough to avoid putting secrets or PII
  in route descriptions.

The implementation follows the current x402 seller shape: FastAPI middleware,
`RouteConfig`, `PaymentOption`, an `exact` EVM scheme, and a facilitator-backed
resource server.

For AWS hosting, see [`AWS_X402_DEPLOY.md`](AWS_X402_DEPLOY.md).
