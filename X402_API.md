# x402 API Publishing

Yes: leCore can be published as a paid API with x402.

The implementation lives in `holographic_x402_api.py`. It wraps
`LocalAgentCore` with a small FastAPI app and applies x402 middleware only to
the public read/compute routes:

- `POST /v1/recall`
- `POST /v1/route`
- `GET /v1/dashboard`
- `POST /leos/v1/recall`, at the leOS CA offer price
- `POST /leos/v1/route`, at the leOS CA offer price
- `GET /leos/v1/dashboard`, at the leOS CA offer price

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
  -d '{"text":"local agents need deterministic durable memory","label":"memory"}'
```

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

## Production Notes

- Use a real receiving wallet and a production facilitator.
- Put the API behind HTTPS.
- Keep route prices explicit; avoid wildcard paid route configs for this first
  product surface.
- Keep writes admin-only. Use `LECORE_X402_TENANT_SECRET` and
  `LECORE_X402_TENANT_STATE_DIR` for isolated private customer memory.
- Treat x402 payment metadata as public enough to avoid putting secrets or PII
  in route descriptions.

The implementation follows the current x402 seller shape: FastAPI middleware,
`RouteConfig`, `PaymentOption`, an `exact` EVM scheme, and a facilitator-backed
resource server.

For AWS hosting, see [`AWS_X402_DEPLOY.md`](AWS_X402_DEPLOY.md).
