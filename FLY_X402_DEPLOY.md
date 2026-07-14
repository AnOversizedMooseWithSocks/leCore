# Fly.io x402 Deployment

This is the default production shape for serving `LocalAgentCore` as an
x402-paid API. It uses `Dockerfile.x402` and `fly.x402.toml`; no cloud-specific
code changes are needed. (For the AWS variant and the full wallet-storage
discussion, see [`AWS_X402_DEPLOY.md`](AWS_X402_DEPLOY.md) — the wallet
guidance there is provider-agnostic and applies here too.)

## Seller Security Model

Same as the AWS doc's short answer: the seller side needs **no private key**
in the container. It only needs:

- the public receiving wallet address (`LECORE_X402_PAY_TO`)
- x402/facilitator configuration
- an admin token for seller-only memory writes

Use a cold wallet, hardware wallet, or Safe/multisig as the receiving address.

## First Deploy (testnet)

```bash
fly launch --config fly.x402.toml --no-deploy   # creates the app, keeps our config
fly volumes create lecore_data --config fly.x402.toml --size 1

# Secrets first: the app fails loud on boot without a pay-to address.
fly secrets set --config fly.x402.toml \
  LECORE_X402_PAY_TO="0xYourReceivingWallet" \
  LECORE_X402_ADMIN_TOKEN="$(openssl rand -hex 24)"

fly deploy --config fly.x402.toml
```

Verify:

```bash
curl https://lecore-x402.fly.dev/health
curl https://lecore-x402.fly.dev/pricing
# Paid route must answer 402 with a `payment-required` challenge header:
curl -si https://lecore-x402.fly.dev/v1/dashboard | head -5
```

Seed seller memory (writes persist to the volume via `LECORE_X402_STATE`):

```bash
curl -X POST https://lecore-x402.fly.dev/admin/remember \
  -H "Content-Type: application/json" \
  -H "X-Admin-Token: <your admin token>" \
  -d '{"text":"local agents need deterministic durable memory","label":"memory"}'
```

## Custom Domain

```bash
fly certs add lecore.rati.foundation --config fly.x402.toml
```

Point DNS at the app, then set `LECORE_X402_PUBLIC_URL` in `fly.x402.toml` to
the custom domain so the landing page and `/pricing` advertise the right
endpoint.

## Mainnet Flip

The defaults are Base Sepolia + the signup-free x402.org testnet facilitator,
which does **not** settle real funds. To charge real USDC on Base:

```toml
LECORE_X402_NETWORK = "eip155:8453"
LECORE_X402_FACILITATOR_URL = "https://api.cdp.coinbase.com/platform/v2/x402"
```

The CDP facilitator requires Coinbase Developer Platform credentials — check
the current x402/CDP docs for the auth shape and set any required keys with
`fly secrets set`, never in `fly.x402.toml`. Re-verify the `payment-required`
challenge advertises `eip155:8453` before announcing the endpoint.

## Production Checklist

- Mainnet network id + production facilitator before announcing.
- Receiving address is cold/multisig, never a hot key in the container.
- `LECORE_X402_ADMIN_TOKEN` set via `fly secrets`, rotated if shared.
- Volume mounted and `LECORE_X402_STATE` set, or accept that admin writes
  reset to the demo core on every restart.
- Keep paid route configs explicit; no wildcard paid routes.
- No secrets or PII in route descriptions or payment metadata.
