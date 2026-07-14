# AWS x402 Deployment

This is the production shape for serving `LocalAgentCore` as an x402-paid API
on AWS.

## Short Answer

Yes, we can launch this on AWS. For the **seller** side of x402, the service
does **not** need a wallet private key in the container. It only needs:

- the public receiving wallet address (`LECORE_X402_PAY_TO`)
- x402/facilitator configuration
- an admin token for seller-only memory writes
- a tenant-token secret if private customer memory is enabled

The receiving wallet should be a cold wallet, hardware wallet, Safe/multisig,
or a custody wallet. The API simply tells x402 where funds should go.

Only build an AWS-hosted signing wallet if the app itself must **spend** funds
or pay upstream APIs as a buyer.

## Recommended AWS Architecture

- **ECS Fargate** runs the `Dockerfile.x402` container.
- The image includes a pinned NoSQLite CLI for an optional semantic-memory
  backend; it is disabled by default.
- **Application Load Balancer** terminates HTTPS and forwards to port `4021`.
- **ECR** stores the container image.
- **Secrets Manager** stores `LECORE_X402_ADMIN_TOKEN` and production
  facilitator credentials.
  Store `LECORE_X402_TENANT_SECRET` there too when private tenants are enabled.
- **SSM Parameter Store or plain task env** stores non-secret config like
  `LECORE_X402_PAY_TO`, `LECORE_X402_PRICE`, `LECORE_X402_NETWORK`, and
  `LECORE_X402_TENANT_STATE_DIR`.
- **CloudWatch Logs** captures service logs.
- **AWS WAF** can rate-limit and block bad traffic at the ALB.

Protected paid routes:

- `POST /v1/recall`
- `POST /v1/route`
- `GET /v1/dashboard`

Free routes:

- `GET /health`
- `GET /pricing`

Seller-only route:

- `POST /admin/remember`, guarded by `X-Admin-Token`
- `POST /admin/tenant-token`, guarded by `X-Admin-Token`

## Build And Push

```bash
aws ecr create-repository --repository-name lecore-x402

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
REGION="${AWS_REGION:-us-west-2}"
IMAGE="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/lecore-x402:latest"

aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"

docker build -f Dockerfile.x402 -t "$IMAGE" .
docker push "$IMAGE"
```

## Runtime Environment

Non-secret environment variables:

```text
LECORE_X402_PAY_TO=0xYourReceivingWallet
LECORE_X402_PRICE=$0.0011
LECORE_X402_NETWORK=eip155:8453
LECORE_X402_FACILITATOR_URL=https://api.cdp.coinbase.com/platform/v2/x402
LECORE_X402_TENANT_STATE_DIR=/data/tenants
LECORE_X402_MEMORY_BACKEND=core
```

Secrets Manager values:

```text
LECORE_X402_ADMIN_TOKEN=<random admin token>
LECORE_X402_TENANT_SECRET=<random tenant-token signing secret>
CDP_API_KEY_ID=<if required by facilitator setup>
CDP_API_KEY_SECRET=<if required by facilitator setup>
```

Use ECS task definition `secrets` entries for secrets, not literal environment
variables in the task definition.

## Optional NoSQLite Cutover

The container has `/usr/local/bin/nosqlite` built from the vendored source
snapshot pinned at `8964da27670c752121b8e6d26d113577429b02f6`. To use it for
`/v1/recall`, add:

```text
LECORE_X402_MEMORY_BACKEND=nosqlite
LECORE_X402_NOSQLITE_BIN=/usr/local/bin/nosqlite
LECORE_X402_NOSQLITE_DATA_DIR=/data/nosqlite
LECORE_X402_NOSQLITE_DURABILITY=sync
```

Mount `/data/nosqlite` on durable storage. NoSQLite deliberately takes a
nonblocking exclusive writer lock for the whole process, so a single data path
must have exactly one active ECS writer. Use a deliberate drain-and-replace
maintenance deployment for the cutover; do not rely on the normal overlapping
rolling deployment. The service currently stays on `core` until that operation
is scheduled.

For a no-serving-impact validation phase, use:

```text
LECORE_X402_MEMORY_BACKEND=core
LECORE_X402_NOSQLITE_SHADOW=1
LECORE_X402_NOSQLITE_BIN=/usr/local/bin/nosqlite
LECORE_X402_NOSQLITE_DATA_DIR=/data/nosqlite
```

That mirrors admin writes and compares recall internally while preserving the
existing LocalAgentCore response as the source of truth.

## Wallet Storage Decision

### Seller API, Recommended

Do **not** store a private key in AWS.

The API receives payments; it does not spend. x402 payment verification and
settlement happen through the facilitator. The service only advertises
`payTo`.

Best receiving wallet options:

- Safe/multisig
- hardware wallet
- cold wallet
- custodial account dedicated to receipts

### Buyer/Spender API, If Needed Later

If the leCore agent itself needs to pay other x402 APIs, use a separate signer
service:

1. Create an AWS KMS asymmetric signing key with `ECC_SECG_P256K1`.
2. Derive the public Ethereum address from `kms:GetPublicKey`.
3. Allow only a narrow IAM role to call `kms:Sign`.
4. Sign EIP-712/EIP-3009 payload digests through KMS.
5. Enforce spend limits in application logic before every signing request.
6. Log every signing request with CloudTrail and app-level audit records.

This keeps the private key non-exportable: it never appears in the container.

### High-Assurance Signer

For larger balances or stronger isolation, put the signing service in **AWS
Nitro Enclaves** and allow KMS decrypt/sign only when enclave attestation
matches the expected image measurement.

### Last Resort

Storing a raw private key in Secrets Manager is acceptable only for testnet or
very small hot-wallet balances. If used, wrap it with strict IAM, rotation
plans, spend limits, CloudTrail alarms, and a tiny blast radius.

## First Production Checklist

- Use mainnet network id and production facilitator URL.
- Put the ALB behind HTTPS only.
- Keep `/admin/remember` private or blocked from the public ALB path.
- Keep `/admin/tenant-token` private or blocked from the public ALB path.
- Keep paid route configs explicit; avoid wildcard paid routes at first.
- Add WAF rate limits.
- Add CloudWatch alarms on 5xx, 402 spikes, and admin write attempts.
- Use tenant tokens plus isolated tenant state before offering private customer
  memory.
- Mount `LECORE_X402_TENANT_STATE_DIR` on shared durable storage. Tenant writes
  reload under an OS-level lock and use atomic replacement, so rolling ECS tasks
  do not overwrite one another.
- Preserve the `.x402-memory-transactions` directory inside tenant state. It is
  the durable outbox for core-to-NoSQLite writes; callers should send an
  `Idempotency-Key` on `/admin/remember` retries so a timeout cannot duplicate
  a memory.
- Do not enable NoSQLite on the same EFS directory in overlapping ECS tasks;
  schedule a single-writer drain-and-replace cutover instead.
- Do not put secrets or PII in x402 route descriptions or payment metadata.

## Local Smoke Before AWS

```bash
pip install ".[x402]"
export LECORE_X402_PAY_TO="0xYourReceivingWallet"
export LECORE_X402_ADMIN_TOKEN="local-admin-secret"
export LECORE_X402_TENANT_SECRET="local-tenant-secret"
python holographic_x402_api.py --unpaid-dev --host 127.0.0.1 --port 4021
```

Then:

```bash
curl http://127.0.0.1:4021/health
curl http://127.0.0.1:4021/pricing
curl -X POST http://127.0.0.1:4021/v1/route \
  -H "Content-Type: application/json" \
  -d '{"task":"search local agent memory"}'
```
