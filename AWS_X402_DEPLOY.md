# AWS x402 Deployment

This is the production shape for serving the hosted leCore Agent Memory &
Routing API with x402 payments on AWS. The service is backed internally by
`LocalAgentCore`.

## Short Answer

Yes, we can launch this on AWS. For the **seller** side of x402, the service
does **not** need a wallet private key in the container. It only needs:

- the public receiving wallet address (`LECORE_X402_PAY_TO`)
- x402/facilitator configuration
- an admin token for seller-only memory writes
- a tenant-token secret if private customer memory is enabled
- a separate versioned memory-encryption keyring for durable customer memory

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
  Store `LECORE_X402_MEMORY_KEYS` as a separate secret; never derive it from or
  reuse an admin, tenant-token, facilitator, or wallet secret.
- **SSM Parameter Store or plain task env** stores non-secret config like
  `LECORE_X402_PAY_TO`, `LECORE_X402_PRICE`, `LECORE_X402_NETWORK`,
  `LECORE_X402_PUBLIC_URL`, and `LECORE_X402_TENANT_STATE_DIR`.
- **CloudWatch Logs** captures service logs.
- **AWS WAF** can rate-limit and block bad traffic at the ALB.

Protected paid routes:

- `POST /v1/memory` (private tenant + idempotency key required)
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

The deployed preview runs in `us-east-1` from the existing
`lecore-x402-api` ECR repository. Build a unique ARM64 image, then pin the
deployment to its digest. Never deploy `latest`.

```bash
ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
REGION="us-east-1"
REPOSITORY="lecore-x402-api"
REGISTRY="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"
REVISION="$(git rev-parse HEAD)"
IMAGE_TAG="${REVISION:0:12}-$(date -u +%Y%m%dT%H%M%SZ)"
IMAGE="$REGISTRY/$REPOSITORY:$IMAGE_TAG"

aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin "$REGISTRY"

docker build --platform linux/arm64 \
  --label "org.opencontainers.image.revision=$REVISION" \
  -f Dockerfile.x402 -t "$IMAGE" .
docker push "$IMAGE"

DIGEST="$(aws ecr describe-images --region "$REGION" \
  --repository-name "$REPOSITORY" --image-ids imageTag="$IMAGE_TAG" \
  --query 'imageDetails[0].imageDigest' --output text)"
PINNED_IMAGE="$REGISTRY/$REPOSITORY@$DIGEST"

aws ecr wait image-scan-complete --region "$REGION" \
  --repository-name "$REPOSITORY" --image-id imageDigest="$DIGEST"
aws ecr describe-image-scan-findings --region "$REGION" \
  --repository-name "$REPOSITORY" --image-id imageDigest="$DIGEST" \
  --query 'imageScanFindings.findingSeverityCounts'
```

Review the scan before registration. Do not deploy if the new image regresses
against the active image or violates the release vulnerability policy.

## Runtime Environment

Non-secret environment variables:

```text
LECORE_X402_PAY_TO=0xYourReceivingWallet
LECORE_X402_PRICE=$0.0011
LECORE_X402_NETWORK=eip155:8453
LECORE_X402_FACILITATOR_URL=https://api.cdp.coinbase.com/platform/v2/x402
LECORE_X402_PUBLIC_URL=https://lecore.rati.foundation
LECORE_X402_TENANT_STATE_DIR=/data/tenants
LECORE_X402_MEMORY_BACKEND=core
```

Secrets Manager values:

```text
LECORE_X402_ADMIN_TOKEN=<random admin token>
LECORE_X402_TENANT_SECRET=<random tenant-token signing secret>
LECORE_X402_MEMORY_KEYS={"active":"v1","keys":{"v1":"<32-byte base64 key>"}}
CDP_API_KEY_ID=<if required by facilitator setup>
CDP_API_KEY_SECRET=<if required by facilitator setup>
```

Use ECS task definition `secrets` entries for secrets, not literal environment
variables in the task definition.

Generate the first keyring offline and send it directly to Secrets Manager;
never print the deployed value in logs or commit it:

```bash
umask 077
python -c 'import base64,json,secrets; print(json.dumps({"active":"v1","keys":{"v1":base64.urlsafe_b64encode(secrets.token_bytes(32)).decode()}}))' \
  | aws secretsmanager create-secret --region us-east-1 \
      --name /lecore/x402/memory-keys --secret-string file:///dev/stdin
```

Grant the ECS execution role `secretsmanager:GetSecretValue` only on that exact
secret ARN, and add it to the `app` container's task-definition `secrets` array
as `LECORE_X402_MEMORY_KEYS`.

## Durable Memory Security Boundary

The production `core` backend compresses each tenant file and durable retry
journal, derives a per-record key with HKDF-SHA256, and encrypts/authenticates it
with AES-256-GCM before the atomic write. Tenant/file identity is authenticated
as associated data, so copying ciphertext between tenants is rejected. The
application keyring is independent from EFS encryption at rest and TLS in
transit; all three layers remain enabled.

Paid `/v1/memory` requires durable state, a valid private-tenant token, and an
`Idempotency-Key`. The service refuses to start paid durable mode without the
memory keyring. It also refuses NoSQLite or NoSQLite shadow mode while the
application encryption keyring is configured, because those files do not yet
use this envelope.

For key rotation, add a new key while retaining the old one, make the new id
`active`, and deploy. Startup rewraps every tenant and journal record under the
active key. Verify clean startup and encrypted storage, then remove the old key
and launch fresh tasks. A secret update alone is insufficient: ECS injects the
value only when a task starts.

For the one-time migration from legacy plaintext, do not use the normal
overlapping 100/200 rollout: an old task cannot read the new ciphertext and can
still write plaintext. Drain to zero or otherwise guarantee one writer, add
`LECORE_X402_ALLOW_PLAINTEXT_MIGRATION=1`, and start one new task with the
keyring. After it rewrites and verifies every file, remove the flag and perform
a fresh deployment. The flag must never remain enabled during normal service.

## ECS Rollout

The live service is `lonely-forest-cluster/lecore-x402-api`. Treat its current
task definition as the rollback target and change only the `app` container
image. This preserves the task roles, ARM64 runtime, CPU and memory, logging,
EFS volume, environment, and secret references.

The commands below assume `REGION`, `PINNED_IMAGE`, and `DIGEST` are still set
from the build step and that `jq` is installed.

```bash
CLUSTER="lonely-forest-cluster"
SERVICE="lecore-x402-api"
umask 077
DEPLOY_DIR="$(mktemp -d /tmp/lecore-x402-deploy.XXXXXX)"
trap 'rm -rf -- "$DEPLOY_DIR"' EXIT
ROLLBACK_TASK_DEF="$(aws ecs describe-services --region "$REGION" \
  --cluster "$CLUSTER" --services "$SERVICE" \
  --query 'services[0].taskDefinition' --output text)"

aws ecs describe-task-definition --region "$REGION" \
  --task-definition "$ROLLBACK_TASK_DEF" --include TAGS \
  --output json > "$DEPLOY_DIR/described-task.json"
jq '.taskDefinition' "$DEPLOY_DIR/described-task.json" \
  > "$DEPLOY_DIR/base-task.json"
TASK_TAGS="$(jq -c '.tags // []' "$DEPLOY_DIR/described-task.json")"

jq --arg image "$PINNED_IMAGE" '
  if ([.containerDefinitions[] | select(.name == "app")] | length) != 1
  then error("expected exactly one app container") else . end
  |
  del(
    .taskDefinitionArn, .revision, .status, .requiresAttributes,
    .compatibilities, .registeredAt, .registeredBy, .deregisteredAt
  )
  | (.containerDefinitions[] | select(.name == "app").image) = $image
' "$DEPLOY_DIR/base-task.json" > "$DEPLOY_DIR/next-task.json"

jq -S '
  del(
    .taskDefinitionArn, .revision, .status, .requiresAttributes,
    .compatibilities, .registeredAt, .registeredBy, .deregisteredAt
  )
  | (.containerDefinitions[] | select(.name == "app").image) = "__IMAGE__"
' "$DEPLOY_DIR/base-task.json" > "$DEPLOY_DIR/base.normalized.json"
jq -S '
  (.containerDefinitions[] | select(.name == "app").image) = "__IMAGE__"
' "$DEPLOY_DIR/next-task.json" > "$DEPLOY_DIR/next.normalized.json"
cmp "$DEPLOY_DIR/base.normalized.json" "$DEPLOY_DIR/next.normalized.json"

CURRENT_TASK_DEF="$(aws ecs describe-services --region "$REGION" \
  --cluster "$CLUSTER" --services "$SERVICE" \
  --query 'services[0].taskDefinition' --output text)"
test "$CURRENT_TASK_DEF" = "$ROLLBACK_TASK_DEF"

NEW_TASK_DEF="$(aws ecs register-task-definition --region "$REGION" \
  --cli-input-json "file://$DEPLOY_DIR/next-task.json" --tags "$TASK_TAGS" \
  --query 'taskDefinition.taskDefinitionArn' --output text)"

aws ecs update-service --region "$REGION" --cluster "$CLUSTER" \
  --service "$SERVICE" --task-definition "$NEW_TASK_DEF"
aws ecs wait services-stable --region "$REGION" \
  --cluster "$CLUSTER" --services "$SERVICE"
```

If the equality check fails, another operator changed the service after this
rollout began. Stop, inspect that task definition, and rebuild the candidate
from the new base rather than overwriting it.

Normal rolling deployment is safe for later image-only changes while
`LECORE_X402_MEMORY_BACKEND=core` and every task already has the same keyring.
The first plaintext-to-encrypted migration is deliberately a drain-and-replace
operation. Do not turn on the single-writer NoSQLite backend in the same rollout.

## Verify And Roll Back

Verify all of the following before considering the rollout complete:

- ECS reports one running task, none pending, and the service references
  `NEW_TASK_DEF`.
- The running `app` container's `imageDigest` equals `DIGEST`.
- The ALB target is healthy.
- `GET /health` and `GET /pricing` return `200`.
- `GET /v1/dashboard` returns `402`, and its decoded `payment-required` header
  advertises exactly
  `https://lecore.rati.foundation/v1/dashboard`.
- `/health` still reports private tenancy and durable transactions, and the
  EFS-backed memory state is present.
- `/health.memory_backend.storage` reports `durable=true`, `encrypted=true`,
  `cipher=AES-256-GCM`, `compression=zlib`, and
  `plaintext_migration_enabled=false` after migration.
- `POST /v1/memory` without payment returns `402`; after an authorized testnet
  payment, tenant token, and idempotency key, it stores once and can be recalled.
- CloudWatch logs since the rollout contain no new errors, tracebacks, or
  exceptions.

Roll back on a stability wait failure, an unhealthy target, a wrong image
digest, any `5xx`, missing durable state, or an incorrect payment resource URL:

```bash
aws ecs update-service --region "$REGION" --cluster "$CLUSTER" \
  --service "$SERVICE" --task-definition "$ROLLBACK_TASK_DEF"
aws ecs wait services-stable --region "$REGION" \
  --cluster "$CLUSTER" --services "$SERVICE"
```

Re-run the endpoint and digest checks after rollback. Do not deregister the
rollback task definition or delete its ECR digest.

## Optional NoSQLite Cutover (Not Compatible With Encrypted Production Memory)

The container has `/usr/local/bin/nosqlite` built from the vendored source
snapshot pinned at `8964da27670c752121b8e6d26d113577429b02f6`. To use it for
`/v1/recall`, add:

```text
LECORE_X402_MEMORY_BACKEND=nosqlite
LECORE_X402_NOSQLITE_BIN=/usr/local/bin/nosqlite
LECORE_X402_NOSQLITE_DATA_DIR=/data/nosqlite
LECORE_X402_NOSQLITE_DURABILITY=sync
```

NoSQLite currently sits outside the authenticated application-encryption
boundary. The API fails closed if it is selected together with
`LECORE_X402_MEMORY_KEYS`; do not use this mode for customer memory.

For an explicitly unencrypted development deployment, mount `/data/nosqlite` on durable storage. NoSQLite deliberately takes a
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
- Set `LECORE_X402_PUBLIC_URL` to the canonical HTTPS endpoint so payment
  challenges never depend on forwarded request headers.
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

## Pre-deployment Smoke Test

```bash
pip install ".[x402]"
export LECORE_X402_PAY_TO="0xYourReceivingWallet"
export LECORE_X402_ADMIN_TOKEN="dev-admin-secret"
export LECORE_X402_TENANT_SECRET="dev-tenant-secret"
python holographic_x402_api.py --unpaid-dev --host 127.0.0.1 --port 4021
```

Then:

```bash
curl http://127.0.0.1:4021/health
curl http://127.0.0.1:4021/pricing
curl -X POST http://127.0.0.1:4021/v1/route \
  -H "Content-Type: application/json" \
  -d '{"task":"search tenant-scoped agent memory"}'
```
