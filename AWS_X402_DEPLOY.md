# AWS x402 Deployment

This is the production shape for serving `LocalAgentCore` as an x402-paid API
on AWS.

## Short Answer

Yes, we can launch this on AWS. For the **seller** side of x402, the service
does **not** need a wallet private key in the container. It only needs:

- the public receiving wallet address (`LECORE_X402_PAY_TO`)
- x402/facilitator configuration
- an admin token for seller-only memory writes

The receiving wallet should be a cold wallet, hardware wallet, Safe/multisig,
or a custody wallet. The API simply tells x402 where funds should go.

Only build an AWS-hosted signing wallet if the app itself must **spend** funds
or pay upstream APIs as a buyer.

## Recommended AWS Architecture

- **ECS Fargate** runs the `Dockerfile.x402` container.
- **Application Load Balancer** terminates HTTPS and forwards to port `4021`.
- **ECR** stores the container image.
- **Secrets Manager** stores `LECORE_X402_ADMIN_TOKEN` and production
  facilitator credentials.
- **SSM Parameter Store or plain task env** stores non-secret config like
  `LECORE_X402_PAY_TO`, `LECORE_X402_PRICE`, and `LECORE_X402_NETWORK`.
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
LECORE_X402_PRICE=$0.001
LECORE_X402_NETWORK=eip155:8453
LECORE_X402_FACILITATOR_URL=https://api.cdp.coinbase.com/platform/v2/x402
```

Secrets Manager values:

```text
LECORE_X402_ADMIN_TOKEN=<random admin token>
CDP_API_KEY_ID=<if required by facilitator setup>
CDP_API_KEY_SECRET=<if required by facilitator setup>
```

Use ECS task definition `secrets` entries for secrets, not literal environment
variables in the task definition.

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
- Keep paid route configs explicit; avoid wildcard paid routes at first.
- Add WAF rate limits.
- Add CloudWatch alarms on 5xx, 402 spikes, and admin write attempts.
- Keep customer memory isolated before offering paid writes.
- Do not put secrets or PII in x402 route descriptions or payment metadata.

## Local Smoke Before AWS

```bash
pip install ".[x402]"
export LECORE_X402_PAY_TO="0xYourReceivingWallet"
export LECORE_X402_ADMIN_TOKEN="local-admin-secret"
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
