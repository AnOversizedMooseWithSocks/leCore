#!/bin/bash
set -Eeuo pipefail
exec > >(tee -a /var/log/qwen35-bootstrap.log) 2>&1
shutdown -h +480
mkdir -p /opt/qwen35-acceptance-v4/inputs
if ! command -v aws >/dev/null 2>&1; then
  dnf install -y awscli2 || dnf install -y awscli
fi
aws s3 cp s3://zero-training-022118847419/qwen35-acceptance/c1ddc22-2fc06364-a6c9d113/v4-formal/input/runner.sh /opt/qwen35-acceptance-v4/inputs/runner.sh --only-show-errors
echo '8abb394ccae8893406dcd4d055c24b406a90426d4fee1a529d41ea761d2d7754  /opt/qwen35-acceptance-v4/inputs/runner.sh' | sha256sum -c -
chmod 700 /opt/qwen35-acceptance-v4/inputs/runner.sh
exec /opt/qwen35-acceptance-v4/inputs/runner.sh
