#!/bin/bash
set -Eeuo pipefail
exec > >(tee -a /var/log/qwen35-bootstrap.log) 2>&1
shutdown -h +300
mkdir -p /opt/qwen35-acceptance-v7/inputs
if ! command -v aws >/dev/null 2>&1; then
  dnf install -y awscli2 || dnf install -y awscli
fi
aws s3 cp s3://zero-training-022118847419/qwen35-acceptance/cb3b1d2-2fc06364-a6c9d113/v7-formal/input/runner.sh /opt/qwen35-acceptance-v7/inputs/runner.sh --only-show-errors
echo '705bfcf15653eab1996d94b480fb7cb097a1b0ef5c0da38de9266458075798ed  /opt/qwen35-acceptance-v7/inputs/runner.sh' | sha256sum -c -
chmod 700 /opt/qwen35-acceptance-v7/inputs/runner.sh
exec /opt/qwen35-acceptance-v7/inputs/runner.sh
