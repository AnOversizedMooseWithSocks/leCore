#!/bin/bash
set -Eeuo pipefail
exec > >(tee -a /var/log/qwen35-bootstrap.log) 2>&1
shutdown -h +390
mkdir -p /opt/qwen35-acceptance-v6/inputs
if ! command -v aws >/dev/null 2>&1; then
  dnf install -y awscli2 || dnf install -y awscli
fi
aws s3 cp s3://zero-training-022118847419/qwen35-acceptance/3e130dd-2fc06364-a6c9d113/v6-formal/input/runner.sh /opt/qwen35-acceptance-v6/inputs/runner.sh --only-show-errors
echo '28aced75dbe2eb23af96814ae20eb44efa31b4b220049a7e804dbe4191440618  /opt/qwen35-acceptance-v6/inputs/runner.sh' | sha256sum -c -
chmod 700 /opt/qwen35-acceptance-v6/inputs/runner.sh
exec /opt/qwen35-acceptance-v6/inputs/runner.sh
