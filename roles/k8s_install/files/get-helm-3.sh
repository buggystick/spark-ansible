#!/usr/bin/env bash
set -euo pipefail
# Simple wrapper to install Helm v3 using the official installer script.
# This keeps pipeline logic out of Ansible YAML.
curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
