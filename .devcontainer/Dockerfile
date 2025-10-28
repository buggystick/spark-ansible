FROM mcr.microsoft.com/devcontainers/base:ubuntu-24.04

RUN apt-get update &&         apt-get install -y --no-install-recommends           python3 python3-pip python3-venv python3-dev           git curl jq make docker.io           && rm -rf /var/lib/apt/lists/*

WORKDIR /workspaces/spark-ansible
