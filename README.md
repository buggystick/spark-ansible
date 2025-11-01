# DGX Sparks — Triton + TensorRT-LLM (PP=2, TP=1)

This repo is tuned for **two DGX Sparks, one Blackwell-class GPU per node**.
We run **TensorRT-LLM** via **Triton Inference Server** across the two nodes using:

- **Tensor Parallelism (TP) = 1** (no intra-layer sharding; one GPU per node)
- **Pipeline Parallelism (PP) = 2** (layers split across the two nodes)
- **World size = 2**

It also includes:
- **NVIDIA GPU Operator** (drivers/toolkit/DCGM)
- **NVIDIA Network Operator** (RDMA / UCX device plugin) to use your **ConnectX-7** link
- **Longhorn** using a **directory** on NVMe (no reformat)
- A small **OpenAI-compatible proxy** + **Model Manager** (load/unload Triton models)
- **Open WebUI** preconfigured to talk to the proxy

Hosts (adjust in `inventory/hosts.ini`):
- primary: `spark-6b57`
- worker:  `spark-7659`
- CX7/QSFP IF: `enp1s0f0np0`

## Quick Start

1) **Edit inventory & vars**
- `inventory/hosts.ini` (hostnames, user)
- `group_vars/all.yml` (tp=1, pp=2, iface names, dirs)

2) **Add your Hugging Face token**
```bash
cp group_vars/secrets.yml.example group_vars/secrets.yml
# edit hf_token: "hf_xxx"
# (optional) ansible-vault encrypt group_vars/secrets.yml
```

3) **Bootstrap K8s + Operators + Longhorn**
```bash
ansible-playbook -i inventory/hosts.ini site.yml --tags k8s,nvidia_gpu_operator,nvidia_net_operator,longhorn
```

4) **Build engines (once per model, TP=1 PP=2)**
```bash
ansible-playbook -i inventory/hosts.ini site.yml --tags trtllm_build
```

> **Note:** The playbook now exports `HF_HUB_DISABLE_HF_TRANSFER=1` by default to avoid DNS failures on restricted clusters. Set the variable in your job spec or shell (for example `export HF_HUB_DISABLE_HF_TRANSFER=0`) before running `ansible-playbook` if you need to re-enable Hugging Face Transfer.

This produces, e.g.:
```
/srv/models/qwen235b_fp4/trtllm/engine_rank0.plan
/srv/models/qwen235b_fp4/trtllm/engine_rank1.plan
/srv/models/qwen235b_fp4/config.pbtxt
```

5) **Deploy Triton + proxy + Open WebUI**
```bash
ansible-playbook -i inventory/hosts.ini site.yml --tags triton_trtllm,openai_proxy,openwebui
```

6) **Point Open WebUI to the single endpoint**
- Base URL: `http://openai-proxy.default.svc.cluster.local:8000/v1`
- Choose the active model via the proxy’s **Model Manager**:
  ```bash
  curl "http://openai-proxy.default.svc.cluster.local:8000/admin/set_model?name=qwen235b_fp4"
  ```

## TP/PP Recap for This Hardware

- **TP=1**: one GPU per node → cannot shard within a layer.
- **PP=2**: split layers across nodes; activations flow once per token step over CX7.
- **World size=2**: one rank per node.

If you later add more nodes, increase **PP** accordingly and rebuild engines with the new `--pp-size`.

## Common Commands

```bash
kubectl get pods -A
kubectl -n openwebui port-forward svc/open-webui 8080:8080  # UI at http://localhost:8080
# Traefik ingress controller (NodePorts 32080/32443 by default)
kubectl -n traefik get svc traefik -o wide
# Longhorn UI ingress (default host: longhorn.local)
kubectl -n longhorn-system get ingress -o wide
# Access via: http://longhorn.local (point DNS/hosts to a node IP or curl -H "Host: longhorn.local" http://<node-ip>:32080)
```

## Quickstart (dev / Codex-friendly)
```bash
make setup
make lint
make test
# dry-run
make check PLAY=site.yml
```

See `AGENTS.md` for agent “house rules”. CI will run Codex review on PRs if you set `OPENAI_API_KEY` as a repo secret.



License: MIT
