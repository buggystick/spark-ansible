# DGX Sparks — vLLM (OpenAI API) on 2× DGX (PP=2, TP=1)

This repo is tuned for **two DGX Sparks, one Blackwell-class GPU per node**.
We now serve models with **vLLM** (OpenAI-compatible API) across the two nodes using Ray:

- **Tensor Parallelism (TP) = 1** (per-GPU)
- **Pipeline Parallelism (PP) = 2** (one stage per Spark)
- **World size = 2** (Ray head + 1 worker)

It also includes:
- **NVIDIA GPU Operator** (drivers/toolkit/DCGM)
- **NVIDIA Network Operator** (RDMA / UCX device plugin) to use your **ConnectX-7** link
- **Longhorn** using a **directory** on NVMe (no reformat)
- **KubeRay** operator + RayCluster spanning both nodes
- **Open WebUI** preconfigured to talk to the vLLM service

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

4) **Join DGX Sparks as workers (over SSH)**
```bash
# ensure Sparks are uncommented in inventory/hosts.ini under [workers]
export KUBECONFIG=$PWD/kubeconfig-talos.yaml
ansible-playbook -i inventory/hosts.ini site.yml --tags join --limit workers
```

5) **Deploy Ray + vLLM + Open WebUI**
```bash
ansible-playbook -i inventory/hosts.ini site.yml --tags kuberay,vllm,openwebui
```

> The play:
> - Installs KubeRay operator
> - Creates a RayCluster (head on primary, one worker)
> - Submits a RayJob that runs `vllm serve` with `TP=1, PP=2`
> - Exposes vLLM via `svc/vllm` on port 8000 (OpenAI API)
> Models pull from Hugging Face into a Longhorn-backed cache (`vllm-cache` PVC). Provide gated access by setting `hf_token` in `group_vars/secrets.yml` (creates `Secret/hf-token`).

6) **Tear everything down (optional reset)**
```bash
ansible-playbook -i inventory/hosts.ini uninstall.yml
# or use: just uninstall
```

7) **Point Open WebUI to the vLLM endpoint**
- Base URL: `http://vllm.default.svc.cluster.local:8000/v1`
- Switch models/parallelism by editing `vllm_model`, `vllm_tensor_parallel_size`, `vllm_pipeline_parallel_size` in `group_vars/all.yml` (or inventory) and rerun: `ansible-playbook -i inventory/hosts.ini site.yml --tags vllm`

## TP/PP Recap for This Hardware

- **TP=1**: one GPU per node → cannot shard within a layer.
- **PP=2**: split layers across nodes; activations flow once per token step over CX7.
- **World size=2**: one rank per node.

If you later add more nodes, increase **PP** (pipeline parallel size) accordingly by setting `vllm_pipeline_parallel_size` and rerunning the vLLM role.

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
just setup                     # install/upgrade Python deps
just lint                      # yamllint + ansible-lint
just test                      # full Molecule test matrix
# dry-run
just check play=site.yml       # ansible-playbook --check
```

### Command reference

| Command | Purpose |
| --- | --- |
| `just setup` | Upgrade pip and install `requirements.txt` into the active venv. |
| `just galaxy` | Install Ansible Galaxy collections from `requirements.yml`. |
| `just lint` | Run yamllint and ansible-lint across the repo. |
| `just test` | Execute the entire Molecule test matrix (create → verify → destroy). |
| `just converge` | Run `molecule converge` only for quick iterations. |
| `just idempotence` | Check that the converge play is idempotent. |
| `just destroy` | Tear down Molecule-created resources without re-running tests. |
| `just check play=site.yml` | Dry-run `ansible-playbook` against `inventory/hosts.ini`. |
| `just converge-play site.yml "-e foo=bar"` | Run `ansible-playbook`; optional quoted args are appended to the command. |
| `just uninstall` | Run the dedicated uninstall playbook (`uninstall.yml`). |
| `just k -- …` | Raw kubectl passthrough (uses `KUBECONFIG` or the default path). |
| `just kgp namespace=foo` | `kubectl get pods` within a namespace. |
| `just kgpa` | `kubectl get pods -A`. |
| `just klogs pod=my-pod [container=ctr]` | Tail pod logs, optionally scoping to a container. |
| `just kdesc resource=deploy name=foo namespace=bar` | Describe a Kubernetes object. |
| `just knodes` | `kubectl get nodes -o wide`. |
| `just kctx` | Show kubeconfig contexts (helpful when toggling clusters). |

### Handy kubectl helpers

Export `KUBECONFIG` if you need something other than `/etc/kubernetes/admin.conf`, then run:

```bash
just k -- get pods -A            # passthrough wrapper
just kgp namespace=openwebui     # kubectl get pods -n openwebui
just kgpa                        # kubectl get pods -A
just klogs pod=my-pod namespace=default [container=my-container]
just kdesc resource=sts name=vllm namespace=default
just knodes                      # kubectl get nodes -o wide
```

See `AGENTS.md` for agent “house rules”. CI will run Codex review on PRs if you set `OPENAI_API_KEY` as a repo secret.



License: MIT
