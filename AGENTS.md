# AGENTS.md
## Repo purpose
Automate provisioning of two NVIDIA DGX Spark nodes (`spark-6b57`, `spark-7659`) with Kubernetes + Longhorn + TRT‑LLM + Open WebUI.

## How to run locally (sandbox-friendly)
- Install deps: `just setup`
- Lint all: `just lint`
- Test roles with Molecule (Docker): `just test`
- Dry-run a play: `just check play=site.yml`

## Conventions
- Prefer `just` recipes; do not invoke `ansible-playbook` directly in CI.
- Keep tasks idempotent; PRs must pass `molecule idempotence`.
- Conventional Commits for messages.
- New roles must include a Molecule scenario and docs updates.

## Safety boundaries for agents
- Do **NOT** run against real hosts. Use Molecule + Docker only.
- If inventory is changed, update `docs/INVENTORY.md`.
- If adding Galaxy roles, update `requirements.yml` and run `just galaxy`.

## Useful commands
- `just setup`       # upgrade pip + install Python deps locally
- `just galaxy`      # install Ansible Galaxy deps from requirements.yml
- `just lint`        # yamllint + ansible-lint combo
- `just test`        # full Molecule test matrix
- `just converge`    # Molecule converge only (fast iteration)
- `just idempotence` # Molecule idempotence check
- `just destroy`     # tear down Molecule resources
- `just check play=site.yml`  # ansible-playbook --check --diff
- `just converge-play site.yml "-e foo=bar"`  # ansible-playbook real run (quote extra args)
- `just k -- get pods -A`     # kubectl passthrough using the repo default kubeconfig
- `just klogs pod=my-pod namespace=foo`  # tail pod logs via kubectl
- `just knodes`               # kubectl get nodes -o wide
