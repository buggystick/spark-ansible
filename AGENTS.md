# AGENTS.md
## Repo purpose
Automate provisioning of two NVIDIA DGX Spark nodes (`spark-6b57`, `spark-7659`) with Kubernetes + Longhorn + TRT‑LLM + Open WebUI.

## How to run locally (sandbox-friendly)
- Install deps: `python3 -m pip install -U pip && pip install -r requirements.txt`
- Lint all: `make lint`
- Test roles with Molecule (Docker): `make test`
- Dry-run a play: `make check PLAY=site.yml`

## Conventions
- Prefer `make` targets; do not invoke `ansible-playbook` directly in CI.
- Keep tasks idempotent; PRs must pass `molecule idempotence`.
- Conventional Commits for messages.
- New roles must include a Molecule scenario and docs updates.

## Safety boundaries for agents
- Do **NOT** run against real hosts. Use Molecule + Docker only.
- If inventory is changed, update `docs/INVENTORY.md`.
- If adding Galaxy roles, update `requirements.yml` and run `make galaxy`.

## Useful commands
- `make setup`     # install Python deps
- `make galaxy`    # install Ansible Galaxy deps
- `make converge`  # run Molecule converge
- `make idempotence`
- `make destroy`
- `make check PLAY=site.yml`  # ansible-playbook --check --diff
