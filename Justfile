set shell := ["bash", "-euo", "pipefail", "-c"]

py := env_var_or_default("PY", "python3")
inventory := "inventory/hosts.ini"
kubeconfig := env_var_or_default("KUBECONFIG", "/etc/kubernetes/admin.conf")

# Run the default validation combo (lint + molecule test)
default: lint test

# Install Python dependencies into the active environment
setup:
	{{py}} -m pip install -U pip
	{{py}} -m pip install -r requirements.txt

# Install Galaxy collections from requirements.yml (if present)
galaxy:
	if [ -f requirements.yml ]; then \
		ansible-galaxy install -r requirements.yml --force; \
	else \
		echo "No requirements.yml"; \
	fi

# Run yamllint and ansible-lint across the repo
lint:
	yamllint .
	ansible-lint -v

# End-to-end Molecule test matrix
test: galaxy
	molecule test

# Molecule converge only (no verify/destroy)
converge: galaxy
	molecule converge

# Molecule idempotence check only
idempotence: galaxy
	molecule idempotence

# Destroy Molecule resources without re-running tests
destroy: galaxy
	molecule destroy

# Run ansible-playbook in --check (dry-run) mode
check play="site.yml":
	ansible-playbook -i {{inventory}} {{play}} --check --diff -vv

# Run ansible-playbook for real against the inventory
converge-play play="site.yml":
	ansible-playbook -i {{inventory}} {{play}}

# --- Kubernetes helpers ---

# Generic pass-through to kubectl (usage: just k -- get pods -A)
k +cmd:
	kubectl --kubeconfig {{kubeconfig}} {{cmd}}

# Get pods in a namespace (default namespace=default)
kgp namespace="default":
	kubectl --kubeconfig {{kubeconfig}} -n {{namespace}} get pods

# Get pods across all namespaces
kgpa:
	kubectl --kubeconfig {{kubeconfig}} get pods -A

# Tail logs for a pod (requires pod, optional container)
klogs pod namespace="default" container="":
	if [ -z "{{pod}}" ]; then echo "pod parameter is required" >&2; exit 1; fi
	if [ -n "{{container}}" ]; then \
		kubectl --kubeconfig {{kubeconfig}} -n {{namespace}} logs {{pod}} -c {{container}} -f; \
	else \
		kubectl --kubeconfig {{kubeconfig}} -n {{namespace}} logs {{pod}} -f; \
	fi

# Describe a resource in a namespace (usage: just kdesc deployment my-deploy namespace=custom)
kdesc resource name namespace="default":
	kubectl --kubeconfig {{kubeconfig}} -n {{namespace}} describe {{resource}} {{name}}

# Quick node status
knodes:
	kubectl --kubeconfig {{kubeconfig}} get nodes -o wide

# Current context overview
kctx:
	kubectl --kubeconfig {{kubeconfig}} config get-contexts
