    .PHONY: all setup galaxy lint test converge idempotence destroy check fix

    PY ?= python3

    setup:
    	$(PY) -m pip install -U pip
    	$(PY) -m pip install -r requirements.txt

    galaxy:
    	@if [ -f requirements.yml ]; then ansible-galaxy install -r requirements.yml --force; else echo "No requirements.yml"; fi

    lint:
    	yamllint .
    	ansible-lint -v

    test: galaxy
    	molecule test

    converge: galaxy
    	molecule converge

    idempotence:
    	molecule idempotence

    destroy:
    	molecule destroy

    check:
    	@if [ -z "$(PLAY)" ]; then echo "Usage: make check PLAY=site.yml" && exit 2; fi
    	ansible-playbook -i inventory/hosts $(PLAY) --check --diff -vv

all: lint test
