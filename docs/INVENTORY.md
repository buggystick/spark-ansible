# Inventory Notes

This repo’s CI and Codex agents must **not** run against production hosts.
Use Molecule (Docker) for validation.

When you are ready to run for real, create a secure inventory (private)
and ensure SSH, sudo, and network prerequisites are satisfied. Keep
sensitive inventories out of the repo or encrypted (ansible-vault).
