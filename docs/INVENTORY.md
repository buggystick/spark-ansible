# Inventory Notes

This repo’s CI and Codex agents must **not** run against production hosts.
Use Molecule (Docker) for validation.

When you are ready to run for real:

- Use your desktop (`dareed-gentoo`) as the lone `[primary]` control plane with
  `ansible_connection=local`.
- Keep DGX boxes (`spark-6b57`, `spark-7659`) under `[workers]` so GPU jobs land
  only there. The playbook now re-applies the control-plane taint automatically
  so no workloads schedule on the desktop.
- Create a secure private inventory (or ansible-vault encrypted copy) if host
  details differ from the defaults, and ensure SSH/sudo/network prerequisites are met.
