#!/usr/bin/env python3
"""Build TRT-LLM engines for configured models."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

try:
    from huggingface_hub import snapshot_download  # type: ignore
except Exception:  # pragma: no cover - dependency optional in container
    snapshot_download = None


# HF Transfer relies on the `hf_transfer` binary which in turn requires DNS
# access to `transfer.xethub.hf.co`.  The build cluster we target cannot
# resolve that domain, leading to repeated download failures like the
# following:
#
#   Reqwest(reqwest::Error { kind: Request, url: ... transfer.xethub.hf.co ...
#          error: "failed to lookup address information: Temporary failure in
#          name resolution" })
#
# When the environment variable is unset the hub automatically opts into HF
# Transfer.  Force-disable it by default so we fall back to the regular
# HTTPS endpoints that *are* reachable in the cluster.  Operators can re-enable
# HF Transfer by exporting HF_HUB_DISABLE_HF_TRANSFER=0 before launching the
# job if their environment supports it.
os.environ.setdefault("HF_HUB_DISABLE_HF_TRANSFER", "1")


def log(message: str) -> None:
    """Log a message with flushing enabled for real-time feedback."""
    print(message, flush=True)


def run(cmd: Iterable[str]) -> None:
    """Run a command, streaming progress and raising on failure."""
    cmd_list = list(cmd)
    log(">> " + " ".join(cmd_list))
    result = subprocess.run(cmd_list)
    if result.returncode != 0:
        sys.exit(result.returncode)


def resolve_checkpoint(hf_id: str, model_name: str) -> str:
    """Return a local directory containing the HF checkpoint."""
    cache_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    cache_root.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or None

    if snapshot_download is not None:
        return snapshot_download(
            repo_id=hf_id,
            cache_dir=str(cache_root),
            token=token,
        )

    slug = hf_id.replace("/", "--")
    snapshots_dir = cache_root / "hub" / f"models--{slug}" / "snapshots"
    if not snapshots_dir.is_dir():
        raise RuntimeError(
            f"Checkpoint for {hf_id} not found at {snapshots_dir}; install huggingface_hub in the TRT-LLM image."
        )
    revisions = sorted(p for p in snapshots_dir.iterdir() if p.is_dir())
    if not revisions:
        raise RuntimeError(f"No snapshots available for {hf_id} in {snapshots_dir}.")
    latest = revisions[-1]
    log(f"Using cached Hugging Face snapshot for {model_name} ({hf_id}): {latest}")
    return str(latest)


def main() -> None:
    tp = int(os.environ.get("TP", "1"))
    pp = int(os.environ.get("PP", "1"))
    world = tp * pp

    models = json.loads(os.environ.get("MODELS_JSON", "[]"))
    if not models:
        log("No models specified in MODELS_JSON; nothing to build.")
        return

    total_models = len(models)
    models_to_build = models
    job_index_raw = os.environ.get("JOB_COMPLETION_INDEX")
    if job_index_raw is not None:
        try:
            job_index = int(job_index_raw)
        except ValueError:
            log(
                f"Invalid JOB_COMPLETION_INDEX={job_index_raw!r}; running serial build for all {total_models} model(s)."
            )
        else:
            if 0 <= job_index < total_models:
                selected = models[job_index]
                models_to_build = [selected]
                log(
                    "JOB_COMPLETION_INDEX="
                    f"{job_index}: building model {selected['name']} ({job_index + 1}/{total_models})."
                )
            else:
                log(
                    f"JOB_COMPLETION_INDEX={job_index} out of range for {total_models} model(s); running serial build."
                )

    log(
        f"Preparing to build {len(models_to_build)} model(s) with TP={tp}, PP={pp}, WORLD={world}."
    )

    for idx, model in enumerate(models_to_build, start=1):
        name = model["name"]
        hf_id = model["hf_id"]
        out_dir = Path("/models") / name / "trtllm"
        out_dir.mkdir(parents=True, exist_ok=True)

        log(f"[{idx}/{len(models_to_build)}] Resolving checkpoint for {name} ({hf_id})...")
        checkpoint_dir = resolve_checkpoint(hf_id, name)
        log(f"[{idx}/{len(models_to_build)}] Checkpoint directory: {checkpoint_dir}")

        cmd = [
            "trtllm-build",
            "--checkpoint_dir",
            checkpoint_dir,
            "--output_dir",
            str(out_dir),
            "--world_size",
            str(world),
            "--tp_size",
            str(tp),
            "--pp_size",
            str(pp),
            "--gpus_per_node",
            "1",
            "--max_input_len",
            "8192",
            "--max_seq_len",
            str(8192 + 1024),
        ]

        start = time.time()
        run(cmd)
        elapsed = time.time() - start

        config_lines = [
            f'name: "{name}"',
            'backend: "tensorrtllm"',
            'max_batch_size: 32',
            f'parameters: {{ key: "tensor_parallel_size" value: {{ string_value: "{tp}" }} }}',
            f'parameters: {{ key: "pipeline_parallel_size" value: {{ string_value: "{pp}" }} }}',
            'instance_group [{ kind: KIND_GPU, count: 1 }]',
        ]
        config = "\n".join(config_lines) + "\n"

        model_dir = Path("/models") / name
        model_dir.mkdir(parents=True, exist_ok=True)
        with (model_dir / "config.pbtxt").open("w") as cfg:
            cfg.write(config)

        log(f"== Completed {name} -> {out_dir} in {elapsed:.1f}s")

    log("All builds done.")


if __name__ == "__main__":
    main()
