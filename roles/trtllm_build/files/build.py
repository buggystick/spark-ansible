#!/usr/bin/env python3
"""Minimal TRT-LLM build helper that leans on the container tooling."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

try:
    from huggingface_hub import snapshot_download  # type: ignore
except Exception as exc:  # pragma: no cover - container should provide the dep
    print(f"huggingface_hub is required in the TensorRT-LLM container: {exc}", file=sys.stderr)
    sys.exit(2)


MODELS_ROOT = Path("/models")
CHECKPOINT_SUBDIR = "checkpoint"
ENGINE_SUBDIR = "trtllm"


def _load_models() -> List[Dict[str, Any]]:
    raw = os.environ.get("MODELS_JSON", "[]")
    try:
        models = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"MODELS_JSON is not valid JSON: {exc}: {raw[:200]}", file=sys.stderr)
        sys.exit(2)

    if not isinstance(models, list):
        print("MODELS_JSON must be a JSON array", file=sys.stderr)
        sys.exit(2)

    completion_idx = os.environ.get("JOB_COMPLETION_INDEX")
    if completion_idx is not None:
        try:
            idx = int(completion_idx)
        except ValueError:
            print(f"Ignoring invalid JOB_COMPLETION_INDEX={completion_idx!r}")
        else:
            if 0 <= idx < len(models):
                models = [models[idx]]
                print(f"Selected model index {idx} from JOB_COMPLETION_INDEX")
            else:
                print(f"JOB_COMPLETION_INDEX {idx} out of range for {len(models)} model(s)")

    return models


def _download_checkpoint(model: Dict[str, Any], token: str | None) -> Path:
    repo_id = model["hf_id"]
    name = model.get("name") or repo_id.split("/")[-1]
    revision = model.get("revision")

    local_dir = MODELS_ROOT / name / CHECKPOINT_SUBDIR
    local_dir.mkdir(parents=True, exist_ok=True)

    download_args: Dict[str, Any] = {
        "repo_id": repo_id,
        "local_dir": str(local_dir),
        "local_dir_use_symlinks": False,
    }
    if revision:
        download_args["revision"] = revision
    if token:
        download_args["token"] = token

    print(f"Downloading {repo_id} -> {local_dir}")
    snapshot_download(**download_args)
    return local_dir


def _build_engine(checkpoint_dir: Path, model: Dict[str, Any], tp: str | None, pp: str | None) -> Path:
    name = model.get("name") or checkpoint_dir.parent.name
    output_dir = MODELS_ROOT / name / ENGINE_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd: List[str] = [
        "trtllm-build",
        "--checkpoint_dir",
        str(checkpoint_dir),
        "--output_dir",
        str(output_dir),
    ]

    max_input = str(model.get("max_input_len", 8192))
    max_seq = str(model.get("max_seq_len", 9216))
    cmd += ["--max_input_len", max_input, "--max_seq_len", max_seq]

    if tp:
        cmd += ["--tensor_parallel_size", tp]
    if pp:
        cmd += ["--pipeline_parallel_size", pp]

    extra_args: Iterable[str] = model.get("extra_args", [])
    cmd.extend(str(arg) for arg in extra_args)

    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)
    return output_dir


def _write_triton_config(model: Dict[str, Any], tp: str | None, pp: str | None) -> None:
    name = model.get("name") or model["hf_id"].split("/")[-1]
    model_root = MODELS_ROOT / name
    model_root.mkdir(parents=True, exist_ok=True)
    config = model_root / "config.pbtxt"

    tp_val = tp or "1"
    pp_val = pp or "1"
    contents = "\n".join(
        [
            f'name: "{name}"',
            'backend: "tensorrtllm"',
            'max_batch_size: 32',
            f'parameters: {{ key: "tensor_parallel_size" value: {{ string_value: "{tp_val}" }} }}',
            f'parameters: {{ key: "pipeline_parallel_size" value: {{ string_value: "{pp_val}" }} }}',
            'instance_group [{ kind: KIND_GPU, count: 1 }]',
        ]
    )
    config.write_text(contents + "\n")



def main() -> None:
    models = _load_models()
    if not models:
        print("No models to build; exiting.")
        return

    token = os.environ.get("HF_TOKEN")
    tp = os.environ.get("TP")
    pp = os.environ.get("PP")

    for model in models:
        if not isinstance(model, dict) or "hf_id" not in model:
            print(f"Skipping invalid model entry: {model}", file=sys.stderr)
            continue

        checkpoint_dir = _download_checkpoint(model, token)
        engine_dir = _build_engine(checkpoint_dir, model, tp, pp)
        _write_triton_config(model, tp, pp)
        print(f"Completed build for {model.get('name') or model['hf_id']} -> {engine_dir}")


if __name__ == "__main__":
    main()
