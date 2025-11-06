#!/usr/bin/env python3
"""Minimal TRT-LLM build helper that leans on the container tooling."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import logging
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

try:
    from huggingface_hub import snapshot_download  # type: ignore
except Exception as exc:  # pragma: no cover - container should provide the dep
    print(f"huggingface_hub is required in the TensorRT-LLM container: {exc}", file=sys.stderr)
    sys.exit(2)


MODELS_ROOT = Path("/models")
CHECKPOINT_SUBDIR = "checkpoint"
ENGINE_SUBDIR = "trtllm"


logger = logging.getLogger("trtllm_build")


def _setup_logging() -> None:
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    logger.debug("Logging initialised with level %s", logging.getLevelName(level))


def _log_gpu_sanity() -> None:
    cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_devices:
        logger.info("CUDA_VISIBLE_DEVICES=%s", cuda_devices)
    else:
        logger.info("CUDA_VISIBLE_DEVICES is not set")

    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        logger.warning("nvidia-smi binary not found in PATH; skipping GPU probe")
        return

    commands = ((nvidia_smi, "-L"), (nvidia_smi,))
    for cmd in commands:
        logger.info("Probing GPU visibility with: %s", " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0:
            stdout = proc.stdout.strip() or "<no output>"
            logger.info("%s output:\n%s", os.path.basename(cmd[0]), stdout)
            if proc.stderr:
                logger.debug("%s stderr:\n%s", os.path.basename(cmd[0]), proc.stderr.strip())
        else:
            stdout = proc.stdout.strip()
            stderr = proc.stderr.strip()
            details = stdout or "<no stdout>"
            if stderr:
                details = f"{details}\n[stderr]\n{stderr}"
            logger.warning(
                "%s exited with code %s:\n%s",
                os.path.basename(cmd[0]),
                proc.returncode,
                details,
            )


def _load_models() -> List[Dict[str, Any]]:
    raw = os.environ.get("MODELS_JSON", "[]")
    try:
        models = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("MODELS_JSON is not valid JSON: %s: %s", exc, raw[:200])
        sys.exit(2)

    if not isinstance(models, list):
        logger.error("MODELS_JSON must be a JSON array")
        sys.exit(2)

    completion_idx = os.environ.get("JOB_COMPLETION_INDEX")
    if completion_idx is not None:
        try:
            idx = int(completion_idx)
        except ValueError:
            logger.warning("Ignoring invalid JOB_COMPLETION_INDEX=%r", completion_idx)
        else:
            if 0 <= idx < len(models):
                models = [models[idx]]
                logger.info("Selected model index %s from JOB_COMPLETION_INDEX", idx)
            else:
                logger.warning(
                    "JOB_COMPLETION_INDEX %s out of range for %s model(s)",
                    idx,
                    len(models),
                )

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

    logger.info("Downloading %s to %s", repo_id, local_dir)
    snapshot_download(**download_args)
    return local_dir


@lru_cache(maxsize=1)
def _supported_trtllm_flags() -> Set[str]:
    """Return the CLI flags exposed by ``trtllm-build``.

    We shell out to ``trtllm-build --help`` once so we can avoid passing
    parameters that are unsupported in the currently bundled version.
    """

    help_cmd = ["trtllm-build", "--help"]
    logger.debug("Inspecting trtllm-build flags via: %s", " ".join(help_cmd))
    proc = subprocess.run(help_cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        logger.warning(
            "Unable to inspect trtllm-build flags (exit code %s); assuming none",
            proc.returncode,
        )
        if proc.stderr:
            logger.debug("trtllm-build --help stderr:\n%s", proc.stderr.strip())
        return set()

    flags = set(re.findall(r"--([\w-]+)", proc.stdout))
    logger.debug("trtllm-build supports flags: %s", ", ".join(sorted(flags)) or "<none>")
    return flags


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

    supported_flags = _supported_trtllm_flags()

    if tp:
        flag = "tensor_parallel_size"
        if flag in supported_flags:
            cmd += [f"--{flag}", tp]
        else:
            logger.info("Skipping unsupported trtllm-build flag --%s", flag)
    if pp:
        flag = "pipeline_parallel_size"
        if flag in supported_flags:
            cmd += [f"--{flag}", pp]
        else:
            logger.info("Skipping unsupported trtllm-build flag --%s", flag)

    extra_args: Iterable[str] = model.get("extra_args", [])
    cmd.extend(str(arg) for arg in extra_args)

    logger.info("Starting TensorRT-LLM build: %s", " ".join(cmd))
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as exc:
        logger.error("trtllm-build failed with exit code %s", exc.returncode)
        raise
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
    logger.info("Wrote Triton config for %s to %s", name, config)



def main() -> None:
    _setup_logging()
    logger.info("Starting TRT-LLM build helper")
    _log_gpu_sanity()

    models = _load_models()
    if not models:
        logger.info("No models to build; exiting")
        return

    logger.info("Planning to build %s model(s)", len(models))

    token = os.environ.get("HF_TOKEN")
    tp = os.environ.get("TP")
    pp = os.environ.get("PP")
    logger.info("Tensor parallel size: %s | Pipeline parallel size: %s", tp or "1", pp or "1")

    for model in models:
        if not isinstance(model, dict) or "hf_id" not in model:
            logger.error("Skipping invalid model entry: %s", model)
            continue

        model_name = model.get("name") or model["hf_id"]
        logger.info("Processing model %s", model_name)

        checkpoint_dir = _download_checkpoint(model, token)
        engine_dir = _build_engine(checkpoint_dir, model, tp, pp)
        _write_triton_config(model, tp, pp)
        logger.info("Completed build for %s -> %s", model_name, engine_dir)


if __name__ == "__main__":
    main()
