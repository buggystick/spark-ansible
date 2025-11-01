#!/usr/bin/env python3
"""
Rich-status TRT-LLM engine builder.

Features:
- Timestamped logs & 30s heartbeats for long operations
- Environment summary: arch, python, driver libs, /dev/nvidia*
- Per-file progress bar when downloading from Hugging Face (uses tqdm if present)
- Live, line-by-line streaming of `trtllm-build` output
- Disk space checks on /models before/after each model
- JOB_COMPLETION_INDEX to support parallel completions-style Jobs
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from shutil import disk_usage
from typing import Iterable, List, Optional

# --- Optional deps from huggingface_hub ---------------------------------------
try:
    from huggingface_hub import snapshot_download, hf_hub_download, list_repo_files  # type: ignore
    HF_AVAILABLE = True
except Exception:
    snapshot_download = None  # type: ignore
    hf_hub_download = None  # type: ignore
    list_repo_files = None  # type: ignore
    HF_AVAILABLE = False

# tqdm for progress bars (optional)
try:
    from tqdm.auto import tqdm  # type: ignore
    TQDM_AVAILABLE = True
except Exception:
    tqdm = None  # type: ignore
    TQDM_AVAILABLE = False


# ==============================================================================
# Logging & plumbing
# ==============================================================================

def _ts() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S%z")


def log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


class Heartbeat:
    """Periodically print a heartbeat while long work is in progress."""
    def __init__(self, label: str, interval_s: int = 30) -> None:
        self.label = label
        self.interval = interval_s
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._start = time.time()

    def _run(self) -> None:
        tick = 1
        while not self._stop.wait(self.interval):
            elapsed = int(time.time() - self._start)
            log(f"… {self.label} still running (elapsed {elapsed}s) [tick {tick}]")
            tick += 1

    def start(self) -> None:
        self._t.start()

    def stop(self) -> None:
        self._stop.set()
        self._t.join(timeout=2)


def run_stream(cmd: Iterable[str], label: str) -> None:
    """Run a command with live stdout and a heartbeat. Raise if rc != 0."""
    cmd_list = list(cmd)
    log(">> " + " ".join(cmd_list))
    hb = Heartbeat(label)
    hb.start()
    try:
        proc = subprocess.Popen(
            cmd_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        last_lines: List[str] = []
        for line in proc.stdout:
            line = line.rstrip("\n")
            if line:
                print(line, flush=True)
                last_lines.append(line)
                if len(last_lines) > 50:
                    last_lines.pop(0)
        proc.wait()
        if proc.returncode != 0:
            log(f"[ERROR] Command failed (rc={proc.returncode}). Last output lines:")
            for l in last_lines[-15:]:
                print("  " + l, flush=True)
            sys.exit(proc.returncode)
    finally:
        hb.stop()


# ==============================================================================
# Hugging Face download helpers (with progress)
# ==============================================================================

def _env_token() -> Optional[str]:
    tok = os.environ.get("HF_TOKEN")
    return tok if tok and tok.strip() else None


def _hf_cache_root() -> Path:
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))


def download_repo_with_progress(repo_id: str, cache_dir: str, token: Optional[str]) -> str:
    """
    Download a HF repo with a per-file progress bar (tqdm if available).
    This is resilient when `snapshot_download` is quiet or tqdm isn't auto-enabled.

    Returns the local snapshot directory.
    """
    if not HF_AVAILABLE:
        raise RuntimeError(
            "huggingface_hub is not available in this container. "
            "Install it or ensure the weights are pre-cached."
        )

    # Try list_repo_files to get a manifest. If it fails (e.g., private repo without token),
    # fall back to snapshot_download with a heartbeat only.
    files: List[str] = []
    try:
        if list_repo_files is not None:
            files = list(list_repo_files(repo_id=repo_id, token=token))
    except Exception as e:
        log(f"[download] list_repo_files failed (continuing with snapshot): {e}")

    # If we have a file list, do per-file downloads to show progress.
    if files:
        log(f"[download] {repo_id}: {len(files)} files listed; starting per-file download")
        progress_iter = tqdm(files, desc=f"Downloading {repo_id}", unit="file") if TQDM_AVAILABLE else files
        last_path = ""
        for f in progress_iter:
            try:
                last_path = hf_hub_download(  # type: ignore
                    repo_id=repo_id,
                    filename=f,
                    cache_dir=cache_dir,
                    token=token,
                    force_download=False,
                    local_files_only=False,
                )
            except Exception as e:
                log(f"[download] warning: failed to fetch {f}: {e}")
        # After fetching files, ask snapshot_download to resolve the snapshot dir quickly
        hb = Heartbeat(f"finalizing snapshot for {repo_id}")
        hb.start()
        try:
            sd = snapshot_download(  # type: ignore
                repo_id=repo_id,
                cache_dir=cache_dir,
                token=token,
                local_files_only=True,  # already fetched; just unify
            )
        finally:
            hb.stop()
        return sd

    # Fallback: snapshot_download (with heartbeat)
    log(f"[download] {repo_id}: snapshot_download fallback (no file list)")
    hb = Heartbeat(f"downloading {repo_id}")
    hb.start()
    t0 = time.time()
    try:
        sd = snapshot_download(  # type: ignore
            repo_id=repo_id,
            cache_dir=cache_dir,
            token=token,
        )
    finally:
        hb.stop()
    log(f"[download] {repo_id}: completed in {time.time() - t0:.1f}s")
    return sd


def resolve_checkpoint(hf_id: str, model_name: str) -> str:
    """Return a local directory containing the HF checkpoint (with visible progress)."""
    cache_root = _hf_cache_root()
    cache_root.mkdir(parents=True, exist_ok=True)
    token = _env_token()

    # Prefer per-file progress flow if we have huggingface_hub
    if HF_AVAILABLE:
        log(
            f"Resolving HF snapshot for {model_name} ({hf_id}) "
            f"(token={'yes' if token else 'no'}, tqdm={'yes' if TQDM_AVAILABLE else 'no'})"
        )
        return download_repo_with_progress(hf_id, str(cache_root), token)

    # No huggingface_hub: try to locate a cached snapshot the "classic" way
    slug = hf_id.replace("/", "--")
    snapshots_dir = cache_root / "hub" / f"models--{slug}" / "snapshots"
    if not snapshots_dir.is_dir():
        raise RuntimeError(
            f"Checkpoint for {hf_id} not found at {snapshots_dir}; "
            f"install huggingface_hub or pre-stage the weights."
        )
    revisions = sorted(p for p in snapshots_dir.iterdir() if p.is_dir())
    if not revisions:
        raise RuntimeError(f"No snapshots available for {hf_id} in {snapshots_dir}.")
    latest = revisions[-1]
    log(f"Using cached Hugging Face snapshot for {model_name} ({hf_id}): {latest}")
    return str(latest)


# ==============================================================================
# Main build
# ==============================================================================

def _driver_visibility_hint() -> None:
    # Quick driver visibility hints (best-effort; do not fail)
    try:
        out = subprocess.run(["/sbin/ldconfig", "-p"], capture_output=True, text=True)
        if out.returncode == 0:
            for lib in ("libcuda.so.1", "libnvidia-ml.so.1"):
                present = (lib in out.stdout)
                log(f"Driver lib visible: {lib}: {'yes' if present else 'no'}")
    except Exception:
        pass
    try:
        devs = os.listdir("/dev")
        ndevs = [d for d in devs if d.startswith("nvidia")]
        log(f"/dev nodes: {ndevs if ndevs else 'none'}")
    except Exception:
        pass


def main() -> None:
    tp = int(os.environ.get("TP", "1"))
    pp = int(os.environ.get("PP", "1"))
    world = tp * pp

    models_json = os.environ.get("MODELS_JSON", "[]")
    try:
        models = json.loads(models_json)
    except Exception as e:
        log(f"[ERROR] MODELS_JSON is not valid JSON ({e}); value: {models_json[:200]}…")
        sys.exit(2)

    if not models:
        log("No models specified in MODELS_JSON; nothing to build.")
        return

    # Environment summary
    log(f"Node arch={platform.machine()} Python={platform.python_version()} "
        f"TP={tp} PP={pp} WORLD={world}")
    _driver_visibility_hint()

    total_models = len(models)
    models_to_build = models

    # Completions-style selection for parallel jobs
    job_index_raw = os.environ.get("JOB_COMPLETION_INDEX")
    if job_index_raw is not None:
        try:
            job_index = int(job_index_raw)
        except ValueError:
            log(f"Invalid JOB_COMPLETION_INDEX={job_index_raw!r}; running all {total_models} model(s).")
        else:
            if 0 <= job_index < total_models:
                selected = models[job_index]
                models_to_build = [selected]
                log(f"JOB_COMPLETION_INDEX={job_index}: building model {selected['name']} "
                    f"({job_index + 1}/{total_models}).")
            else:
                log(f"JOB_COMPLETION_INDEX={job_index} out of range for {total_models} model(s); building all.")

    log(f"Preparing to build {len(models_to_build)} model(s) with TP={tp}, PP={pp}, WORLD={world}.")

    for idx, model in enumerate(models_to_build, start=1):
        name = model["name"]
        hf_id = model["hf_id"]
        out_dir = Path("/models") / name / "trtllm"
        out_dir.mkdir(parents=True, exist_ok=True)

        log(f"[{idx}/{len(models_to_build)}] Resolving checkpoint for {name} ({hf_id})…")
        checkpoint_dir = resolve_checkpoint(hf_id, name)
        log(f"[{idx}/{len(models_to_build)}] Checkpoint directory: {checkpoint_dir}")

        # Disk space before
        try:
            total_b, used_b, free_b = disk_usage("/models")
            log(f"[{idx}] /models free: {free_b/1e9:.1f} GB (total {total_b/1e9:.1f} GB)")
        except Exception:
            pass

        # Build command (use TRT-LLM CLI that your container provides)
        cmd = [
            "trtllm-build",
            "--checkpoint_dir", checkpoint_dir,
            "--output_dir", str(out_dir),
            "--world_size", str(world),
            "--tp_size", str(tp),
            "--pp_size", str(pp),
            "--gpus_per_node", "1",
            "--max_input_len", "8192",
            "--max_seq_len", str(8192 + 1024),
        ]

        t0 = time.time()
        run_stream(cmd, label=f"trtllm-build {name}")
        elapsed = time.time() - t0

        # Write minimal Triton config.pbtxt
        cfg_lines = [
            f'name: "{name}"',
            'backend: "tensorrtllm"',
            'max_batch_size: 32',
            f'parameters: {{ key: "tensor_parallel_size" value: {{ string_value: "{tp}" }} }}',
            f'parameters: {{ key: "pipeline_parallel_size" value: {{ string_value: "{pp}" }} }}',
            'instance_group [{ kind: KIND_GPU, count: 1 }]',
        ]
        model_dir = Path("/models") / name
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "config.pbtxt").write_text("\n".join(cfg_lines) + "\n")

        # Disk space after
        try:
            total_b, used_b, free_b = disk_usage("/models")
            log(f"[{idx}] /models free after: {free_b/1e9:.1f} GB")
        except Exception:
            pass

        log(f"== Completed {name} -> {out_dir} in {elapsed:.1f}s")

    log("All builds done.")


if __name__ == "__main__":
    main()
