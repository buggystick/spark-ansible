#!/usr/bin/env bash
set -euo pipefail

TP_VAL=${TP:-1}
PP_VAL=${PP:-1}
WORLD_VAL=${WORLD:-1}

echo "TP=${TP_VAL} PP=${PP_VAL} WORLD=${WORLD_VAL}"
echo "Models dir: /models"

echo "[gpu-check] Probing GPU availability..."

# 1) Prefer TensorRT's trtexec if present (ships in many TRT images)
if command -v trtexec >/dev/null 2>&1; then
  echo "[gpu-check] Using trtexec..."
  if trtexec --device=0 --dumpDeviceInfo --iterations=1 >/tmp/trtexec.out 2>&1; then
    echo "[gpu-check] trtexec OK:"
    sed -n '1,80p' /tmp/trtexec.out | sed 's/^/[trtexec] /'
    GPU_CHECK_OK=1
  else
    echo "[gpu-check] trtexec failed (non-fatal):"
    sed -n '1,60p' /tmp/trtexec.out | sed 's/^/[trtexec] /'
  fi
fi

# 2) Fallback: NVML via libnvidia-ml.so.1 (no python packages needed)
if [ -z "${GPU_CHECK_OK:-}" ]; then
  set +e
  python3 - <<'PY'
import ctypes, ctypes.util, sys
def nvml_probe():
    try:
        lib = ctypes.CDLL(ctypes.util.find_library('nvidia-ml') or 'libnvidia-ml.so.1')
        # int nvmlInit_v2(void)
        assert lib.nvmlInit_v2() == 0
        count = ctypes.c_uint()
        # int nvmlDeviceGetCount_v2(unsigned int*)
        assert lib.nvmlDeviceGetCount_v2(ctypes.byref(count)) == 0
        print(f"[gpu-check] NVML sees {count.value} device(s)")
        if count.value == 0:
            return 1
        # Print a couple device names
        namebuf = (ctypes.c_char * 96)()
        for i in range(min(4, count.value)):
            h = ctypes.c_void_p()
            assert lib.nvmlDeviceGetHandleByIndex_v2(i, ctypes.byref(h)) == 0
            assert lib.nvmlDeviceGetName(h, namebuf, 96) == 0
            print(f"[gpu-check] NVML device {i}: {namebuf.value.decode()}")
        return 0
    except Exception as e:
        print(f"[gpu-check] NVML probe failed: {e}")
        return 2
sys.exit(nvml_probe())
PY
  nvml_status=$?
  set -e
  case ${nvml_status:-99} in
    0) GPU_CHECK_OK=1 ;;
    1) echo "[gpu-check] NVML reported 0 devices";;
    2) echo "[gpu-check] NVML not available in container (non-fatal)";;
    *) echo "[gpu-check] NVML probe exited with unexpected status ${nvml_status}";;
  esac
fi

# 3) Fallback: CUDA Driver API via libcuda.so.1
if [ -z "${GPU_CHECK_OK:-}" ]; then
  set +e
  python3 - <<'PY'
import ctypes, ctypes.util, sys
def cuda_probe():
    try:
        lib = ctypes.CDLL(ctypes.util.find_library('cuda') or 'libcuda.so.1')
        # int cuInit(unsigned int)
        if lib.cuInit(0) != 0:
            print("[gpu-check] cuInit failed")
            return 2
        count = ctypes.c_int(0)
        # int cuDeviceGetCount(int*)
        if lib.cuDeviceGetCount(ctypes.byref(count)) != 0:
            print("[gpu-check] cuDeviceGetCount failed")
            return 2
        print(f"[gpu-check] CUDA Driver sees {count.value} device(s)")
        return 0 if count.value > 0 else 1
    except Exception as e:
        print(f"[gpu-check] CUDA driver probe failed: {e}")
        return 2
sys.exit(cuda_probe())
PY
  cuda_status=$?
  set -e
  case ${cuda_status:-99} in
    0) GPU_CHECK_OK=1 ;;
    1) echo "[gpu-check] CUDA driver reported 0 devices";;
    2) echo "[gpu-check] CUDA driver libs not visible (non-fatal)";;
    *) echo "[gpu-check] CUDA driver probe exited with unexpected status ${cuda_status}";;
  esac
fi

# Final decision: print /dev nodes for extra context; don't hard-fail the job here.
ls -l /dev/nvidia* 2>/dev/null || echo "[gpu-check] No /dev/nvidia* nodes visible"

if [ -n "${GPU_CHECK_OK:-}" ]; then
  echo "[gpu-check] ✔ GPU appears available to the container"
else
  echo "[gpu-check] ✖ No GPU detected by trtexec/NVML/CUDA driver"
  # If you want to fail early, uncomment the next line:
  exit 42
fi

# Ensure required Python deps are present without downgrading TensorRT-LLM pins.
# - tqdm / tqdm-loggable: nice-to-have progress bars for build.py
# - huggingface_hub: required for fetching checkpoints
# - TensorRT-LLM 1.2.0rc1 requires very specific versions for core ML deps; the
#   resolver otherwise drags in bleeding-edge releases that violate those caps.
COMMON_PIP_ARGS=(
  --upgrade
  --upgrade-strategy eager
  --no-cache-dir
)

python3 -m pip install "${COMMON_PIP_ARGS[@]}" \
  'datasets==3.1.0' \
  'numpy==1.26.4' \
  'pillow==10.3.0' \
  'setuptools<80' \
  'torch==2.8.0' \
  'tqdm>=4.66.0' \
  'tqdm-loggable>=0.2' \
  'huggingface_hub>=0.24.0,<1.0.0' \
  'transformers==4.56.0'

# TensorRT-LLM 1.2.0rc1 insists on ModelOpt ~=0.33.0. The optional [hf] extra for
# that release requires transformers<4.54, which conflicts with TRT-LLM's own
# 4.56.0 pin. Install the core package along with the handful of Hugging
# Face-oriented utilities that the extra would have provided, but keep
# transformers at the version mandated by TRT-LLM.
python3 -m pip install "${COMMON_PIP_ARGS[@]}" \
  'nvidia-modelopt==0.33.0' \
  'accelerate<1.2.0' \
  'safetensors>=0.4.3,<1.0.0' \
  'sentencepiece>=0.1.99'

python3 /opt/trtllm/build.py "$@"
