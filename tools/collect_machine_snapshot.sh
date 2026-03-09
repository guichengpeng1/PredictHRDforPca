#!/usr/bin/env bash

set -euo pipefail

OUT_FILE="${1:-machine_snapshot_$(date +%Y%m%d_%H%M%S).md}"

PY_BIN=""
if [ -x "/home/ubuntu/miniconda3/envs/starf/bin/python" ]; then
  PY_BIN="/home/ubuntu/miniconda3/envs/starf/bin/python"
elif [ -x "/Users/gui/miniconda3/bin/python" ]; then
  PY_BIN="/Users/gui/miniconda3/bin/python"
elif [ -x "/home/ubuntu/miniconda3/bin/python" ]; then
  PY_BIN="/home/ubuntu/miniconda3/bin/python"
elif command -v python >/dev/null 2>&1; then
  PY_BIN="$(command -v python)"
elif command -v python3 >/dev/null 2>&1; then
  PY_BIN="$(command -v python3)"
fi

OS_NAME="$(uname -s)"

{
  echo "# Machine Snapshot"
  echo
  echo "Date: $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "PWD: $(pwd)"
  echo
  echo "## OS"
  if [ "$OS_NAME" = "Darwin" ]; then
    sw_vers || true
  elif [ -f /etc/os-release ]; then
    cat /etc/os-release
  else
    uname -srm || true
  fi
  echo
  echo '```'
  uname -a || true
  echo '```'
  echo
  echo "## Hardware"
  if [ "$OS_NAME" = "Darwin" ]; then
    system_profiler SPHardwareDataType | sed -n '1,80p' || true
  else
    lscpu || true
    echo
    free -h || true
  fi
  echo
  echo "## GPU"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || true
  else
    echo "nvidia-smi not found"
  fi
  echo
  echo "## Storage"
  df -h / "$(pwd)" 2>/dev/null || df -h / || true
  if [ "$OS_NAME" != "Darwin" ]; then
    echo
    lsblk -o NAME,SIZE,FSTYPE,MOUNTPOINT,MODEL 2>/dev/null || true
  fi
  echo
  echo "## Python paths"
  which -a python python3 pip pip3 2>/dev/null || true
  echo
  echo "## Conda Python / ML stack"
  if [ -n "$PY_BIN" ]; then
    echo "Python used: $PY_BIN"
    "$PY_BIN" - <<'PY'
import json
import sys

info = {
    "python": sys.version.replace("\n", " "),
}

try:
    import torch
    info["torch"] = torch.__version__
    info["cuda_available"] = torch.cuda.is_available()
    info["mps_available"] = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if torch.cuda.is_available():
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
except Exception as e:
    info["torch_error"] = str(e)

for mod in ["timm", "openslide", "pandas", "sklearn", "PIL"]:
    try:
        __import__(mod)
        info[f"module_{mod}"] = "OK"
    except Exception as e:
        info[f"module_{mod}"] = f"ERR: {e}"

print(json.dumps(info, indent=2, ensure_ascii=False))
PY
  else
    echo "No usable python interpreter found in PATH"
  fi
  echo
  echo "## OpenSlide libraries"
  if [ "$OS_NAME" = "Darwin" ]; then
    if command -v rg >/dev/null 2>&1; then
      ls /opt/homebrew/lib 2>/dev/null | rg 'openslide|libopenslide' || true
    else
      ls /opt/homebrew/lib 2>/dev/null | grep -E 'openslide|libopenslide' || true
    fi
  else
    ldconfig -p 2>/dev/null | grep -i openslide || true
  fi
} > "$OUT_FILE"

echo "Wrote snapshot to $OUT_FILE"
