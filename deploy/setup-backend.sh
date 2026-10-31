#!/usr/bin/env bash
# Ubuntu x86_64 backend. Run as ubuntu from /opt/visionguard; see DEPLOY.md.
# First install: bash deploy/setup-backend.sh cpu data/videos/demo.mp4
# Rerun/migration: bash deploy/setup-backend.sh gpu
set -euo pipefail
umask 077

MODE="${1:-}"
case "$MODE" in cpu|gpu) ;; *) echo 'Usage: bash deploy/setup-backend.sh cpu|gpu [video-path]' >&2; exit 2 ;; esac
[[ "$(id -un)" == ubuntu ]] || { echo 'Run as the Ubuntu AMI user ubuntu, not root.' >&2; exit 2; }
[[ "$(uname -m)" == x86_64 ]] || { echo 'Use x86_64 EC2, not Graviton/ARM.' >&2; exit 2; }
source /etc/os-release
[[ "$ID" == ubuntu && ( "$VERSION_ID" == 22.04 || "$VERSION_ID" == 24.04 ) ]] || { echo 'Use Ubuntu 22.04 or 24.04.' >&2; exit 2; }
cd /opt/visionguard
[[ -f requirements.txt && -f deploy/prepare_backend.py ]] || { echo 'Upload the complete repository first.' >&2; exit 2; }
[[ ! -f .env ]] || { echo 'Move the copied development .env aside first; it can override server configuration.' >&2; exit 2; }
[[ -f deploy/runtime/config.yaml || -n "${2:-}" ]] || { echo 'First install requires a Linux video path as argument 2.' >&2; exit 2; }
if [[ -n "${2:-}" && ! -f "$2" ]]; then echo "Video not found: $2" >&2; exit 2; fi
if [[ "$MODE" == gpu ]]; then
  command -v nvidia-smi >/dev/null || { echo 'Use an NVIDIA-driver GPU AMI; nvidia-smi is missing.' >&2; exit 2; }
  nvidia-smi
fi

INDEX="${VG_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
if [[ "$MODE" == gpu ]]; then INDEX="${VG_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"; fi
if [[ "$MODE" == cpu ]]; then
  [[ "$INDEX" == https://download.pytorch.org/whl/cpu ]] || { echo 'CPU mode requires the CPU wheel index.' >&2; exit 2; }
else
  [[ "$INDEX" =~ ^https://download\.pytorch\.org/whl/cu[0-9]+$ ]] || { echo 'Use an official stable CUDA wheel index.' >&2; exit 2; }
fi
GLIB_PACKAGE=libglib2.0-0
if [[ "$VERSION_ID" == 24.04 ]]; then GLIB_PACKAGE=libglib2.0-0t64; fi
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip libgl1 "$GLIB_PACKAGE" ffmpeg curl openssl
# Stop before touching an existing venv used by camera workers.
if systemctl is-active --quiet visionguard-backend; then sudo systemctl stop visionguard-backend; fi
python3 -m venv venv
venv/bin/python -m pip install --upgrade pip
# Pin the locally verified torch/vision pair. Never use a nightly/test index.
venv/bin/python -m pip install --only-binary=:all: --upgrade --force-reinstall \
  'torch==2.14.1' 'torchvision==0.29.1' --index-url "$INDEX"
CONSTRAINTS="$(mktemp)"
trap 'rm -f "$CONSTRAINTS"' EXIT
venv/bin/python -c 'from importlib.metadata import version; print("torch==" + version("torch")); print("torchvision==" + version("torchvision"))' > "$CONSTRAINTS"
venv/bin/python -m pip install --only-binary=:all: -c "$CONSTRAINTS" -r requirements.txt
venv/bin/python -m pip check
venv/bin/python - "$MODE" <<'PY'
import sys
import cv2
import torch
import torchvision
print('OpenCV:', cv2.__version__, 'PyTorch:', torch.__version__, 'torchvision:', torchvision.__version__)
if sys.argv[1] == 'gpu':
    if not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable: fix the AMI driver / wheel compatibility before starting.')
    a = torch.ones((32, 32), device='cuda')
    assert float((a @ a)[0, 0].item()) == 32.0
    torch.cuda.synchronize()
    print('CUDA tensor operation passed:', torch.cuda.get_device_name(0))
elif torch.version.cuda is not None:
    raise SystemExit('CPU mode unexpectedly has a CUDA torch build; use a fresh venv for a mode change.')
PY
PREPARE=(--mode "$MODE")
if [[ -n "${2:-}" ]]; then PREPARE+=(--video "$2"); fi
venv/bin/python deploy/prepare_backend.py "${PREPARE[@]}"

if ! sudo test -f /etc/visionguard.env; then
  TOKEN="$(openssl rand -hex 32)"
  printf 'VISIONGUARD_DASHBOARD_TOKEN=%s\n' "$TOKEN" | sudo tee /etc/visionguard.env >/dev/null
  unset TOKEN
fi
sudo chown root:root /etc/visionguard.env
sudo chmod 600 /etc/visionguard.env
TOKEN="$(sudo sed -n 's/^VISIONGUARD_DASHBOARD_TOKEN=//p' /etc/visionguard.env)"
[[ "$TOKEN" =~ ^[A-Za-z0-9_-]{16,}$ ]] || { echo 'Invalid/missing token in /etc/visionguard.env; use an unquoted hex or URL-safe token.' >&2; exit 2; }
sudo install -m 644 deploy/visionguard-backend.service /etc/systemd/system/visionguard-backend.service
sudo systemctl daemon-reload
sudo systemctl enable visionguard-backend
sudo systemctl restart visionguard-backend

# /api/status exists; /api/health does not. Keep the token out of process args.
READY=false
for ((attempt=0; attempt<120; attempt++)); do
  if curl --config <(printf 'header = "Authorization: Bearer %s"\n' "$TOKEN") \
      --fail --silent --max-time 3 --output /dev/null http://127.0.0.1:8080/api/status; then READY=true; break; fi
  sleep 3
done
unset TOKEN
if [[ "$READY" != true ]]; then
  echo 'Not HTTP-ready. Inspect: sudo journalctl -u visionguard-backend -n 100 --no-pager' >&2
  exit 1
fi
systemctl --no-pager --full status visionguard-backend
echo 'HTTP-ready. Verify camera status, errors and detections using DEPLOY.md before cutover.'
echo 'Token: /etc/visionguard.env. Transfer privately; never put it in Git or the frontend bundle.'
