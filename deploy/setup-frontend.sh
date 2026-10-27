#!/usr/bin/env bash
# Usage: bash setup-frontend.sh BACKEND_PRIVATE_IP TOKEN_FILE [DIST_DIRECTORY]
#        bash setup-frontend.sh --frontend-only [DIST_DIRECTORY]
# TOKEN_FILE: one hex/URL-safe bearer token, not the evaluator password.
# --frontend-only: serve the UI before any backend exists (API calls return 502).
#   Rerun later with the backend address and token file; only the marked lines change.
set -euo pipefail
umask 077
FRONTEND_ONLY=false
PLACEHOLDER_TOKEN=
if [[ "${1:-}" == --frontend-only ]]; then
  FRONTEND_ONLY=true
  BACKEND_ADDR=127.0.0.1
  DIST="${2:-$HOME/dist}"
  # Random throwaway token: nothing listens upstream yet, and the rerun replaces it.
  PLACEHOLDER_TOKEN="$(mktemp)"
  TOKEN_FILE="$PLACEHOLDER_TOKEN"
  od -An -N24 -tx1 /dev/urandom | tr -d ' 
' > "$TOKEN_FILE"
else
  BACKEND_ADDR="${1:?backend private IPv4 address or private DNS name (or --frontend-only)}"
  TOKEN_FILE="${2:?path to the backend token file}"
  DIST="${3:-$HOME/dist}"
fi
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
[[ "$BACKEND_ADDR" =~ ^[A-Za-z0-9][A-Za-z0-9.-]*$ ]] || { echo 'Invalid backend address; omit scheme and port.' >&2; exit 2; }
TOKEN="$(tr -d '\r' < "$TOKEN_FILE")"
[[ "$TOKEN" =~ ^[A-Za-z0-9_-]{16,}$ ]] || { echo 'Invalid token file: expected a single hex/URL-safe token.' >&2; exit 2; }
unset TOKEN
chmod 600 -- "$TOKEN_FILE"
[[ -f "$SCRIPT_DIR/nginx-frontend.conf" ]] || { echo 'Keep nginx-frontend.conf beside this script.' >&2; exit 2; }
[[ -f "$DIST/index.html" || -f /var/www/visionguard/index.html ]] || { echo 'Upload frontend/web/dist (including index.html) first.' >&2; exit 2; }

sudo apt-get update
sudo apt-get install -y nginx apache2-utils rsync python3
# Confirm connectivity/token before changing the existing nginx configuration.
if [[ "$FRONTEND_ONLY" != true ]]; then
python3 - "$BACKEND_ADDR" "$TOKEN_FILE" <<'PY'
import pathlib, sys, urllib.request
address, token_file = sys.argv[1:]
request = urllib.request.Request('http://' + address + ':8080/api/cameras',
    headers={'Authorization': 'Bearer ' + pathlib.Path(token_file).read_text().strip()})
with urllib.request.urlopen(request, timeout=10) as response:
    assert response.status == 200
print('Backend private-network API/token check passed.')
PY
fi
if ! sudo test -f /etc/nginx/visionguard.htpasswd; then
  read -r -p 'Evaluator login name: ' EVALUATOR
  [[ "$EVALUATOR" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Invalid login name.' >&2; exit 2; }
  sudo htpasswd -c /etc/nginx/visionguard.htpasswd "$EVALUATOR"
fi
sudo chown root:www-data /etc/nginx/visionguard.htpasswd
sudo chmod 640 /etc/nginx/visionguard.htpasswd
sudo install -d -m 755 /var/www/visionguard
if [[ -f "$DIST/index.html" ]]; then sudo rsync -a --chmod=D755,F644 -- "$DIST/" /var/www/visionguard/; fi

NEW="$(mktemp)"
OLD="$(mktemp)"
HAD_CONFIG=false
DEFAULT_REMOVED=false
INSTALLED=false
SUCCESS=false
cleanup() {
  if [[ "$SUCCESS" != true && "$INSTALLED" == true ]]; then
    if [[ "$HAD_CONFIG" == true ]]; then
      sudo install -m 600 "$OLD" /etc/nginx/conf.d/visionguard.conf
    else
      sudo rm -f /etc/nginx/conf.d/visionguard.conf
    fi
    if [[ "$DEFAULT_REMOVED" == true ]]; then sudo ln -s /etc/nginx/sites-available/default /etc/nginx/sites-enabled/default; fi
  fi
  rm -f "$NEW" "$OLD" ${PLACEHOLDER_TOKEN:+"$PLACEHOLDER_TOKEN"}
}
trap cleanup EXIT
if sudo test -f /etc/nginx/conf.d/visionguard.conf; then
  HAD_CONFIG=true
  sudo cat /etc/nginx/conf.d/visionguard.conf > "$OLD"
fi
# Change only marked lines on rerun, preserving HTTPS directives from Certbot.
python3 - "$SCRIPT_DIR/nginx-frontend.conf" "$OLD" "$NEW" "$BACKEND_ADDR" "$HAD_CONFIG" "$TOKEN_FILE" <<'PY'
from pathlib import Path
import re, sys
source, old, new, address, existing, token_file = sys.argv[1:]
token = Path(token_file).read_text().strip()
if existing == 'true':
    text = Path(old).read_text()
    text, addresses = re.subn(r'^\s*server [^;]+; # VG_BACKEND_ADDRESS$',
        lambda _: '    server ' + address + ':8080; # VG_BACKEND_ADDRESS', text, flags=re.M)
    text, tokens = re.subn(r'^\s*proxy_set_header Authorization [^\n]+# VG_BACKEND_TOKEN$',
        lambda _: '    proxy_set_header Authorization "Bearer ' + token + '"; # VG_BACKEND_TOKEN', text, flags=re.M)
    if addresses != 1 or tokens != 1:
        raise SystemExit('Existing nginx config lacks unique VG markers; back it up and migrate it manually.')
else:
    text = Path(source).read_text().replace('__BACKEND_ADDR__', address).replace('__BACKEND_TOKEN__', token)
if '__BACKEND_' in text:
    raise SystemExit('Unrendered backend placeholders')
Path(new).write_text(text)
PY
sudo install -m 600 "$NEW" /etc/nginx/conf.d/visionguard.conf
INSTALLED=true
if [[ -e /etc/nginx/sites-enabled/default || -L /etc/nginx/sites-enabled/default ]]; then
  [[ -L /etc/nginx/sites-enabled/default && "$(readlink -f /etc/nginx/sites-enabled/default)" == /etc/nginx/sites-available/default ]] || {
    echo 'A custom default site exists; resolve the listener conflict manually.' >&2; exit 2;
  }
  sudo unlink /etc/nginx/sites-enabled/default
  DEFAULT_REMOVED=true
fi
sudo nginx -t
# Back by itself: enabled = starts at every boot; Restart=always = restarted if it dies.
sudo install -d -m 755 /etc/systemd/system/nginx.service.d
printf '[Service]
Restart=always
RestartSec=3
' | sudo tee /etc/systemd/system/nginx.service.d/visionguard-restart.conf >/dev/null
sudo chmod 644 /etc/systemd/system/nginx.service.d/visionguard-restart.conf
sudo systemctl daemon-reload
sudo systemctl enable --now nginx
sudo systemctl reload nginx
SUCCESS=true
if [[ "$FRONTEND_ONLY" == true ]]; then
  echo 'Frontend is live; API calls fail (502) until you rerun with BACKEND_PRIVATE_IP TOKEN_FILE.'
fi
echo 'Nginx configured. Use an SSH tunnel for the first test; configure HTTPS before public login.'
