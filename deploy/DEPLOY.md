# VisionGuard deployment: frontend EC2 + CPU/GPU backend EC2

This guide uses **two instances in normal operation**:

```text
Browser -- HTTPS (or SSH tunnel for first test) --> Frontend EC2
                                                    nginx + React dist
                                                         |
                                                  private IPv4:8080
                                                         |
                                                  Backend EC2
                                               Python + models + data
```

The backend can be CPU initially and an NVIDIA GPU instance later. CPU and GPU
are alternative backend hosts, not distributed halves of one CV pipeline.
A replacement backend may temporarily be a third instance during migration.
No AWS resources are created by these scripts.

## 1. Prerequisites and scope

- Frontend: Ubuntu 22.04/24.04, with SSH access as `ubuntu`.
- Backend: **Ubuntu 22.04/24.04 x86_64** (Python 3.10/3.12 respectively).
  These scripts do not target Graviton/ARM or Amazon Linux.
- GPU backend: an Ubuntu NVIDIA-driver/Deep Learning AMI with working
  `nvidia-smi`. Setup defaults to the stable CUDA 13.0 wheel channel and checks
  an actual CUDA tensor operation. Install a compatible driver using the AMI's
  supported process; the script does not install/rewrite NVIDIA drivers.
- Both instances in the **same VPC**, normally the same region. For simple
  installation they can use public subnets/public IPs for SSH and outbound
  package downloads, with the backend port restricted as below. A backend in
  a private subnet needs an existing route for package downloads.
- Models and a real test video must be uploaded. Git does not contain ignored
  `.pt`/`.onnx`/OpenVINO weights or `data/`.
- Build the frontend on your development PC, with Node/npm satisfying
  `frontend/web/package.json` (Node >=20), not on the frontend EC2.

The scripts prepare the existing application; they do not resolve its remaining
accuracy/concurrency bugs or prove AWS capacity. Initial deployment starts
**one looping video with person detection + person tracking only**. Other model
parameters are copied, but modules are enabled deliberately after validation.
The original `configs/config.yaml` and local camera selections are not edited.

## 2. EC2 networking

Create dedicated security groups:

| Instance | Inbound | Source |
| --- | --- | --- |
| Frontend | TCP 22 | Your administrator IP only |
| Frontend | TCP 80/443 | Evaluator IPs, or the intended public audience after HTTPS setup |
| Backend | TCP 22 | Your administrator IP only |
| Backend | TCP 8080 | **Frontend security group**, not `0.0.0.0/0` |

Use the backend's **private** IPv4/DNS address in Nginx. Confirm both groups allow
needed egress and that subnet routes/network ACLs permit the traffic. Do not
open backend port 8080 publicly to solve a proxy error.

A camera at `192.168.x.x`/`172.16.x.x` on your local LAN is not automatically
reachable from EC2: RTSP needs a routed private connection/VPN or another
explicitly secured network path. Start with an uploaded video.

## 3. Build and transfer files from your PC

From the repository root, in PowerShell:

```powershell
Set-Location frontend/web
npm ci
npm run build
Set-Location ../..
```

Confirm `frontend/web/dist/index.html` exists before continuing.
Prepare a backend archive. This includes local model files but excludes the
Windows virtualenv, development environment, local data and stored cameras:

```powershell
tar -czf "$env:TEMP/visionguard-deploy.tgz" --exclude=.git --exclude=.venv --exclude=venv --exclude=env --exclude=node_modules --exclude=dist --exclude=runs --exclude=logs --exclude=.env --exclude=deploy/runtime --exclude=configs/cameras.yaml --exclude=data --exclude=*.pem --exclude=*.key --exclude=*.ppk .
scp -i KEY.pem "$env:TEMP/visionguard-deploy.tgz" ubuntu@BACKEND_PUBLIC_IP:~/
scp -i KEY.pem "data/videos/Example of Hi-Definition Video Surveillance of a Factory Floor - by CCTVDOC.COM_.mp4" ubuntu@BACKEND_PUBLIC_IP:~/demo.mp4
scp -i KEY.pem -r frontend/web/dist ubuntu@FRONTEND_PUBLIC_IP:~/
scp -i KEY.pem deploy/setup-frontend.sh deploy/nginx-frontend.conf ubuntu@FRONTEND_PUBLIC_IP:~/
```

Replace the key and host placeholders. The archive can be large because model
weights are included. Do not upload your Windows `.env`, `.venv`, or
`configs/cameras.yaml`: saved Windows paths/webcam sources will not work on EC2.
Face databases and existing deployment state have a separate migration step.
Keep SSH key pairs outside the repository (for example `$HOME\.ssh\`). The
`*.pem`/`*.key`/`*.ppk` excludes are a safety net; before uploading, this
should print nothing:

```powershell
tar -tzf "$env:TEMP/visionguard-deploy.tgz" | Select-String -Pattern '\.(pem|key|ppk)$|(^|/)\.env$'
```

## 4. Install the CPU backend

SSH to the backend, then run these **Ubuntu commands**:

```bash
sudo install -d -o ubuntu -g ubuntu /opt/visionguard
tar --no-same-owner -xzf ~/visionguard-deploy.tgz -C /opt/visionguard
cd /opt/visionguard
mkdir -p data/videos
mv ~/demo.mp4 data/videos/demo.mp4
bash deploy/setup-backend.sh cpu data/videos/demo.mp4
```

Run setup as `ubuntu`, not with `sudo bash`; individual privileged operations
use sudo. Use `bash ...` to avoid SCP/executable-bit differences.

Setup installs system dependencies, creates `venv`, installs CPU torch wheels
and project dependencies, checks packages, validates required assets, creates
`deploy/runtime/config.yaml`, and starts `visionguard-backend.service`.

The torch/torchvision pair is pinned to **2.14.1 / 0.29.1**, matching the verified
local pair. Linux x86_64 wheels for Python 3.10/3.12 were checked in the official
CPU and cu130 indexes. This is wheel availability verification, not a completed
Ubuntu installation test. Main requirements retain their existing pins.

Runtime paths:

| Purpose | Path |
| --- | --- |
| Code, venv, model files | `/opt/visionguard/` |
| Deployment config | `/opt/visionguard/deploy/runtime/config.yaml` |
| Dashboard-saved camera selections | `/opt/visionguard/deploy/runtime/cameras.yaml` |
| Backend bearer token | `/etc/visionguard.env` (root, mode 600) |
| Face/event databases and snapshots | Existing configured `data/` paths |
| Service logs | systemd journal |

First config generation enables only person detection/tracking, uses a looping
file and sets `max_streams: 8` to accommodate five viewers at configuration
level. This does not fix the application's overflow/rapid-switch accounting
bugs. CPU mode selects CPU devices; GPU mode selects CUDA for Ultralytics and
YOLOFace. Existing OpenVINO adapters, including Re-ID, retain CPU devices.

Rerunning setup preserves generated module/camera selections and the token.
It stops the existing service before changing its venv. A failed reinstall may
leave the service stopped: inspect the error, fix it, and rerun. This is not a
zero-downtime package updater. Keep a known-working EBS/instance backup before
upgrading an existing deployed installation.

## 5. Check the backend before setting up Nginx

On the backend:

```bash
cd /opt/visionguard
venv/bin/python deploy/prepare_backend.py --mode cpu --check
sudo systemctl status visionguard-backend --no-pager
sudo journalctl -u visionguard-backend -n 100 --no-pager
```

Authenticated API inspection, keeping the token out of curl process arguments:

```bash
TOKEN="$(sudo sed -n 's/^VISIONGUARD_DASHBOARD_TOKEN=//p' /etc/visionguard.env)"
curl --fail --config <(printf 'header = "Authorization: Bearer %s"\n' "$TOKEN") http://127.0.0.1:8080/api/cameras
curl --fail --config <(printf 'header = "Authorization: Bearer %s"\n' "$TOKEN") http://127.0.0.1:8080/api/status
unset TOKEN
```

`/api/status` and `/api/cameras` exist. **There is no `/api/health` endpoint.**
An HTTP 200 means the API answered; also check camera status is `live`, frame
IDs advance, `frame.errors` is empty and `stats.frames_failed` is not increasing.
Then visually confirm detections on the actual clip. Allow time for model load.

## 6. Privately transfer the token and install the frontend

From your PC, use an SSH-to-SSH pipeline; do not paste the token into a public
command, browser URL, frontend source, Vite environment variable or Git:

```powershell
ssh -i KEY.pem ubuntu@BACKEND_PUBLIC_IP "sudo sed -n 's/^VISIONGUARD_DASHBOARD_TOKEN=//p' /etc/visionguard.env" | ssh -i KEY.pem ubuntu@FRONTEND_PUBLIC_IP 'umask 077; cat > ~/visionguard-backend.token'
```

On the **frontend EC2**, replace `10.0.1.25` with the backend private address:

```bash
bash ~/setup-frontend.sh 10.0.1.25 ~/visionguard-backend.token ~/dist
```

The script checks upstream connectivity/token, installs Nginx, prompts for an
evaluator username/password, copies the built UI and validates Nginx before
reload. No backend token is printed. Its rendered config is root-readable only.
Do not share `nginx -T` output: it contains the upstream bearer token.

All routes require evaluator Basic authentication. Nginx replaces this browser
Authorization header with the **separate backend bearer token** upstream.
Preserving `$http_host` keeps browser Origin checks consistent on POST requests.
API paths, query strings and MJPEG are proxied without changing the React app.
MJPEG buffering and request buffering for large video uploads are disabled.

First test through an SSH tunnel from your PC:

```powershell
ssh -i KEY.pem -N -L 18080:127.0.0.1:80 ubuntu@FRONTEND_PUBLIC_IP
```

Open `http://127.0.0.1:18080/portal/dashboard` and enter the evaluator login.
The SSH tunnel protects transport for this test. **Do not send passwords,
faces or camera data over public plain HTTP.** For normal browser access, attach
a domain and configure HTTPS on the frontend (for example Certbot's Nginx
integration). Make the certificate-validation method work with your security
group: HTTP challenges need reachable port 80; restricted sites can use DNS
validation. After enabling HTTPS, confirm HTTP redirects to HTTPS.

Nginx config: `/etc/nginx/conf.d/visionguard.conf`.
Evaluator passwords: `/etc/nginx/visionguard.htpasswd`.
To add/change a login, use `sudo htpasswd /etc/nginx/visionguard.htpasswd NAME`.
The backend token is not an evaluator password and should not be distributed.

### Frontend first, before any backend exists

`deploy/visionguard-frontend.tgz` (gitignored) bundles `dist/`,
`setup-frontend.sh` and `nginx-frontend.conf`. Rebuild it after frontend changes
(see section 3), then from your PC:

```powershell
scp -i KEY.pem deploy/visionguard-frontend.tgz ubuntu@FRONTEND_PUBLIC_IP:~/
```

On the frontend EC2 (the browser "EC2 Instance Connect" terminal works):

```bash
tar -xzf ~/visionguard-frontend.tgz -C ~
bash ~/visionguard-frontend/setup-frontend.sh --frontend-only ~/visionguard-frontend/dist
```

The UI loads; API calls and video return 502 until a backend exists. Nginx is
enabled at boot and restarted if it dies (`nginx.service.d/visionguard-restart.conf`).
Once the backend is up, transfer its token as above and rerun with
`BACKEND_PRIVATE_IP TOKEN_FILE`: only the two marked lines change.
An auto-assigned public IP changes on stop/start; attach an Elastic IP to keep the URL.

## 7. Add cameras/modules and perform the acceptance check

Start with one camera, then increase to your intended five-camera/five-viewer
workload. Five visible people, five cameras and five viewers are different tests.

- Use the Cameras page to upload/select real videos or enter reachable RTSP
  sources. Linux paths must refer to files on the **backend**, not your PC.
- Per-camera selections are saved next to the runtime config. Required modules
  must be selected together; e.g. fall needs pose detection.
- Parameters/models are copied from the existing config. The Modules page can
  enable other modules; validate their required assets before enabling them.
- Face recognition needs genuine enrolled samples; migration of existing face
  data is described below. Do not claim recognition of a back-facing person
  from a face that was never observed.
- PPE/Re-ID/liveness/fall/abandoned-object accuracy still needs scenario-specific
  validation. Fire/smoke require custom model setup.

Acceptance steps on the deployed hosts:

1. Refresh a nested URL such as `/portal/cameras`: the SPA should load.
2. Verify unauthenticated page/API requests receive 401; evaluator login works.
3. Switch cameras; observe correct video, advancing frame IDs and overlays.
4. Exercise a legitimate setting change/upload: no 403 Origin or 413 body error.
5. Open five distinct browser sessions, verify video **and** responsive status.
6. Observe five people, occlusion and re-entry; review identity/track continuity.
7. Run the intended positive and negative safety clips and inspect events/alerts.
8. Restart the service and confirm stored cameras/enrollments remain intact.
9. Inspect journal and API errors. Record results; HTTP availability alone is
   not proof that an AI module is detecting correctly.

Existing regression tests can be run in a separate development/staging venv
with `requirements-dev.txt`; installing test packages is not part of runtime
setup. Known existing failures/limits are listed in section 11.

## 8. Switch the backend from CPU to GPU

1. Launch an approved-quota **x86_64 NVIDIA GPU** instance with an Ubuntu driver
   AMI. Keep frontend/GPU backend in the same VPC, or establish private routing.
   Apply the backend security group. A g6 host is an example, not a tested
   capacity promise. Choose the instance from your measured camera workload.
2. Upload the same code/models/video as in section 3. On the new backend:

```bash
cd /opt/visionguard
nvidia-smi
bash deploy/setup-backend.sh gpu data/videos/demo.mp4
venv/bin/python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

If the compatible wheel channel differs, set it explicitly, for example
`VG_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 bash deploy/setup-backend.sh gpu data/videos/demo.mp4`.
The exact pinned pair must exist in that index and support the GPU/driver.
Do not install a full CUDA toolkit just because a wheel channel is named CUDA.
The setup tensor test must pass; then confirm model logs actually select CUDA.

3. **Migrate state consistently.** On the CPU backend stop the service before
   archiving SQLite files, otherwise WAL writes can be missed. On the new GPU
   backend stop its service too before restoring. These commands include local
   videos, face/event DBs, snapshots and saved camera selections, but not tokens:

```bash
# CPU backend
sudo systemctl stop visionguard-backend
cd /opt/visionguard
tar -czf ~/visionguard-state.tgz data deploy/runtime
```

Transfer this private archive via your PC or another secure SSH path. It may
contain biometric templates, snapshots and camera credentials. From PowerShell:

```powershell
scp -i KEY.pem ubuntu@CPU_PUBLIC_IP:~/visionguard-state.tgz "$env:TEMP/visionguard-state.tgz"
scp -i KEY.pem "$env:TEMP/visionguard-state.tgz" ubuntu@GPU_PUBLIC_IP:~/
```

Then on the GPU backend:

```bash
sudo systemctl stop visionguard-backend
cd /opt/visionguard
tar --no-same-owner -xzf ~/visionguard-state.tgz
# Preserve migrated selections, update model devices and validate Linux assets.
venv/bin/python deploy/prepare_backend.py --mode gpu
sudo systemctl start visionguard-backend
```

Use the same `/opt/visionguard` layout and upload all assets referenced by saved
cameras. SQLite and other state remain local to the selected backend; CPU and
GPU instances do not share a live database automatically.

4. Transfer the **GPU backend's** token to the frontend as in section 6. Rerun
   `setup-frontend.sh NEW_GPU_PRIVATE_IP TOKEN_FILE`. Only the two marked Nginx
   backend lines change; existing HTTPS directives are preserved. Nginx syntax
   is checked before reload. No React rebuild is needed.
5. Repeat the acceptance steps, verify real CUDA inference and watch errors.
   Keep the CPU instance/state for rollback until satisfied; stop unused compute.
   To roll back, start CPU and restore its private address/token on the frontend.

## 9. Start/stop for a mentor demo

Start the backend EC2 in the AWS console, wait for systemd/model startup, then
open the frontend. After the demo, stop the backend **through EC2**. Enabling
systemd starts the app when that instance boots; it does not make a stopped EC2
start when someone opens a page. Browser-triggered AWS wake-up automation is a
separate feature and is not included here.

Use EBS for persistent code/data, not disposable instance-store storage.
Recheck public SSH/frontend addresses after stop/start; auto-assigned public
IPv4 addresses can change. A replacement backend has a new private address,
which must be updated in the frontend. Stopping compute does not remove EBS,
allocated-address or other retained-resource charges.

## 9a. Wake on demand (GPU sleeps when idle)

Files: `deploy/wake/`. The frontend runs `vgwake` (127.0.0.1:9100, nginx routes
`/wake/status` and `POST /wake/start`). Opening any `/portal/*` page starts the
GPU backend and shows a startup screen with the real stages (EC2 running, SSH
reachable, tunnel up, API answering, YOLO frames advancing); the dashboard
opens only when all pass. Public pages never need the backend.

- Tunnel: `vg-tunnel-run` reads the backend's current IP from
  `/var/lib/vgwake/backend_host` (written by vgwake after every start) and pins
  the host key under the alias `vg-backend`. Backend user `vgtunnel` may only
  forward to 127.0.0.1:8080 (`/etc/ssh/sshd_config.d/90-vgtunnel.conf`).
- Idle stop: after `idle_minutes` (15) without same-origin browser requests
  (`/var/log/nginx/vg_activity.log`) and with no open upstream connection
  (streams, requests in flight), confirmed twice a minute apart. Enabled only
  when AWS confirms the shutdown behaviour is `stop` and a dry-run stop succeeds.
- Abuse limits: same-origin POST only, 4 starts/min per IP, 60 s cooldown,
  `max_starts_per_day`, `daily_gpu_hours`, and `budget_gpu_hours` for the period.
- Backend fail-safe: `vg-idle-failsafe.timer` powers off after 45 minutes with
  no API connection. It does nothing until `/etc/vg-failsafe/armed` exists;
  create that only after confirming the instance's shutdown behaviour is `stop`.

IAM (done by the account owner in the console): create a role for **EC2** with
the policy in `deploy/wake/iam-policy.json` (start/stop only the backend
instance; describe calls), then EC2 -> frontend instance -> Actions -> Security
-> Modify IAM role -> attach it. `sudo systemctl restart vgwake`, then
`curl -s -H "Host: www.vision-guard.tech" 127.0.0.1:9100/wake/status` should
show `"idle_stop": {"enabled": true}`.

## 10. Troubleshooting

| Symptom | Check |
| --- | --- |
| Nginx 502/504 | Backend running, private address, SG reference, routes, journal |
| 401 after evaluator login | Backend token mismatch; transfer the selected backend token again |
| POST 403 | Preserve browser Host/Origin and Sec-Fetch-Site; do not remove backend origin checks |
| Static UI works, API fails | Nginx `/api/` proxy; do not rebuild React with a private backend URL |
| Video buffers | `/stream.mjpg` buffering off; inspect camera source and current module latency |
| 5th viewer empty/503 | Check deployed max_streams; known overflow/accounting bug still needs an application fix |
| Service exits after a clip | Enable loop or use a live source; EOF can be a clean application exit |
| CUDA false/kernel/driver error | Compatible NVIDIA driver, architecture and stable wheel channel; tensor test |
| Missing model / hash mismatch | Upload the exact configured weights; do not disable hash checks |
| Windows path / webcam error | Linux video path and runtime camera file; do not copy local cameras.yaml blindly |
| Nginx setup fails | Previous config is restored on validation failure; resolve the reported error |
| Setup rerun after HTTPS | Keep VG_BACKEND_ADDRESS and VG_BACKEND_TOKEN markers; script preserves TLS lines |

Useful commands:

```bash
sudo nginx -t
sudo systemctl status nginx --no-pager
sudo journalctl -u visionguard-backend -f
sudo systemctl restart visionguard-backend
```

## 11. Validation limits

Local checks executed on 27 October 2026:

- Bash syntax passed for both setup scripts, including embedded Python parsing.
- Deployment/configuration/API/camera/runtime/hardening regressions: **311 passed,
  1 skipped** (the Windows-inapplicable POSIX-permissions test).
- Real factory CCTV through the generated profiles: **40 CPU + 40 CUDA frames**,
  zero failed frames/module errors; both produced person detections/tracks.
- Official Nginx **1.30.6 Windows build**: `nginx -t` passed with temporary local
  paths; actual React SPA route, Basic auth, bearer forwarding, API/snapshot,
  same-origin POST, foreign-origin rejection and five held MJPEG streams passed.
  The migration renderer retained TLS directives; a real TLS certificate was
  not issued/tested.
- Exact pinned Linux wheel availability was checked for OpenCV, OpenVINO,
  Ultralytics and the CPU/cu130 torch/torchvision pair.

Detailed local outputs are in ignored `runs/deployment-validation-20261027/`
(`proxy-results.json` and `real-pipeline-results.json`). The proxy test used a
real CCTV image to validate delivery, not an AI accuracy benchmark. The CPU
profile was tested using the existing local torch installation on its CPU
device; a fresh Linux CPU-wheel installation remains an EC2 acceptance step.

Preparing these files is not proof of an AWS deployment. Ubuntu apt installation,
systemd boot/reboot, EC2 private networking, TLS issuance and AWS GPU inference
must be verified on the actual hosts. Local checks and results are reported in
the accompanying change review; do not describe unexecuted EC2 steps as passed.

Earlier application audit: 1,336 regression tests passed, one configuration-
expectation test failed (it expects identity modules disabled in the source
config), one platform test skipped. Real-model integration: nine passed, three
skipped because their expected default export was absent. These are historical
results, not automatic validation of every new server install.

Outstanding application issues: stream overflow/rapid-switch slot accounting,
synchronous face requests delaying the API event loop, daily gate totals over
1,000 events, four-person crop limit in the configured long-gun model, and
camera-specific face/PPE/Re-ID accuracy. Deployment scripts do not fix these.

## Official references

- [AWS security-group rules and referencing](https://docs.aws.amazon.com/vpc/latest/userguide/security-group-rules.html)
- [AWS EC2 stop/start behaviour](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/Stop_Start.html)
- [Nginx proxy headers and buffering](https://nginx.org/en/docs/http/ngx_http_proxy_module.html)
- [Nginx Basic authentication](https://docs.nginx.com/nginx/admin-guide/security-controls/configuring-http-basic-authentication/)
- [PyTorch installation selector](https://pytorch.org/get-started/locally/)
- [Official CPU torch index](https://download.pytorch.org/whl/cpu/torch/)
- [Official CUDA 13.0 torch index](https://download.pytorch.org/whl/cu130/torch/)
