#!/usr/bin/env python3
"""VisionGuard wake service (runs on the always-on frontend instance).

Starts the GPU backend when someone opens a backend feature, reports each real
startup stage to the startup screen, and stops the backend again after
inactivity. Listens on 127.0.0.1 only; nginx exposes GET /wake/status and
POST /wake/start.

Stages are measured, never simulated: EC2 state, SSH reachable, tunnel up,
API answering, model producing frames. Estimates are the median of previous
real startups (or a labelled default before the first one).
"""
from __future__ import annotations

import json
import logging
import os
import socket
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

CONFIG_PATH = Path(os.environ.get("VGWAKE_CONFIG", "/etc/vgwake/config.json"))
STATE_DIR = Path(os.environ.get("VGWAKE_STATE", "/var/lib/vgwake"))

STAGES = [
    ("instance", "Powering on the GPU server"),
    ("network", "Waiting for the server to accept connections"),
    ("tunnel", "Opening the encrypted tunnel"),
    ("api", "Starting the VisionGuard API"),
    ("model", "Running the YOLO model on the GPU"),
]
STAGE_IDS = [sid for sid, _ in STAGES]
# Used only until a real startup has been measured; reported as such.
DEFAULT_SECONDS = {"instance": 40, "network": 50, "tunnel": 10, "api": 30, "model": 15}
START_TIMEOUT = 15 * 60
# AWS often has no spare GPU for a few minutes; keep asking instead of giving up.
CAPACITY_RETRY_SECONDS = 60
CAPACITY_RETRY_ATTEMPTS = 10

log = logging.getLogger("vgwake")


def now() -> float:
    return time.time()


def today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def load_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def save_json(path: Path, data) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, path)


class Aws:
    """The EC2 calls this service needs. Credentials come from the instance role."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.error = None
        try:
            import boto3  # apt: python3-boto3
            from botocore.config import Config

            self.ec2 = boto3.client(
                "ec2",
                region_name=cfg["region"],
                config=Config(connect_timeout=3, read_timeout=6, retries={"max_attempts": 2}),
            )
        except Exception as exc:
            self.ec2 = None
            self.error = f"AWS client unavailable: {exc}"

    def describe(self):
        reply = self.ec2.describe_instances(InstanceIds=[self.cfg["instance_id"]])
        inst = reply["Reservations"][0]["Instances"][0]
        return inst["State"]["Name"], inst.get("PublicIpAddress")

    def start(self):
        self.ec2.start_instances(InstanceIds=[self.cfg["instance_id"]])

    def stop(self):
        self.ec2.stop_instances(InstanceIds=[self.cfg["instance_id"]])

    def self_test(self) -> tuple[bool, str]:
        """Idle stop is enabled only when this passes."""
        from botocore.exceptions import ClientError

        attr = self.ec2.describe_instance_attribute(
            InstanceId=self.cfg["instance_id"], Attribute="instanceInitiatedShutdownBehavior"
        )["InstanceInitiatedShutdownBehavior"]["Value"]
        if attr != "stop":
            return False, f"shutdown behaviour is '{attr}', not 'stop'"
        try:
            self.ec2.stop_instances(InstanceIds=[self.cfg["instance_id"]], DryRun=True)
        except ClientError as exc:
            if exc.response["Error"]["Code"] != "DryRunOperation":
                return False, f"stop permission missing ({exc.response['Error']['Code']})"
        return True, "ok"


class Waker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.lock = threading.Lock()
        self.aws = Aws(cfg)
        self.token = Path(cfg["token_file"]).read_text().strip()
        self.history = load_json(STATE_DIR / "history.json", [])
        self.usage = load_json(STATE_DIR / "usage.json", {})
        self.ec2_state = None  # None = unknown (no AWS access)
        self.ip = self.read_host_file()
        self.aws_error = self.aws.error
        self.last_describe = 0.0
        self.stages = {sid: "pending" for sid in STAGE_IDS}
        self.cycle = None  # current startup: {"requested": t, "done": {stage: t}, "cold": bool}
        self.ready_since = None
        # When a wake-up we started finished: the visitor gets idle_minutes from then.
        # Readiness noticed any other way (service restart, a brief failed check)
        # must not reset the idle clock.
        self.woken_at = 0.0
        self.error = None
        self.message = "Checking the GPU backend"
        self.last_start_call = 0.0
        self.pending_start = False
        self.capacity_retry = None  # {"attempt": n, "next_at": t} after InsufficientInstanceCapacity
        self.watched_until = now() + 60
        self.prev_frames = None
        self.last_conn_seen = 0.0
        self.idle_candidate_since = None
        self.idle_stop_ok = False
        self.idle_stop_reason = "not checked yet"
        self.last_self_test = 0.0
        self.last_usage_tick = None
        self.last_snapshot = None  # served by /wake/status without waiting for a tick
        self.snapshot_at = 0.0
        self.wake_loop = threading.Event()  # cuts the idle 30 s sleep short when someone looks

    # ---------- probes: each one is a real check ----------
    @staticmethod
    def tcp_open(host, port, timeout=3.0) -> bool:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            return False

    def tunnel_up(self) -> bool:
        out = subprocess.run(
            ["ss", "-Htn", "state", "established", "dst", f"{self.ip}:22"],
            capture_output=True, text=True, timeout=5,
        ).stdout
        return bool(out.strip()) and self.tcp_open("127.0.0.1", 8080, 2)

    def api_status(self):
        req = urllib.request.Request(
            self.cfg["api"] + "/api/status",
            headers={"Authorization": "Bearer " + self.token, "Host": self.cfg["allowed_hosts"][0],
                     "Connection": "close"},
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return json.loads(resp.read())
        except (urllib.error.URLError, OSError, ValueError):
            return None

    @staticmethod
    def open_backend_connections() -> int:
        out = subprocess.run(
            ["ss", "-Htn", "state", "established", "dport", "=", ":8080"],
            capture_output=True, text=True, timeout=5,
        ).stdout
        return sum(1 for line in out.splitlines() if "127.0.0.1:8080" in line)

    def last_browser_activity(self) -> float:
        try:
            st = os.stat(self.cfg["activity_log"])
            return st.st_mtime if st.st_size > 0 else 0.0
        except OSError:
            return 0.0

    # ---------- AWS state and the tunnel's target address ----------
    def read_host_file(self):
        try:
            return Path(self.cfg["host_file"]).read_text().strip() or None
        except OSError:
            return None

    def write_host_file(self, ip: str):
        path = Path(self.cfg["host_file"])
        tmp = path.with_suffix(".tmp")
        tmp.write_text(ip + ("\n" if ip else ""))
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)

    def refresh_aws(self, force=False):
        if self.aws.ec2 is None:
            return
        busy = self.cycle or self.pending_start or self.ec2_state in ("pending", "stopping")
        if not force and now() - self.last_describe < (4 if busy else 30):
            return
        self.last_describe = now()
        try:
            state, ip = self.aws.describe()
            self.aws_error = None
        except Exception as exc:
            self.aws_error = f"Cannot read the GPU server state from AWS ({type(exc).__name__})"
            log.warning("describe failed: %s", exc)
            return
        self.ec2_state = state
        new_ip = ip if state == "running" else None
        if new_ip != self.ip:
            log.info("backend address %s -> %s", self.ip, new_ip)
            self.ip = new_ip
            self.write_host_file(new_ip or "")
            if new_ip:
                # Reconnect now instead of waiting for keep-alives to notice the old address died.
                subprocess.run(["sudo", "-n", "/usr/bin/systemctl", "restart", "vg-backend-tunnel.service"],
                               capture_output=True, timeout=20)

    def account_usage(self):
        t = now()
        if self.ec2_state in ("pending", "running") and self.last_usage_tick is not None:
            day = self.usage.setdefault(today(), {"gpu_seconds": 0, "starts": 0})
            day["gpu_seconds"] += min(t - self.last_usage_tick, 120)
        self.last_usage_tick = t
        for old in sorted(self.usage)[:-45]:
            del self.usage[old]
        save_json(STATE_DIR / "usage.json", self.usage)

    def period_gpu_hours(self) -> float:
        start, end = self.cfg["budget_period_start"], self.cfg["budget_period_end"]
        return sum(d["gpu_seconds"] for day, d in self.usage.items() if start <= day <= end) / 3600

    # ---------- main loop ----------
    def tick(self):
        with self.lock:
            self.refresh_aws()
            self.account_usage()
            if self.pending_start and self.ec2_state == "stopped":
                self._call_start()
            elif (self.capacity_retry and self.cycle and self.ec2_state == "stopped"
                  and now() >= self.capacity_retry["next_at"]):
                self._call_start()
            self.evaluate()
            self.idle_check()
            self.last_snapshot = self.snapshot()
            self.snapshot_at = now()
            if self.aws.ec2 is not None and now() - self.last_self_test > 3600:
                self.last_self_test = now()
                try:
                    self.idle_stop_ok, self.idle_stop_reason = self.aws.self_test()
                except Exception as exc:
                    self.idle_stop_ok = False
                    self.idle_stop_reason = f"self-test could not run ({type(exc).__name__})"
                    self.last_self_test = now() - 3300  # retry in 5 minutes
                log.info("idle-stop self-test: %s (%s)", self.idle_stop_ok, self.idle_stop_reason)

    def evaluate(self):
        st = self.ec2_state
        done = dict.fromkeys(STAGE_IDS, False)
        done["instance"] = st == "running" or (st is None and self.ip is not None)
        # Probe the backend only while someone waits or it is believed up.
        probing = done["instance"] and (
            self.cycle is not None or now() < self.watched_until or self.ready_since is not None
        )
        done["network"] = probing and bool(self.ip) and self.tcp_open(self.ip, self.cfg.get("ssh_port", 22))
        done["tunnel"] = done["network"] and self.tunnel_up()
        status = self.api_status() if done["tunnel"] else None
        done["api"] = status is not None
        if status:
            frames = (status.get("stats") or {}).get("frames_processed")
            modules = [m.get("name") for m in (status.get("info") or {}).get("modules", [])]
            healthy = bool(status.get("running")) and "person_detection" in modules and isinstance(frames, int)
            # Ready only when the frame count changes between two real samples: the model is
            # running (a camera change restarts the count, so "changed", not "grew").
            done["model"] = healthy and self.prev_frames is not None and frames != self.prev_frames
            if not healthy:
                self.prev_frames = None
            elif self.prev_frames is None or frames != self.prev_frames:
                self.prev_frames = frames
            elif self.ready_since is not None:
                done["model"] = False  # frames stopped moving
        else:
            self.prev_frames = None
        if not probing and self.ready_since is not None:
            done = dict.fromkeys(STAGE_IDS, True)  # nobody watching: keep last verified state

        first_missing = next((sid for sid in STAGE_IDS if not done[sid]), None)
        for i, sid in enumerate(STAGE_IDS):
            if all(done[s] for s in STAGE_IDS[: i + 1]):
                self.stages[sid] = "done"
            elif sid == first_missing:
                self.stages[sid] = "active"
            else:
                self.stages[sid] = "pending"
        if self.cycle:
            for sid in STAGE_IDS:
                if self.stages[sid] == "done":
                    self.cycle["done"].setdefault(sid, now())

        if first_missing is None:
            if self.ready_since is None:
                self.ready_since = now()
                log.info("backend ready")
            if self.cycle:
                self.woken_at = now()
            if self.cycle:
                self.record_history()
                self.cycle = None
            self.error = None
            self.message = "GPU backend is ready"
            return
        if self.ready_since is not None:
            log.info("backend not ready any more (%s)", first_missing)
        self.ready_since = None
        self.message = dict(STAGES)[first_missing]
        if self.capacity_retry and self.cycle:
            self.message = (f"AWS has no free L4 GPU right now; retrying automatically "
                            f"(attempt {self.capacity_retry['attempt'] + 1} of {CAPACITY_RETRY_ATTEMPTS})")
        elif st == "stopped" and not self.cycle and not self.pending_start:
            self.message = "GPU backend is asleep"
        elif st == "stopping":
            self.message = "GPU backend is shutting down; it starts again right after"
        if self.cycle and now() - self.cycle["requested"] > START_TIMEOUT:
            self.error = "Startup took longer than 15 minutes. Try again; if it repeats, check the backend logs."
            self.stages[first_missing] = "failed"
            self.cycle = None

    def record_history(self):
        # Measured from the accepted start, so waiting for AWS capacity does not inflate estimates.
        prev, entry = self.cycle.get("sent", self.cycle["requested"]), {}
        for sid in STAGE_IDS:
            t = self.cycle["done"].get(sid, prev)
            entry[sid] = round(max(0.0, t - prev), 1)
            prev = t
        if self.cycle.get("cold") and sum(entry.values()) > 5:
            self.history = (self.history + [entry])[-20:]
            save_json(STATE_DIR / "history.json", self.history)
            log.info("startup measured: %s", entry)

    def typical(self) -> tuple[dict, str]:
        runs = [h for h in self.history if all(k in h for k in STAGE_IDS)]
        if not runs:
            return dict(DEFAULT_SECONDS), "default estimate: no startup measured yet"
        return ({sid: statistics.median(h[sid] for h in runs) for sid in STAGE_IDS},
                f"median of {len(runs)} measured startup{'s' if len(runs) != 1 else ''}")

    def idle_check(self):
        if self.ready_since is None or self.cycle or self.ec2_state != "running":
            self.idle_candidate_since = None
            return
        if self.open_backend_connections() > 0:  # a stream or a request in flight
            self.last_conn_seen = now()
        last = max(self.last_browser_activity(), self.last_conn_seen, self.woken_at, self.last_start_call)
        idle_for = now() - last
        if idle_for < self.cfg["idle_minutes"] * 60:
            self.idle_candidate_since = None
            return
        if not (self.cfg.get("idle_stop_enabled") and self.idle_stop_ok):
            return
        if self.idle_candidate_since is None:  # confirm on a second check a minute later
            self.idle_candidate_since = now()
            return
        if now() - self.idle_candidate_since < 60:
            return
        log.info("stopping backend: no browser activity for %.0f s and no open connections", idle_for)
        try:
            self.aws.stop()
            self.ec2_state = "stopping"
            self.ip = None
            self.write_host_file("")
            self.ready_since = None
            self.message = "GPU backend stopped after inactivity"
        except Exception as exc:
            log.warning("stop failed: %s", exc)
        self.idle_candidate_since = None

    # ---------- start request ----------
    def request_start(self) -> tuple[int, dict]:
        with self.lock:
            self.watched_until = now() + 120
            self.refresh_aws(force=True)
            if self.ready_since is not None:
                return 200, self.snapshot()
            if self.aws.ec2 is None or self.aws_error:
                return 503, {**self.snapshot(), "error": self.aws_error or "AWS access is not configured"}
            day = self.usage.setdefault(today(), {"gpu_seconds": 0, "starts": 0})
            if self.ec2_state in ("stopped", "stopping"):
                if day["starts"] >= self.cfg["max_starts_per_day"]:
                    return 429, {**self.snapshot(), "error": "Daily start limit reached (protects the budget). Try again tomorrow."}
                if day["gpu_seconds"] >= self.cfg["daily_gpu_hours"] * 3600:
                    return 429, {**self.snapshot(), "error": "Daily GPU-hours limit reached (protects the budget). Try again tomorrow."}
                if self.period_gpu_hours() >= self.cfg["budget_gpu_hours"]:
                    return 429, {**self.snapshot(), "error": "The GPU budget for this demo period is used up. The public pages stay available."}
                if self.cycle is None:
                    self.cycle = {"requested": now(), "done": {}, "cold": True}
                    self.error = None
                    self.capacity_retry = None
                if self.ec2_state == "stopping":
                    self.pending_start = True
                elif now() - self.last_start_call >= self.cfg["start_cooldown_seconds"]:
                    self._call_start()
            elif self.ec2_state in ("pending", "running") and self.cycle is None:
                self.cycle = {"requested": now(), "done": {}, "cold": self.ec2_state == "pending"}
                self.error = None
            return 202, self.snapshot()

    def _call_start(self):
        from botocore.exceptions import ClientError

        self.pending_start = False
        self.last_start_call = now()
        try:
            self.aws.start()
            self.usage.setdefault(today(), {"gpu_seconds": 0, "starts": 0})["starts"] += 1
            self.ec2_state = "pending"
            self.capacity_retry = None
            if self.cycle:
                self.cycle["sent"] = now()
            log.info("StartInstances sent")
        except ClientError as exc:
            code = exc.response["Error"]["Code"]
            attempt = (self.capacity_retry or {"attempt": 0})["attempt"] + 1
            if code == "InsufficientInstanceCapacity" and attempt < CAPACITY_RETRY_ATTEMPTS:
                self.capacity_retry = {"attempt": attempt, "next_at": now() + CAPACITY_RETRY_SECONDS}
                log.warning("start failed: %s (attempt %d, retrying in %d s)", code, attempt, CAPACITY_RETRY_SECONDS)
                return
            self.capacity_retry = None
            self.error = ("AWS still has no free L4 GPU in this zone after several tries. Please try again later."
                          if code == "InsufficientInstanceCapacity"
                          else f"AWS refused to start the GPU server ({code}).")
            self.cycle = None
            log.warning("start failed: %s (giving up)", code)

    # ---------- status ----------
    def snapshot(self) -> dict:
        typical, basis = self.typical()
        total = sum(typical.values()) or 1
        ready = self.ready_since is not None
        done_w = sum(typical[s] for s in STAGE_IDS if self.stages[s] == "done")
        stages, eta = [], None
        stage_start = self.cycle["requested"] if self.cycle else None
        elapsed_active = 0.0
        for sid, label in STAGES:
            status = self.stages[sid]
            if self.error and status == "active":
                status = "failed"
            entry = {"id": sid, "label": label, "status": status, "typical_seconds": round(typical[sid])}
            if self.cycle:
                if sid in self.cycle["done"]:
                    entry["seconds"] = round(self.cycle["done"][sid] - stage_start, 1)
                    stage_start = self.cycle["done"][sid]
                elif entry["status"] == "active":
                    elapsed_active = now() - stage_start
                    entry["seconds"] = round(elapsed_active, 1)
            stages.append(entry)
        if self.cycle:
            remaining = sum(typical[s] for s in STAGE_IDS if self.stages[s] != "done")
            eta = max(0, round(remaining - elapsed_active))
        if ready:
            state = "ready"
        elif self.error:
            state = "error"
        elif self.cycle or self.pending_start or self.ec2_state == "pending":  # incl. capacity retries
            state = "starting"
        elif self.ec2_state == "stopping":
            state = "stopping"
        elif self.ec2_state == "stopped":
            state = "asleep"
        else:
            state = "unknown"
        day = self.usage.get(today(), {"gpu_seconds": 0, "starts": 0})
        return {
            "state": state,
            "ready": ready,
            "message": self.message,
            "error": self.error or (self.aws_error if state in ("asleep", "unknown") else None),
            "progress": 1.0 if ready else round(done_w / total, 3),
            "eta_seconds": eta,
            "eta_basis": basis,
            "elapsed_seconds": round(now() - self.cycle["requested"]) if self.cycle else None,
            "stages": stages,
            "idle_minutes": self.cfg["idle_minutes"],
            "budget": {
                "starts_today": day["starts"],
                "max_starts_per_day": self.cfg["max_starts_per_day"],
                "gpu_hours_today": round(day["gpu_seconds"] / 3600, 2),
                "daily_gpu_hours": self.cfg["daily_gpu_hours"],
                "period_gpu_hours": round(self.period_gpu_hours(), 2),
                "budget_gpu_hours": self.cfg["budget_gpu_hours"],
                "budget_period": [self.cfg["budget_period_start"], self.cfg["budget_period_end"]],
            },
            "idle_stop": {
                "enabled": bool(self.cfg.get("idle_stop_enabled") and self.idle_stop_ok),
                "reason": self.idle_stop_reason if self.cfg.get("idle_stop_enabled") else "disabled in config",
            },
        }

    def status(self) -> dict:
        # Never waits on a tick's network checks: answers from the last one.
        self.watched_until = max(self.watched_until, now() + 30)
        if now() - self.snapshot_at > 4:
            self.wake_loop.set()  # stale: check again now rather than after the idle sleep
        if self.last_snapshot is None:
            with self.lock:
                self.last_snapshot = self.snapshot()
        return self.last_snapshot


def make_handler(waker: Waker):
    allowed = set(waker.cfg["allowed_hosts"])

    class Handler(BaseHTTPRequestHandler):
        server_version = "vgwake"
        sys_version = ""

        def log_message(self, fmt, *args):
            return

        def reply(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass  # the browser gave up waiting; its next poll gets the answer

        def host_ok(self) -> bool:
            return self.headers.get("Host", "").split(":")[0] in allowed

        def do_GET(self):
            if self.path != "/wake/status" or not self.host_ok():
                return self.reply(404, {"error": "not found"})
            self.reply(200, waker.status())

        def do_POST(self):
            if self.path != "/wake/start" or not self.host_ok():
                return self.reply(404, {"error": "not found"})
            site = self.headers.get("Sec-Fetch-Site")
            origin = self.headers.get("Origin", "")
            same = site == "same-origin" if site else origin.split("://")[-1] in allowed
            if not same:
                return self.reply(403, {"error": "Start must come from the VisionGuard site"})
            if int(self.headers.get("Content-Length") or 0) > 1024:
                return self.reply(413, {"error": "body too large"})
            code, body = waker.request_start()
            with waker.lock:
                waker.last_snapshot = waker.snapshot()
            self.reply(code, body)

    return Handler


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = json.loads(CONFIG_PATH.read_text())
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    waker = Waker(cfg)

    def loop():
        while True:
            try:
                waker.tick()
            except Exception:
                log.exception("tick failed")
            fast = waker.cycle or waker.pending_start or now() < waker.watched_until
            waker.wake_loop.wait(3 if fast else 30)
            waker.wake_loop.clear()

    threading.Thread(target=loop, daemon=True).start()
    server = ThreadingHTTPServer(("127.0.0.1", cfg.get("port", 9100)), make_handler(waker))
    log.info("listening on 127.0.0.1:%s", cfg.get("port", 9100))
    server.serve_forever()


if __name__ == "__main__":
    main()
