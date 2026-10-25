#!/usr/bin/env python3
"""Run the experiment on a RunPod GPU pod from this machine, with a hard spending cap.

Subcommands (RUNPOD_API_KEY in .env; an SSH key pair at ~/.ssh/runpod_sc is created on first use):
  gpus                         list GPU types with current prices
  up      --gpu ID --count N   create a pod (CUDA >= 13 hosts, public IP, SSH) and wait until SSH answers
  setup   --config C           rsync code, data, teacher-stage outputs and .env up; install the pinned stack;
                               download the model (scripts/cloud/setup_remote.sh)
  run     --config C           start scripts/run_all.py on the pod in the background (nohup)
  guard   --config C --cap USD poll every minute: pull results home periodically; terminate the pod when the run
                               finishes, fails, or spend reaches the cap (results pulled first); run this locally
  pull    --config C           copy runs/<run>/ (without merged checkpoints) and logs/<run>/ back home
  status | ssh | down          pod state / print the SSH command / terminate the pod

The pod id is kept in logs/runpod_pod.json. Spend = cost per hour x uptime, as reported by RunPod.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from sc.config import ROOT, load_config

STATE = ROOT / "logs" / "runpod_pod.json"
KEY = Path.home() / ".ssh" / "runpod_sc"
REMOTE = "/workspace/subliminal-collusion"


def rp():
    import runpod
    key = os.environ.get("RUNPOD_API_KEY")
    assert key, "RUNPOD_API_KEY missing (put it in .env and source env.sh)"
    runpod.api_key = key
    return runpod


def state():
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def ssh_target(pod):
    """(host, port) of the pod's public SSH endpoint, or None while starting."""
    rt = pod.get("runtime") or {}
    for p in rt.get("ports") or []:
        if p.get("privatePort") == 22 and p.get("isIpPublic"):
            return p["ip"], p["publicPort"]
    return None


def ssh_base(host, port):
    return ["ssh", "-i", str(KEY), "-p", str(port), "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ServerAliveInterval=30", "-o", "ConnectTimeout=20", f"root@{host}"]


def ssh(cmd: str, check=True, capture=False):
    st = state()
    r = subprocess.run(ssh_base(st["host"], st["port"]) + [cmd], check=check, text=True,
                       capture_output=capture)
    return r.stdout if capture else r.returncode


def rsync(src, dst, *extra):
    st = state()
    e = f"ssh -i {KEY} -p {st['port']} -o StrictHostKeyChecking=accept-new"
    subprocess.run(["rsync", "-az", "--partial", "-e", e, *extra, src, dst], check=True)


def cmd_gpus(a):
    r = rp()
    rows = []
    for g in r.get_gpus():
        info = r.get_gpu(g["id"])
        price = (info.get("lowestPrice") or {}).get("uninterruptablePrice")
        rows.append((g["id"], g.get("memoryInGb"), info.get("securePrice"), info.get("communityPrice"), price))
    for row in sorted(rows, key=lambda x: (x[2] or 1e9)):
        print(f"{row[0]:40s} {row[1]!s:>4} GB  secure ${row[2]}/h  community ${row[3]}/h  lowest ${row[4]}/h")


def cmd_up(a):
    r = rp()
    if not KEY.exists():
        subprocess.run(["ssh-keygen", "-t", "ed25519", "-N", "", "-C", "runpod-sc", "-f", str(KEY), "-q"], check=True)
    pod = r.create_pod(name="subliminal-collusion", image_name=a.image, gpu_type_id=a.gpu, gpu_count=a.count,
                       cloud_type=a.cloud, container_disk_in_gb=a.disk, volume_in_gb=0, ports="22/tcp",
                       support_public_ip=True, start_ssh=True, allowed_cuda_versions=a.cuda,
                       env={"PUBLIC_KEY": (KEY.with_suffix(".pub")).read_text().strip()})
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({"id": pod["id"], "created": time.time(), "gpu": a.gpu, "count": a.count}, indent=1))
    print(f"pod {pod['id']} created; waiting for SSH ...", flush=True)
    for _ in range(90):
        p = r.get_pod(pod["id"])
        t = ssh_target(p) if p else None
        if t:
            st = state() | {"host": t[0], "port": t[1], "cost_per_hr": p.get("costPerHr")}
            STATE.write_text(json.dumps(st, indent=1))
            if subprocess.run(ssh_base(*t) + ["true"], capture_output=True).returncode == 0:
                print(f"SSH ready: {' '.join(ssh_base(*t))}  (${p.get('costPerHr')}/h)")
                return
        time.sleep(20)
    raise SystemExit("pod did not become reachable in 30 min; check `status`, then `down`")


def cmd_setup(a):
    run = load_config(a.config)["run_name"]
    ssh(f"mkdir -p {REMOTE}")
    files = subprocess.check_output(["git", "-C", str(ROOT), "ls-files"], text=True).split()
    lst = ROOT / "logs" / "rsync_files.txt"
    lst.write_text("\n".join(files + [".env"]) + "\n")
    rsync(f"{ROOT}/", f"root@{state()['host']}:{REMOTE}/", f"--files-from={lst}")
    for d in (f"runs/{run}", f"logs/{run}"):          # teacher-stage outputs and task markers, if any
        if (ROOT / d).exists():
            ssh(f"mkdir -p {REMOTE}/{d}")
            rsync(f"{ROOT}/{d}/", f"root@{state()['host']}:{REMOTE}/{d}/", "--exclude=merged/")
    ssh(f"cd {REMOTE} && SC_MODEL_PATH={REMOTE}/models/{Path(load_config(a.config)['model']['hf_id']).name} "
        f"bash scripts/cloud/setup_remote.sh {shlex.quote(a.config)} 2>&1 | tail -20")


def remote_env(cfg):
    return (f"cd {REMOTE} && export SC_VENV={REMOTE}/.venv SC_MODEL_PATH={REMOTE}/models/{Path(cfg['model']['hf_id']).name}"
            f" && set +u && source env.sh && set -u")


def cmd_run(a):
    cfg = load_config(a.config)
    run = cfg["run_name"]
    job = f"bash {shlex.quote(a.script)}" if a.script else \
        f"python scripts/run_all.py --config {shlex.quote(a.config)} {' '.join(a.extra)}"
    # setsid + stdin from /dev/null: fully detached, so this SSH call returns at once
    ssh(f"{remote_env(cfg)} && mkdir -p logs && setsid nohup {job} > logs/{run}.out 2>&1 < /dev/null &")
    print(f"started; log: logs/{run}.out on the pod (`guard` follows it)")


def cmd_pull(a):
    run = load_config(a.config)["run_name"]
    for d in (f"runs/{run}", f"logs/{run}"):
        (ROOT / d).mkdir(parents=True, exist_ok=True)
        rsync(f"root@{state()['host']}:{REMOTE}/{d}/", f"{ROOT}/{d}/", "--exclude=merged/", "--exclude=*.partial*")
    rsync(f"root@{state()['host']}:{REMOTE}/logs/{run}.out", f"{ROOT}/logs/{run}.out")
    print("pulled")


def spend(r):
    st = state()
    p = r.get_pod(st["id"])
    if not p:
        return None, None
    hrs = (time.time() - st["created"]) / 3600
    return hrs * float(p.get("costPerHr") or st.get("cost_per_hr") or 0), p


def cmd_guard(a):
    r = rp()
    run = load_config(a.config)["run_name"]
    last_pull, warned = 0.0, False
    while True:
        usd, pod = spend(r)
        if pod is None:
            print("pod gone")
            return
        tail = ssh(f"tail -3 {REMOTE}/logs/{run}.out 2>/dev/null", check=False, capture=True) or ""
        finished = "All " in tail and "tasks done" in tail
        failed = "STOPPED:" in tail
        print(f"[{time.strftime('%H:%M')}] spent ${usd:.2f} of ${a.cap:.0f} | {tail.strip().splitlines()[-1] if tail.strip() else '-'}",
              flush=True)
        if time.time() - last_pull > a.pull_every * 60 or finished or failed or usd >= a.cap * 0.97:
            try:
                cmd_pull(a)
                last_pull = time.time()
            except subprocess.CalledProcessError as e:
                print(f"pull failed: {e}")
        if finished or failed or usd >= a.cap * 0.97:
            why = "run finished" if finished else "run stopped on a failure" if failed else f"spend cap ${a.cap} reached"
            if failed and not a.terminate_on_failure and usd < a.cap * 0.97:
                # keep the pod for debugging, but keep guarding: the cap stays enforced until the pod is gone
                if not warned:
                    print(f"{why}; pod kept for debugging; still enforcing the ${a.cap} cap", flush=True)
                    warned = True
                time.sleep(60)
                continue
            r.terminate_pod(state()["id"])
            print(f"{why}: pod terminated after ${usd:.2f}")
            return
        time.sleep(60)


def cmd_status(a):
    r = rp()
    usd, p = spend(r)
    print(json.dumps({"state": state(), "pod": p and {k: p.get(k) for k in ("desiredStatus", "costPerHr", "gpuCount")},
                      "spent_usd": usd}, indent=1, default=str))


def cmd_ssh(a):
    st = state()
    print(" ".join(ssh_base(st["host"], st["port"])))


def cmd_down(a):
    rp().terminate_pod(state()["id"])
    print("terminated", state()["id"])


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("gpus")
    u = sub.add_parser("up")
    u.add_argument("--gpu", required=True, help="GPU type id from `gpus`, e.g. 'NVIDIA H100 80GB HBM3'")
    u.add_argument("--count", type=int, default=8)
    u.add_argument("--image", default="runpod/pytorch:1.4.2-rc.175-cu1300-torch291-ubuntu2404",
                   help="any image with sshd; the pinned stack is installed in a fresh venv")
    u.add_argument("--cloud", default="SECURE", choices=["SECURE", "COMMUNITY", "ALL"])
    u.add_argument("--disk", type=int, default=300)
    u.add_argument("--cuda", nargs="*", default=["13.0"], help="allowed host CUDA versions (driver >= 580)")
    for name in ("setup", "run", "pull", "guard"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        if name == "run":
            p.add_argument("extra", nargs="*")
            p.add_argument("--script", help="run this bash script instead of run_all.py (it must print the guard's markers)")
        if name == "guard":
            p.add_argument("--cap", type=float, required=True, help="hard spend cap in USD")
            p.add_argument("--pull-every", type=float, default=30, help="minutes between result pulls")
            p.add_argument("--terminate-on-failure", action="store_true")
    for name in ("status", "ssh", "down"):
        sub.add_parser(name)
    a = ap.parse_args()
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    sys.exit(main())
