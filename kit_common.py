#!/usr/bin/env python3
"""kit_common.py: shared helpers for soup_task.ipynb (Colab T4).

Every notebook cell starts with
    import sys; sys.path.insert(0, "/content/soup-task"); import kit_common as K; K.init()
so cells are self-contained after a kernel restart; state lives on disk under WORK.

Design rules
  * The notebook kernel never touches CUDA. Every GPU step (soup train/merge/ship, fallbacks,
    verification) runs as a subprocess through the `peak` wrapper below, which records that
    process's torch.cuda.max_memory_allocated/reserved and peak RSS at exit. nvidia-smi totals
    during a stage can then be attributed to one process.
  * Long commands run in the background (`run_bg`) with a timestamped log per command, and are
    awaited with `wait_bg` (re-run the cell to keep waiting). Re-running a finished step is a
    no-op unless restart=True: cells are idempotent.
  * Nothing in here invents a number. Summaries are computed from files the run wrote.

CLI subcommands (used by the notebook through subprocesses):
  bg-runner, peak, fallback-train, fallback-merge, fallback-quantize
"""

from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import json
import logging
import math
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

WORK = Path(os.environ.get("SOUP_TASK_DIR", "/content/soup-task")).resolve()
LOGS = WORK / "logs"
CMD_LOGS = LOGS / "cmd"
OUT = WORK / "outputs"
DATA = WORK / "data"

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET_ID = "d0rj/alpaca-cleaned-ru"
SEED = 42
N_TRAIN, N_EVAL, N_NC = 1500, 200, 64
MAX_LEN = 1024
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
HP = {
    "epochs": 2, "lr": 2e-4, "batch_size": 2, "grad_accum": 8, "max_length": MAX_LEN,
    "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "target_modules": TARGET_MODULES,
    "warmup_ratio": 0.03, "weight_decay": 0.0, "max_grad_norm": 1.0, "scheduler": "cosine",
    "logging_steps": 5, "save_steps": 50,
}
# Negative control: same pipeline, learning rate ~0 (Soup's schema requires lr > 0), 64 rows, 1 epoch.
NC_HP = dict(HP, lr=1e-9, epochs=1, save_steps=1000)
NVSMI_INTERVAL_S = 1
LEAK_JACCARD = 0.6       # eval vs train, word 3-gram Jaccard of the user turn
DUP_JACCARD = 0.8        # within train

SOUP_RUN = OUT / "soup_run"
FALLBACK_RUN = OUT / "fallback_run"
MERGED = OUT / "merged"
MERGED_4BIT = OUT / "merged_4bit"
NC_RUN = OUT / "nc_run"
NC_MERGED = OUT / "nc_merged"
VERIFY_DIR = OUT / "verify"
VERIFY_NC2_DIR = OUT / "verify_nc2"

_LOGGER_NAME = "soup_task"
log = logging.getLogger(_LOGGER_NAME)


# ======================================================================================
# basics
# ======================================================================================
def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="milliseconds")


def rel(p) -> str:
    """Path relative to WORK (Soup refuses outputs outside the cwd)."""
    p = Path(p)
    try:
        return os.path.relpath(p.resolve(), WORK)
    except ValueError:
        return str(p)


def init(verbose: bool = True):
    for d in (LOGS, CMD_LOGS, OUT, DATA, LOGS / "soup_help"):
        d.mkdir(parents=True, exist_ok=True)
    os.chdir(WORK)
    if str(WORK) not in sys.path:
        sys.path.insert(0, str(WORK))
    lg = logging.getLogger(_LOGGER_NAME)
    if not any(getattr(h, "_soup_task", False) for h in lg.handlers):
        lg.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s.%(msecs)03d %(levelname)s %(message)s", "%Y-%m-%dT%H:%M:%S")
        fh = logging.FileHandler(LOGS / "run.log", encoding="utf-8")
        fh.setFormatter(fmt)
        fh._soup_task = True
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        sh._soup_task = True
        lg.addHandler(fh)
        lg.addHandler(sh)
        lg.propagate = False
        lg.info(f"SESSION init: WORK={WORK} tz={time.tzname} python={sys.version.split()[0]} pid={os.getpid()}")
    return lg


def read_json(p, default=None):
    try:
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return default


def write_json(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, p)


def load_jsonl(p, limit=None):
    rows = []
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def write_jsonl(p, rows):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def paths() -> dict:
    return read_json(OUT / "paths.json", {}) or {}


def set_path(key, value):
    d = paths()
    d[key] = None if value is None else str(value)
    write_json(OUT / "paths.json", d)
    log.info(f"paths.json: {key} = {d[key]}")


def sha256_file(p, limit_bytes=None) -> str | None:
    p = Path(p)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        n = 0
        while True:
            b = fh.read(1 << 20)
            if not b:
                break
            h.update(b)
            n += len(b)
            if limit_bytes and n >= limit_bytes:
                break
    return h.hexdigest()[:16]


# ======================================================================================
# nvidia-smi: background sampler (raw CSV) + full snapshots
# ======================================================================================
NVSMI_QUERY = ("timestamp,index,name,memory.used,memory.total,utilization.gpu,utilization.memory,"
               "temperature.gpu,power.draw")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    stat = Path(f"/proc/{pid}/stat")
    if stat.exists():
        try:
            state = stat.read_text().split(")")[-1].split()[0]
            if state == "Z":
                return False
        except Exception:
            pass
    return True


def start_gpu_sampler(interval_s: int = NVSMI_INTERVAL_S):
    pidf = LOGS / "nvidia_smi.pid"
    if pidf.exists():
        try:
            pid = int(pidf.read_text().strip())
            cmdline = Path(f"/proc/{pid}/cmdline")
            if _alive(pid) and (not cmdline.exists() or b"nvidia-smi" in cmdline.read_bytes()):
                log.info(f"nvidia-smi sampler already running (pid {pid})")
                return pid
        except Exception:
            pass
    if shutil.which("nvidia-smi") is None:
        log.warning("nvidia-smi not found: no GPU sampler (is the runtime a GPU runtime?)")
        return None
    fh = open(LOGS / "nvidia_smi.log", "a", encoding="utf-8")
    p = subprocess.Popen(["nvidia-smi", f"--query-gpu={NVSMI_QUERY}", "--format=csv", "-l", str(interval_s)],
                         stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
    pidf.write_text(str(p.pid))
    log.info(f"nvidia-smi sampler started pid={p.pid} every {interval_s}s -> {rel(LOGS / 'nvidia_smi.log')}")
    return p.pid


def stop_gpu_sampler():
    pidf = LOGS / "nvidia_smi.pid"
    if pidf.exists():
        try:
            os.kill(int(pidf.read_text().strip()), signal.SIGTERM)
            log.info("nvidia-smi sampler stopped")
        except Exception as exc:
            log.info(f"sampler stop: {exc!r}")
        pidf.unlink(missing_ok=True)


def nvsmi_snapshot(label: str):
    if shutil.which("nvidia-smi") is None:
        return
    try:
        full = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30).stdout
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"],
                              capture_output=True, text=True, timeout=30).stdout
    except Exception as exc:
        full, apps = f"nvidia-smi failed: {exc!r}\n", ""
    with open(LOGS / "nvidia_smi_snapshots.log", "a", encoding="utf-8") as fh:
        fh.write(f"===== {now_iso()} {label} =====\n{full}\n{apps}\n")


def nvsmi_samples():
    f = LOGS / "nvidia_smi.log"
    out = []
    if not f.exists():
        return out
    for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 5 or parts[0].startswith("timestamp"):
            continue
        try:
            ts = dt.datetime.strptime(parts[0], "%Y/%m/%d %H:%M:%S.%f")
            used = float(parts[3].split()[0])
            total = float(parts[4].split()[0])
            util = float(parts[5].split()[0]) if len(parts) > 5 and parts[5].split()[0].replace(".", "").isdigit() else None
        except Exception:
            continue
        out.append((ts, used, total, util))
    return out


def nvsmi_window(t0_epoch: float, t1_epoch: float) -> dict:
    s = nvsmi_samples()
    if not s:
        return {"nvsmi_samples": 0}
    t0 = dt.datetime.fromtimestamp(t0_epoch)
    t1 = dt.datetime.fromtimestamp(t1_epoch)
    win = [x for x in s if t0 <= x[0] <= t1]
    before = [x for x in s if x[0] < t0]
    baseline = before[-1][1] if before else (win[0][1] if win else None)
    if not win:
        return {"nvsmi_samples": 0, "nvsmi_baseline_gib": baseline / 1024 if baseline is not None else None}
    utils = [x[3] for x in win if x[3] is not None]
    return {
        "nvsmi_samples": len(win),
        "nvsmi_max_used_gib": max(x[1] for x in win) / 1024,
        "nvsmi_baseline_gib": baseline / 1024 if baseline is not None else None,
        "nvsmi_total_gib": win[0][2] / 1024,
        "nvsmi_mean_util_pct": sum(utils) / len(utils) if utils else None,
    }


# ======================================================================================
# stages: timestamps + memory per stage -> logs/stage_memory.jsonl
# ======================================================================================
def _kernel_torch_peaks() -> dict:
    t = sys.modules.get("torch")
    if t is None:
        return {"kernel_torch_imported": False}
    try:
        if t.cuda.is_available() and t.cuda.is_initialized():
            return {"kernel_torch_max_allocated_gib": t.cuda.max_memory_allocated() / 2**30,
                    "kernel_torch_max_reserved_gib": t.cuda.max_memory_reserved() / 2**30}
    except Exception as exc:
        return {"kernel_torch_error": repr(exc)}
    return {"kernel_cuda_initialized": False}


def _append_stage(rec: dict):
    with open(LOGS / "stage_memory.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")


@contextlib.contextmanager
def stage(name: str):
    """In-kernel stage: timestamps, nvidia-smi snapshot + window, kernel torch peaks."""
    t0 = time.time()
    rec = {"stage": name, "started": now_iso(), "start_epoch": t0, "where": "kernel"}
    log.info(f"STAGE START {name}")
    nvsmi_snapshot(f"start {name}")
    t = sys.modules.get("torch")
    try:
        if t is not None and t.cuda.is_available() and t.cuda.is_initialized():
            t.cuda.reset_peak_memory_stats()
    except Exception:
        pass
    try:
        yield rec
        rec.setdefault("status", "ok")
    except BaseException as exc:
        rec["status"] = f"error: {exc!r}"[:300]
        raise
    finally:
        t1 = time.time()
        rec.update(finished=now_iso(), end_epoch=t1, seconds=round(t1 - t0, 1))
        rec.update(_kernel_torch_peaks())
        rec.update(nvsmi_window(t0, t1))
        nvsmi_snapshot(f"end {name}")
        _append_stage(rec)
        log.info(f"STAGE END {name} status={rec['status']} {rec['seconds']}s")


def stage_records() -> list:
    f = LOGS / "stage_memory.jsonl"
    return [json.loads(x) for x in f.read_text().splitlines() if x.strip()] if f.exists() else []


# ======================================================================================
# background commands with timestamped logs
# ======================================================================================
def _cmd_paths(name):
    return (CMD_LOGS / f"{name}.log", CMD_LOGS / f"{name}.rc", CMD_LOGS / f"{name}.pid", CMD_LOGS / f"{name}.meta.json")


def bg_status(name):
    log_p, rc_p, pid_p, meta_p = _cmd_paths(name)
    if rc_p.exists():
        try:
            return "done", int(rc_p.read_text().strip())
        except ValueError:
            return "done", -999
    if pid_p.exists():
        try:
            if _alive(int(pid_p.read_text().strip())):
                return "running", None
        except ValueError:
            pass
        return "dead", None
    return "absent", None


def peak_cmd(peak_json, soup_args=None, script=None, script_args=()):
    """Wrap a soup CLI call or a python script so its CUDA/RSS peaks are dumped at exit."""
    base = [sys.executable, str(WORK / "kit_common.py"), "peak", "--json", rel(peak_json)]
    if soup_args is not None:
        return base + ["--soup", "--"] + [str(x) for x in soup_args]
    return base + ["--script", str(script), "--"] + [str(x) for x in script_args]


def run_bg(name, args, restart=False, timeout_s=None, env_extra=None, peak_json=None):
    st, rc = bg_status(name)
    if st == "running":
        log.info(f"[{name}] already running; use K.wait_bg('{name}')")
        return st
    if st == "done" and not restart:
        log.info(f"[{name}] already finished rc={rc}; pass restart=True to run again")
        return st
    if st in ("done", "dead"):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        for p in _cmd_paths(name) + (CMD_LOGS / f"{name}.recorded",):
            if p.exists():
                p.rename(p.with_name(p.name + f".prev-{stamp}"))
    log_p, rc_p, pid_p, meta_p = _cmd_paths(name)
    meta = {"name": name, "cmd": [str(a) for a in args], "started": now_iso(), "start_epoch": time.time(),
            "cwd": str(WORK), "peak_json": rel(peak_json) if peak_json else None, "timeout_s": timeout_s}
    write_json(meta_p, meta)
    runner = [sys.executable, str(WORK / "kit_common.py"), "bg-runner", "--name", name]
    if timeout_s:
        runner += ["--timeout", str(int(timeout_s))]
    runner += ["--"] + [str(a) for a in args]
    env = os.environ.copy()
    env.update({"PYTHONUNBUFFERED": "1", "TQDM_MININTERVAL": "10", "COLUMNS": "200",
                "HF_HUB_DISABLE_PROGRESS_BARS": "1", "TOKENIZERS_PARALLELISM": "false"})
    env.update(env_extra or {})
    err = open(CMD_LOGS / f"{name}.runner.err", "a", encoding="utf-8")
    p = subprocess.Popen(runner, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=err,
                         cwd=str(WORK), env=env, start_new_session=True)
    pid_p.write_text(str(p.pid))
    log.info(f"[{name}] started pid={p.pid}: {' '.join(str(a) for a in args)}")
    return "started"


def cmd_output(name, strip_ts=True) -> str:
    log_p = _cmd_paths(name)[0]
    if not log_p.exists():
        return ""
    txt = log_p.read_text(encoding="utf-8", errors="replace")
    if not strip_ts:
        return txt
    return "\n".join(line.split(" | ", 1)[1] if " | " in line[:40] else line for line in txt.splitlines())


def _record_bg_stage(name):
    marker = CMD_LOGS / f"{name}.recorded"
    if marker.exists():
        return
    meta = read_json(_cmd_paths(name)[3], {}) or {}
    t0, t1 = meta.get("start_epoch"), meta.get("end_epoch") or time.time()
    rec = {"stage": name, "where": "subprocess", "started": meta.get("started"), "finished": meta.get("finished"),
           "start_epoch": t0, "end_epoch": t1, "seconds": round(t1 - t0, 1) if t0 else None, "rc": meta.get("rc"),
           "cmd": meta.get("cmd")}
    if t0:
        rec.update(nvsmi_window(t0, t1))
    if meta.get("peak_json"):
        pk = read_json(WORK / meta["peak_json"])
        if pk:
            rec["subprocess_peak"] = pk
    _append_stage(rec)
    marker.write_text(now_iso())


def wait_bg(name, max_wait_s=1200, poll_s=15, show_lines=25):
    """Stream new log lines of a background command; returns rc when done, None if still running."""
    log_p = _cmd_paths(name)[0]
    t0 = time.time()
    off = max(0, log_p.stat().st_size - 3000) if log_p.exists() else 0
    nap = 0.25
    while True:
        if log_p.exists():
            with open(log_p, "rb") as fh:
                fh.seek(off)
                new = fh.read()
                off += len(new)
            lines = new.decode("utf-8", errors="replace").splitlines() if show_lines else []
            if len(lines) > show_lines:
                print(f"   ... {len(lines) - show_lines} lines not shown (full log: {rel(log_p)})")
                lines = lines[-show_lines:]
            for line in lines:
                print("   " + line[:400])
        st, rc = bg_status(name)
        if st == "done":
            _record_bg_stage(name)
            log.info(f"[{name}] finished rc={rc}")
            return rc
        if st in ("dead", "absent"):
            log.error(f"[{name}] status={st} (runner died or never started); see {rel(CMD_LOGS / (name + '.runner.err'))}")
            return None
        if max_wait_s is not None and time.time() - t0 > max_wait_s:
            log.info(f"[{name}] still running after {int(time.time() - t0)}s; re-run this cell to keep waiting")
            return None
        time.sleep(nap)
        nap = min(poll_s, nap * 1.6)   # quick commands return fast, long ones are polled every poll_s


def run_cmd(name, args, timeout_s=None, peak_json=None, quiet=False, env_extra=None):
    """Foreground run (always re-runs). Returns (rc, output_without_timestamps)."""
    run_bg(name, args, restart=True, timeout_s=timeout_s, peak_json=peak_json, env_extra=env_extra)
    rc = wait_bg(name, max_wait_s=None, poll_s=1 if timeout_s is None or timeout_s < 120 else 5,
                 show_lines=0 if quiet else 60)
    return rc, cmd_output(name)


def _split_opts(argv):
    """'--k v --flag -- rest...' -> ({k: v, flag: True}, rest). No argparse: the rest may contain anything."""
    if "--" in argv:
        i = argv.index("--")
        head, rest = argv[:i], argv[i + 1:]
    else:
        head, rest = argv, []
    opts, j = {}, 0
    while j < len(head):
        k = head[j].lstrip("-")
        if j + 1 < len(head) and not head[j + 1].startswith("--"):
            opts[k] = head[j + 1]
            j += 2
        else:
            opts[k] = True
            j += 1
    return opts, rest


class _A:
    pass


def _bg_runner_main(argv):
    import select

    o, cmd = _split_opts(argv)
    a = _A()
    a.name = o["name"]
    a.timeout = int(o["timeout"]) if o.get("timeout") not in (None, True) else None
    log_p, rc_p, pid_p, meta_p = _cmd_paths(a.name)
    pid_p.write_text(str(os.getpid()))
    t0 = time.time()
    rc = None
    with open(log_p, "ab") as lf:
        def w(seg: bytes):
            lf.write(now_iso().encode() + b" | " + seg.rstrip(b"\r\n") + b"\n")
            lf.flush()

        w(("$ " + " ".join(cmd)).encode())
        try:
            p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 cwd=str(WORK), start_new_session=True)
        except Exception as exc:
            w(f"failed to start: {exc!r}".encode())
            rc = 127
        else:
            fd = p.stdout.fileno()
            buf = b""
            timed_out = False
            while True:
                if a.timeout and time.time() - t0 > a.timeout:
                    timed_out = True
                    w(f"TIMEOUT after {a.timeout}s: killing process group".encode())
                    try:
                        os.killpg(p.pid, signal.SIGKILL)
                    except Exception:
                        p.kill()
                    break
                r, _, _ = select.select([fd], [], [], 1.0)
                if not r:
                    continue
                chunk = os.read(fd, 65536)
                if not chunk:
                    break
                buf += chunk
                parts = re.split(rb"(\r\n|\n|\r)", buf)
                buf = parts[-1]
                for i in range(0, len(parts) - 1, 2):
                    if parts[i].strip():
                        w(parts[i])
            if buf.strip():
                w(buf)
            rc = p.wait()
            if timed_out:
                rc = 124
        w(f"exit code {rc} after {time.time() - t0:.1f}s".encode())
    meta = read_json(meta_p, {}) or {}
    meta.update(finished=now_iso(), end_epoch=time.time(), rc=rc)
    write_json(meta_p, meta)
    rc_p.write_text(str(rc))
    return 0


def _soup_entry():
    try:
        from importlib.metadata import entry_points

        eps = entry_points(group="console_scripts")
        for ep in eps:
            if ep.name == "soup":
                return ep.load()
    except Exception:
        pass
    from soup_cli.cli import run  # v0.75 layout

    return run


def _peak_main(argv):
    import atexit
    import runpy

    o, rest = _split_opts(argv)
    a = _A()
    a.json = o["json"]
    a.soup = bool(o.get("soup"))
    a.script = o.get("script") if o.get("script") is not True else None
    t0 = time.time()
    started = now_iso()
    out = Path(a.json)
    out.parent.mkdir(parents=True, exist_ok=True)

    def dump():
        rec = {"argv": rest, "soup": a.soup, "script": a.script, "started": started, "finished": now_iso(),
               "wall_s": round(time.time() - t0, 1), "pid": os.getpid()}
        try:
            import resource

            div = 2**30 if sys.platform == "darwin" else 2**20   # ru_maxrss: bytes on macOS, KiB on Linux
            rec["ru_maxrss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / div
            rec["ru_maxrss_children_gib"] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / div
        except Exception:
            pass
        t = sys.modules.get("torch")
        if t is None:
            rec["torch_imported"] = False
        else:
            try:
                if t.cuda.is_available() and t.cuda.is_initialized():
                    rec.update(cuda_initialized=True, device=t.cuda.get_device_name(0),
                               torch_max_allocated_gib=t.cuda.max_memory_allocated() / 2**30,
                               torch_max_reserved_gib=t.cuda.max_memory_reserved() / 2**30)
                    try:
                        out.with_suffix(".memory_summary.txt").write_text(t.cuda.memory_summary())
                    except Exception:
                        pass
                else:
                    rec["cuda_initialized"] = False
            except Exception as exc:
                rec["torch_error"] = repr(exc)
        out.write_text(json.dumps(rec, indent=2))

    atexit.register(dump)
    if a.soup:
        sys.argv = ["soup"] + rest
        fn = _soup_entry()
        res = fn()
        sys.exit(res if isinstance(res, int) else 0)
    else:
        sys.argv = [a.script] + rest
        runpy.run_path(a.script, run_name="__main__")


# ======================================================================================
# Soup CLI discovery (the PyPI version can differ from the docs)
# ======================================================================================
HELP_TARGETS = [[], ["train"], ["merge"], ["data"], ["ship"], ["profile"], ["plan"], ["quantize"], ["export"],
                ["doctor"], ["diagnose"], ["eval"], ["data", "validate"], ["data", "stats"], ["data", "inspect"]]


def soup_caps(refresh=False) -> dict:
    f = OUT / "soup_capabilities.json"
    if f.exists() and not refresh:
        return read_json(f, {})
    caps = {"checked": now_iso()}
    if shutil.which("soup") is None:
        caps["soup_on_path"] = False
        write_json(f, caps)
        log.error("`soup` not on PATH: install cell failed?")
        return caps
    caps["soup_on_path"] = True
    rc, out = run_cmd("soup_version", ["soup", "--version"], quiet=True)
    if rc != 0:
        rc, out = run_cmd("soup_version", ["soup", "version"], quiet=True)
    m = re.search(r"(\d+\.\d+\.\d+)", out or "")
    caps["version"] = m.group(1) if m else None
    texts = {}
    for sub in HELP_TARGETS:
        name = "help_" + "_".join(sub or ["root"])
        rc, out = run_cmd(name, ["soup", *sub, "--help"], quiet=True)
        (LOGS / "soup_help" / f"{name}.txt").write_text(out, encoding="utf-8")
        texts[name] = out if rc == 0 else ""
        caps[name + "_rc"] = rc
    th = texts.get("help_data", "")
    caps["data_has_doctor"] = bool(re.search(r"\bdoctor\b", th))
    if caps["data_has_doctor"]:
        rc, out = run_cmd("help_data_doctor", ["soup", "data", "doctor", "--help"], quiet=True)
        (LOGS / "soup_help" / "help_data_doctor.txt").write_text(out, encoding="utf-8")
        texts["help_data_doctor"] = out
    caps["data_subcommands_seen"] = sorted(set(re.findall(
        r"\b(inspect|validate|convert|dedup|filter|stats|sample|split|preview|download|preprocess|doctor|lint|"
        r"decontaminate|score|langdetect)\b", th)))
    mh = texts.get("help_merge", "")
    caps["merge_has_adapter_flag"] = "--adapter" in mh
    caps["merge_has_save_format"] = "--save-format" in mh
    caps["merge_has_low_memory"] = "--low-memory" in mh
    caps["merge_has_dtype"] = "--dtype" in mh
    caps["train_has_dry_run"] = "--dry-run" in texts.get("help_train", "")
    # v0.77.2: `soup train` refuses to start on a non-TTY stdin without --yes (our subprocesses have no TTY)
    caps["train_has_yes"] = "--yes" in texts.get("help_train", "")
    caps["profile_has_json"] = "--json" in texts.get("help_profile", "")
    caps["has_plan"] = caps.get("help_plan_rc") == 0
    sh = texts.get("help_ship", "")
    caps["ship_has_tuned"] = "--tuned" in sh
    caps["ship_has_general_suite"] = "--general-suite" in sh
    caps["ship_has_device"] = "--device" in sh
    caps["eval_has_quant_check"] = "quant-check" in texts.get("help_eval", "")
    caps["doctor_has_config"] = "--config" in texts.get("help_doctor", "")
    write_json(f, caps)
    log.info(f"soup capabilities: {json.dumps(caps)}")
    return caps


def soup_train_args(config, dry_run=False):
    caps = soup_caps()
    a = ["train", "--config", rel(config)]
    if dry_run:
        a.append("--dry-run")
    if caps.get("train_has_yes"):
        a.append("--yes")
    return a


def soup_merge_args(adapter, output, extra=()):
    caps = soup_caps()
    a = ["merge", "--adapter", rel(adapter)] if caps.get("merge_has_adapter_flag") else ["merge", rel(adapter)]
    a += ["--output", rel(output)]
    if caps.get("merge_has_dtype"):
        a += ["--dtype", "float16"]
    return a + list(extra)


# ======================================================================================
# data preparation
# ======================================================================================
def _norm(t: str) -> str:
    t = re.sub(r"[^\w\s]", " ", (t or "").lower())
    return re.sub(r"\s+", " ", t).strip()


def _shingles(t: str):
    w = _norm(t).split()
    if len(w) < 3:
        return set(w)
    return {tuple(w[i:i + 3]) for i in range(len(w) - 2)}


class NearDupIndex:
    def __init__(self):
        self.sets, self.inv = [], defaultdict(list)

    def max_sim(self, s):
        if not s:
            return 0.0, None
        cnt = Counter()
        for g in s:
            for j in self.inv.get(g, ()):
                cnt[j] += 1
        best, bj = 0.0, None
        for j, inter in cnt.items():
            sim = inter / (len(s) + len(self.sets[j]) - inter)
            if sim > best:
                best, bj = sim, j
        return best, bj

    def add(self, s):
        j = len(self.sets)
        self.sets.append(s)
        for g in s:
            self.inv[g].append(j)
        return j


def cyr_ratio(s: str) -> float:
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if "Ѐ" <= c <= "ӿ") / len(letters)


def _pct(vals, q):
    if not vals:
        return None
    v = sorted(vals)
    k = (len(v) - 1) * q / 100
    f, c = math.floor(k), math.ceil(k)
    return float(v[f] if f == c else v[f] + (v[c] - v[f]) * (k - f))


def _stats(vals):
    if not vals:
        return {}
    return {"n": len(vals), "min": min(vals), "p50": _pct(vals, 50), "p90": _pct(vals, 90), "p95": _pct(vals, 95),
            "p99": _pct(vals, 99), "max": max(vals), "mean": sum(vals) / len(vals)}


OOD_PROMPTS = [
    ("ood_sky", "Объясни простыми словами, почему небо голубое. Ответь в 3–4 предложениях."),
    ("ood_math", "Сколько будет 17 умножить на 23? Покажи вычисление."),
    ("ood_letter", "Напиши короткое официальное письмо с просьбой перенести встречу на пятницу."),
    ("ood_capital", "Назови столицу Казахстана и одну интересную деталь о ней."),
    ("ood_translate", "Переведи на русский: 'The weather is nice today, let's go for a walk.'"),
    ("ood_oneword", "Ответь одним словом: какого цвета снег?"),
    ("ood_list", "Составь список из 5 советов, как подготовиться к собеседованию."),
]

SHIP_FACTS = [  # leg-1 'contains' items for soup ship: basic knowledge that must not regress
    ("Какая столица Франции? Ответь одним словом.", "Париж"),
    ("Сколько будет 2 плюс 2? Ответь числом.", "4"),
    ("Какая самая большая планета Солнечной системы? Ответь одним словом.", "Юпитер"),
    ("На каком континенте находится Египет? Ответь одним словом.", "Африк"),
    ("Сколько дней в неделе? Ответь числом.", "7"),
]


def _load_raw_dataset():
    sha = None
    try:
        from huggingface_hub import HfApi

        sha = HfApi().dataset_info(DATASET_ID).sha
    except Exception as exc:
        log.warning(f"dataset_info failed ({exc!r}); revision not pinned")
    try:
        from datasets import load_dataset

        ds = load_dataset(DATASET_ID, split="train", revision=sha)
        return ds.to_list(), sha, "datasets.load_dataset"
    except Exception as exc:
        log.warning(f"datasets.load_dataset failed ({exc!r}); trying the parquet file directly")
    from huggingface_hub import HfApi, hf_hub_download
    import pandas as pd

    files = [f for f in HfApi().list_repo_files(DATASET_ID, repo_type="dataset", revision=sha) if f.endswith(".parquet")]
    frames = [pd.read_parquet(hf_hub_download(DATASET_ID, f, repo_type="dataset", revision=sha)) for f in files]
    return pd.concat(frames).to_dict("records"), sha, "parquet"


def prepare_data(force=False) -> dict:
    report_p = DATA / "data_report.json"
    if report_p.exists() and (DATA / "train.jsonl").exists() and not force:
        log.info("data already prepared (force=True to redo)")
        return read_json(report_p)
    from transformers import AutoTokenizer
    from verify_trained import TemplateError, encode_example

    rows, sha, how = _load_raw_dataset()
    log.info(f"loaded {len(rows)} rows from {DATASET_ID}@{sha} via {how}")
    reasons = Counter()
    seen_user, seen_out = set(), set()
    pool = []
    for i, r in enumerate(rows):
        instr = (r.get("instruction") or "").strip()
        inp = (r.get("input") or "").strip()
        out = (r.get("output") or "").strip()
        if not instr or not out:
            reasons["empty_instruction_or_output"] += 1
            continue
        if len(out) < 3:
            reasons["output_too_short"] += 1
            continue
        if "```" in out or "```" in inp:
            reasons["code_block"] += 1
            continue
        user = instr + ("\n\n" + inp if inp else "")
        if cyr_ratio(out) < 0.6 or cyr_ratio(user) < 0.5:
            reasons["low_cyrillic"] += 1
            continue
        nu, no = _norm(user), _norm(out)
        if nu in seen_user:
            reasons["exact_dup_user"] += 1
            continue
        if no in seen_out:
            reasons["exact_dup_output"] += 1
            continue
        seen_user.add(nu)
        seen_out.add(no)
        pool.append({"source_index": i, "user": user, "assistant": out})
    log.info(f"after filters: {len(pool)} candidates; dropped {dict(reasons)}")
    random.Random(SEED).shuffle(pool)

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    eval_rows, train_rows = [], []
    eval_idx, train_idx = NearDupIndex(), NearDupIndex()
    eval_outs = set()
    lens = {"train": [], "eval": []}
    max_sims = []
    for c in pool:
        if len(eval_rows) >= N_EVAL and len(train_rows) >= N_TRAIN:
            break
        msgs = [{"role": "user", "content": c["user"]}, {"role": "assistant", "content": c["assistant"]}]
        try:
            en = encode_example(tok, msgs, max_length=MAX_LEN)
        except TemplateError:
            reasons["template_error"] += 1
            continue
        if en["truncated"]:
            reasons["longer_than_max_length"] += 1
            continue
        if len(en["answer_idx"]) < 2:
            reasons["no_answer_tokens"] += 1
            continue
        sh = _shingles(c["user"])
        ln = {"total": en["n_total"], "prompt": en["n_prompt_tokens"], "answer": len(en["answer_idx"])}
        if len(eval_rows) < N_EVAL:
            eval_rows.append({"id": f"eval-{len(eval_rows):04d}", "source_index": c["source_index"], "messages": msgs})
            eval_idx.add(sh)
            eval_outs.add(_norm(c["assistant"]))
            lens["eval"].append(ln)
            continue
        s_eval, _ = eval_idx.max_sim(sh)
        max_sims.append(s_eval)
        if s_eval >= LEAK_JACCARD or _norm(c["assistant"]) in eval_outs:
            reasons["near_dup_of_eval"] += 1
            continue
        s_tr, _ = train_idx.max_sim(sh)
        if s_tr >= DUP_JACCARD:
            reasons["near_dup_within_train"] += 1
            continue
        train_idx.add(sh)
        train_rows.append({"id": f"train-{len(train_rows):04d}", "source_index": c["source_index"], "messages": msgs})
        lens["train"].append(ln)
    if len(train_rows) < N_TRAIN or len(eval_rows) < N_EVAL:
        log.warning(f"only {len(train_rows)} train / {len(eval_rows)} eval rows available")

    # final leakage audit (independent of the filter above): exact user/answer overlap
    tr_u = {_norm(r["messages"][0]["content"]) for r in train_rows}
    tr_a = {_norm(r["messages"][1]["content"]) for r in train_rows}
    audit = {"eval_user_in_train": sum(_norm(r["messages"][0]["content"]) in tr_u for r in eval_rows),
             "eval_answer_in_train": sum(_norm(r["messages"][1]["content"]) in tr_a for r in eval_rows),
             "train_eval_source_index_overlap": len({r["source_index"] for r in train_rows} & {r["source_index"] for r in eval_rows})}

    write_jsonl(DATA / "train.jsonl", [{"messages": r["messages"]} for r in train_rows])
    write_jsonl(DATA / "train_meta.jsonl", [{"id": r["id"], "source_index": r["source_index"]} for r in train_rows])
    write_jsonl(DATA / "eval.jsonl", eval_rows)
    write_jsonl(DATA / "nc_train.jsonl", [{"messages": r["messages"]} for r in train_rows[:N_NC]])
    probes = [{"id": f"indist_{i}", "kind": "in_dist", "prompt": r["messages"][0]["content"],
               "reference": r["messages"][1]["content"]} for i, r in enumerate(eval_rows[:5])]
    probes += [{"id": pid, "kind": "ood", "prompt": p, "reference": None} for pid, p in OOD_PROMPTS]
    write_jsonl(DATA / "probe_prompts.jsonl", probes)
    ship = [{"prompt": r["messages"][0]["content"], "expected": r["messages"][1]["content"], "scoring": "semantic",
             "category": "heldout_semantic"} for r in eval_rows[:30]]
    ship += [{"prompt": q, "expected": a, "scoring": "contains", "category": "facts_ru"} for q, a in SHIP_FACTS]
    ship += [{"prompt": p, "expected": r"[А-Яа-яЁё]{4,}", "scoring": "regex", "category": "answers_in_russian"}
             for _, p in OOD_PROMPTS[:5]]
    write_jsonl(DATA / "ship_tasks.jsonl", ship)

    rep = {
        "created": now_iso(), "dataset": DATASET_ID, "dataset_revision": sha, "load_method": how, "seed": SEED,
        "model_tokenizer": MODEL_ID, "max_length": MAX_LEN, "n_raw": len(rows), "n_pool_after_filters": len(pool),
        "n_train": len(train_rows), "n_eval": len(eval_rows), "n_nc_train": min(N_NC, len(train_rows)),
        "drop_reasons": dict(reasons), "leak_jaccard_threshold": LEAK_JACCARD, "dup_jaccard_threshold": DUP_JACCARD,
        "leakage_audit": audit,
        "train_vs_eval_max_jaccard": _stats(max_sims),
        "token_stats": {f"{split}_{k}": _stats([x[k] for x in lens[split]]) for split in ("train", "eval")
                        for k in ("total", "prompt", "answer")},
        "cyrillic_ratio_answers": _stats([cyr_ratio(r["messages"][1]["content"]) for r in train_rows]),
        "default_system_prompt_injected": "You are Qwen" in tok.apply_chat_template(
            [{"role": "user", "content": "x"}], tokenize=False, add_generation_prompt=True),
        "files": {k: rel(DATA / k) for k in ("train.jsonl", "eval.jsonl", "nc_train.jsonl", "probe_prompts.jsonl",
                                              "ship_tasks.jsonl", "train_meta.jsonl")},
    }
    write_json(report_p, rep)
    log.info(f"data ready: train={len(train_rows)} eval={len(eval_rows)} audit={audit}")
    return rep


# ======================================================================================
# chat template / loss-mask cross-check (CPU)
# ======================================================================================
def masking_check(n=50) -> dict:
    from transformers import AutoTokenizer
    from verify_trained import encode_example

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    rows = load_jsonl(DATA / "train.jsonl", n)
    header = tok("<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]
    res = {"n": 0, "tokenizer": {"eos": tok.eos_token, "eos_id": tok.eos_token_id, "pad": tok.pad_token,
                                 "pad_id": tok.pad_token_id, "bos": tok.bos_token, "pad_is_eos": tok.pad_token_id == tok.eos_token_id}}
    soup_fn = None
    try:
        from soup_cli.data.loss_mask import build_assistant_only_labels as soup_fn  # noqa: N813
        res["soup_mask_function"] = "soup_cli.data.loss_mask.build_assistant_only_labels"
    except Exception as exc:
        res["soup_mask_function"] = f"not importable in this Soup version: {exc!r}"[:300]
    agg = Counter()
    extra_tokens = Counter()
    for r in rows:
        en = encode_example(tok, r["messages"])
        ours = set(en["answer_idx"])
        ids = en["input_ids"]
        res["n"] += 1
        agg["ours_eos_in_answer"] += int(tok.eos_token_id in [ids[i] for i in ours])
        agg["prefix_tokens_match"] += int(en["prefix_tokens_match"])
        agg["content_exact"] += int(en["content_exact"])
        if soup_fn is None:
            continue
        try:
            out = soup_fn(r["messages"], tok, max_length=MAX_LEN)
        except Exception as exc:
            agg["soup_error"] += 1
            res["soup_error_example"] = repr(exc)[:300]
            continue
        s_ids, s_lab = list(out["input_ids"]), list(out["labels"])
        trained = {i for i, lab in enumerate(s_lab) if lab != -100}
        agg["soup_same_input_ids"] += int(s_ids == ids)
        agg["soup_eos_trained"] += int(any(s_ids[i] == tok.eos_token_id for i in trained))
        agg["soup_prompt_tokens_trained"] += int(any(i < en["n_prompt_tokens"] - len(header) for i in trained))
        agg["soup_covers_all_our_answer_tokens"] += int(ours <= trained)
        for i in trained - ours:
            if i < len(s_ids):
                extra_tokens[tok.decode([s_ids[i]])] += 1
    res["counts"] = dict(agg)
    res["soup_trains_extra_tokens"] = dict(extra_tokens.most_common(10))
    ex = rows[0]["messages"]
    en = encode_example(tok, ex)
    full = tok.apply_chat_template(ex, tokenize=False)
    a = en["answer_idx"]
    res["example_render"] = (tok.decode(en["input_ids"][: a[0]]) + "[[TRAINED>>" + tok.decode(en["input_ids"][a[0]: a[-1] + 1])
                             + "<<TRAINED]]" + tok.decode(en["input_ids"][a[-1] + 1:]))[:3000]
    res["full_render_sha"] = hashlib.sha256(full.encode()).hexdigest()[:16]
    write_json(OUT / "template_check.json", res)
    return res


# ======================================================================================
# Soup config
# ======================================================================================
def _yaml_scalar(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int,)) and not isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e6):
            return f"{v:.3e}"          # e.g. 1.000e-09: a float for both YAML 1.1 and 1.2
        return repr(v)
    if v is None:
        return "null"
    s = str(v)
    if re.fullmatch(r"[A-Za-z0-9_./\-]+", s) and s.lower() not in ("yes", "no", "on", "off", "true", "false", "null"):
        return s
    return json.dumps(s, ensure_ascii=False)


def _yaml_lines(d, indent=0):
    out = []
    for k, v in d.items():
        pad = "  " * indent
        if isinstance(v, dict):
            out.append(f"{pad}{k}:")
            out += _yaml_lines(v, indent + 1)
        elif isinstance(v, list):
            out.append(f"{pad}{k}: [" + ", ".join(_yaml_scalar(x) for x in v) + "]")
        else:
            out.append(f"{pad}{k}: {_yaml_scalar(v)}")
    return out


def soup_config_dict(hp, train_file, output_dir, minimal=False) -> dict:
    data = {"train": rel(train_file), "format": "chatml", "max_length": hp["max_length"], "val_split": 0.0}
    training = {"epochs": hp["epochs"], "lr": float(hp["lr"]), "batch_size": hp["batch_size"],
                "gradient_accumulation_steps": hp["grad_accum"], "quantization": "none",
                "gradient_checkpointing": True}
    lora = {"r": hp["lora_r"], "alpha": hp["lora_alpha"], "dropout": hp["lora_dropout"],
            "target_modules": list(hp["target_modules"])}
    if not minimal:
        data["train_on_responses_only"] = True
        training.update({"seed": SEED, "warmup_ratio": hp["warmup_ratio"], "weight_decay": hp["weight_decay"],
                         "max_grad_norm": hp["max_grad_norm"], "scheduler": hp["scheduler"],
                         "optimizer": "adamw_torch", "logging_steps": hp["logging_steps"],
                         "save_steps": hp["save_steps"]})
    training["lora"] = lora
    cfg = {"base": MODEL_ID, "task": "sft"}
    if not minimal:
        cfg["backend"] = "transformers"
    cfg.update({"data": data, "training": training, "output": rel(output_dir)})
    return cfg


def write_soup_config(path, cfg: dict, header=""):
    text = (f"# {header}\n" if header else "") + "\n".join(_yaml_lines(cfg)) + "\n"
    Path(path).write_text(text, encoding="utf-8")
    log.info(f"wrote {rel(path)}")
    return Path(path)


# ======================================================================================
# training artefacts
# ======================================================================================
def find_adapter(root) -> Path | None:
    root = Path(root)
    if not root.exists():
        return None
    if (root / "adapter_config.json").exists() and (root / "adapter_model.safetensors").exists():
        return root

    def stepnum(p):
        tail = p.name.rsplit("-", 1)[-1]
        return int(tail) if tail.isdigit() else -1

    for c in sorted(root.glob("checkpoint-*"), key=stepnum, reverse=True):
        if (c / "adapter_config.json").exists() and (c / "adapter_model.safetensors").exists():
            return c
    for p in root.rglob("adapter_config.json"):
        if (p.parent / "adapter_model.safetensors").exists():
            return p.parent
    return None


def training_history(root) -> dict:
    """Read the HF trainer_state.json with the most steps (Soup) or fallback trainer_log.json."""
    root = Path(root)
    best, best_step = None, -1
    for f in root.rglob("trainer_state.json"):
        st = read_json(f, {}) or {}
        if st.get("global_step", -1) > best_step:
            best, best_step = (f, st), st.get("global_step", -1)
    if best:
        f, st = best
        return {"source": rel(f), "global_step": st.get("global_step"), "max_steps": st.get("max_steps"),
                "num_train_epochs": st.get("num_train_epochs"), "train_batch_size": st.get("train_batch_size"),
                "log_history": st.get("log_history", [])}
    fb = root / "trainer_log.json"
    if fb.exists():
        d = read_json(fb, {})
        return {"source": rel(fb), **d}
    return {"source": None, "log_history": []}


def summarize_history(h: dict) -> dict:
    lh = [x for x in h.get("log_history", []) if "loss" in x]
    losses = [x["loss"] for x in lh]
    gns = [x.get("grad_norm") for x in lh if x.get("grad_norm") is not None]
    def _finite(x):
        vals = [x.get("loss"), x.get("grad_norm")]
        return all((v is None) or (isinstance(v, (int, float)) and math.isfinite(v)) for v in vals)

    bad = [x for x in lh if not _finite(x)]
    return {"source": h.get("source"), "global_step": h.get("global_step"), "max_steps": h.get("max_steps"),
            "n_logged": len(lh), "first_loss": losses[0] if losses else None, "last_loss": losses[-1] if losses else None,
            "min_loss": min(losses) if losses else None, "max_grad_norm": max(gns) if gns else None,
            "nonfinite_entries": len(bad), "train_runtime": next((x.get("train_runtime") for x in h.get("log_history", [])
                                                                   if "train_runtime" in x), None),
            "skipped_steps_fp16": h.get("skipped_steps")}


def plot_history(h: dict, out_png):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        log.warning(f"matplotlib unavailable: {exc!r}")
        return None
    lh = [x for x in h.get("log_history", []) if "loss" in x]
    if not lh:
        return None
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
    st = [x.get("step") for x in lh]
    ax[0].plot(st, [x["loss"] for x in lh], marker=".")
    ax[0].set_title("train loss (as logged)")
    ax[1].plot(st, [x.get("grad_norm") for x in lh], marker=".")
    ax[1].set_title("grad norm")
    ax[2].plot(st, [x.get("learning_rate") for x in lh], marker=".")
    ax[2].set_title("learning rate")
    for a in ax:
        a.set_xlabel("optimizer step")
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return out_png


def grep_log(name, patterns=("warn", "error", "trunc", "skip", "drop", "fp16", "bf16", "trainable", "checkpoint",
                             "nan", "inf", "overflow", "lora", "target", "quantiz", "precision", "dtype", "template",
                             "eos", "pad")) -> list:
    txt = cmd_output(name)
    rx = re.compile("|".join(patterns), re.I)
    return [ln for ln in txt.splitlines() if rx.search(ln)][:300]


# ======================================================================================
# fallback stack: transformers + peft + bitsandbytes (only if Soup fails)
# ======================================================================================
def _fallback_train_main(argv):
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hp", required=True, help="JSON")
    a = ap.parse_args(argv)
    hp = json.loads(a.hp)
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
    from verify_trained import dtype_kw, encode_example

    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ex = []
    for r in load_jsonl(a.train):
        en = encode_example(tok, r["messages"], max_length=hp["max_length"])
        if en["truncated"] or not en["answer_idx"]:
            continue
        labels = [-100] * len(en["input_ids"])
        for i in en["answer_idx"]:
            labels[i] = en["input_ids"][i]
        ex.append((en["input_ids"], labels))
    print(f"[fallback] {len(ex)} training rows", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map={"": 0}, **dtype_kw(torch.float16))
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    lcfg = LoraConfig(r=hp["lora_r"], lora_alpha=hp["lora_alpha"], lora_dropout=hp["lora_dropout"],
                      target_modules=hp["target_modules"], bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    params = [p for p in model.parameters() if p.requires_grad]
    n_tr = sum(p.numel() for p in params)
    print(f"[fallback] trainable params {n_tr:,}", flush=True)
    opt = torch.optim.AdamW(params, lr=hp["lr"], weight_decay=hp["weight_decay"])
    bs, ga, ep = hp["batch_size"], hp["grad_accum"], hp["epochs"]
    steps_per_epoch = math.ceil(math.ceil(len(ex) / bs) / ga)
    total = steps_per_epoch * ep
    sched = get_cosine_schedule_with_warmup(opt, int(total * hp["warmup_ratio"]), total)
    scaler = torch.amp.GradScaler("cuda")
    model.train()
    hist, step, skipped, t0, tokens = [], 0, 0, time.time(), 0
    for e in range(ep):
        order = list(range(len(ex)))
        rng.shuffle(order)
        batches = [order[i:i + bs] for i in range(0, len(order), bs)]
        for w0 in range(0, len(batches), ga):
            window = batches[w0:w0 + ga]
            n_win = sum(sum(1 for t in ex[i][1][1:] if t != -100) for b in window for i in b)
            loss_win = 0.0
            for b in window:
                T = max(len(ex[i][0]) for i in b)
                ids = torch.full((len(b), T), pad, dtype=torch.long)
                lab = torch.full((len(b), T), -100, dtype=torch.long)
                am = torch.zeros((len(b), T), dtype=torch.long)
                for j, i in enumerate(b):
                    n = len(ex[i][0])
                    ids[j, :n] = torch.tensor(ex[i][0])
                    lab[j, :n] = torch.tensor(ex[i][1])
                    am[j, :n] = 1
                ids, lab, am = ids.cuda(), lab.cuda(), am.cuda()
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(input_ids=ids, attention_mask=am).logits
                loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), lab[:, 1:].reshape(-1),
                                       ignore_index=-100, reduction="sum") / max(n_win, 1)
                scaler.scale(loss).backward()
                loss_win += float(loss)
                tokens += int(am.sum())
                del logits, loss
            scaler.unscale_(opt)
            gn = float(torch.nn.utils.clip_grad_norm_(params, hp["max_grad_norm"]))
            s_before = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            if scaler.get_scale() < s_before:
                skipped += 1
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
            rec = {"step": step, "epoch": e + (w0 + len(window)) / len(batches), "loss": loss_win, "grad_norm": gn,
                   "learning_rate": sched.get_last_lr()[0], "grad_scale": scaler.get_scale(), "skipped_so_far": skipped,
                   "tokens_per_s": tokens / (time.time() - t0), "time": now_iso()}
            hist.append(rec)
            if step % hp["logging_steps"] == 0 or step == total:
                print(f"[fallback] {json.dumps(rec)}", flush=True)
    model.save_pretrained(str(out))
    tok.save_pretrained(str(out))
    write_json(out / "trainer_log.json", {"log_history": hist, "global_step": step, "max_steps": total,
                                          "skipped_steps": skipped, "trainable_params": n_tr, "hp": hp,
                                          "train_runtime": time.time() - t0})
    print(f"[fallback] done: {step} steps, {skipped} fp16-skipped steps, saved to {out}", flush=True)


def _fallback_merge_main(argv):
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from verify_trained import dtype_kw

    dm = {"": "cpu"} if a.device == "cpu" else {"": 0}
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map=dm, **dtype_kw(torch.float16))
    m = PeftModel.from_pretrained(base, a.adapter).merge_and_unload()
    m.save_pretrained(a.out)
    AutoTokenizer.from_pretrained(a.adapter if (Path(a.adapter) / "tokenizer_config.json").exists() else MODEL_ID).save_pretrained(a.out)
    print(f"[fallback-merge] saved {a.out}", flush=True)


def _fallback_quantize_main(argv):
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from verify_trained import dtype_kw

    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                           bnb_4bit_compute_dtype=torch.float16)
    m = AutoModelForCausalLM.from_pretrained(a.merged, quantization_config=q, device_map={"": 0}, **dtype_kw(torch.float16))
    m.save_pretrained(a.out)
    AutoTokenizer.from_pretrained(a.merged).save_pretrained(a.out)
    print(f"[fallback-quantize] saved {a.out}", flush=True)


def fallback_cmd(sub, *args):
    return [sys.executable, str(WORK / "kit_common.py"), sub, *[str(x) for x in args]]


# ======================================================================================
# fault injection: do Soup's own checks notice deliberately broken inputs? (CPU only)
# ======================================================================================
FAULT_KEYWORDS = ["warn", "major", "minor", "error", "invalid", "dup", "trunc", "leak", "contamin", "language",
                  "refus", "empty", "role", "unknown", "4bit", "qlora", "nf4", "quantiz", "target", "lr", "length",
                  "exceed", "skip", "drop"]


def fault_injection(timeout_s=300) -> list:
    caps = soup_caps()
    fdir = DATA / "faults"
    fdir.mkdir(parents=True, exist_ok=True)
    train = load_jsonl(DATA / "train.jsonl")
    ev = load_jsonl(DATA / "eval.jsonl")
    base = train[:200]
    u = lambda r: r["messages"][0]  # noqa: E731
    sets = {
        "clean_200": base,
        "dup_25pct": base + base[:50],
        "leak_eval_50": base[:150] + [{"messages": e["messages"]} for e in ev[:50]],
        "too_long_answers": base[:150] + [{"messages": [u(r), {"role": "assistant", "content": r["messages"][1]["content"] * 15}]}
                                          for r in base[150:200]],
        "empty_answers": base[:180] + [{"messages": [u(r), {"role": "assistant", "content": ""}]} for r in base[180:200]],
        "wrong_roles": base[:180] + [{"messages": [{"role": "human", "content": r["messages"][0]["content"]},
                                                   {"role": "gpt", "content": r["messages"][1]["content"]}]} for r in base[180:200]],
        "english_answers": base[:150] + [{"messages": [u(r), {"role": "assistant",
                                                              "content": "Sorry, I can only answer in English. Please ask again."}]}
                                         for r in base[150:200]],
    }
    for k, rows in sets.items():
        write_jsonl(fdir / f"{k}.jsonl", rows)
    jobs = []
    for k in sets:
        jobs.append((k, "validate", ["soup", "data", "validate", rel(fdir / f"{k}.jsonl")]))
    for k in ("clean_200", "dup_25pct", "leak_eval_50", "english_answers"):
        jobs.append((k, "stats", ["soup", "data", "stats", rel(fdir / f"{k}.jsonl")]))
    if caps.get("data_has_doctor"):
        for k in ("clean_200", "too_long_answers", "empty_answers", "wrong_roles"):
            jobs.append((k, "doctor", ["soup", "data", "doctor", rel(fdir / f"{k}.jsonl"), "--model", MODEL_ID,
                                       "--max-length", str(MAX_LEN)]))
    cfgs = {}
    good = soup_config_dict(HP, fdir / "clean_200.jsonl", OUT / "fault_dryrun_out")
    cfgs["cfg_clean"] = good
    for k in ("too_long_answers", "dup_25pct", "empty_answers"):
        c = json.loads(json.dumps(good))
        c["data"]["train"] = rel(fdir / f"{k}.jsonl")
        cfgs[f"cfg_data_{k}"] = c
    c = json.loads(json.dumps(good)); c["training"].pop("quantization"); cfgs["cfg_quantization_omitted"] = c
    c = json.loads(json.dumps(good)); c["training"]["lora"]["target_modules"] = "auto"; cfgs["cfg_target_auto"] = c
    c = json.loads(json.dumps(good)); c["training"]["lr"] = 0.0; cfgs["cfg_lr_zero"] = c
    c = json.loads(json.dumps(good)); c["training"]["quantizaton"] = c["training"].pop("quantization"); cfgs["cfg_typo_key"] = c
    c = json.loads(json.dumps(good)); c["data"]["max_length"] = 128; cfgs["cfg_maxlen_128"] = c
    c = json.loads(json.dumps(good))
    for key in ("lr",):
        c["training"].pop(key)
    c["training"]["lora"] = {"target_modules": TARGET_MODULES}
    cfgs["cfg_defaults_lr_r_alpha"] = c
    if caps.get("train_has_dry_run"):
        for k, c in cfgs.items():
            p = write_soup_config(fdir / f"{k}.yaml", c, header=f"fault-injection config {k}")
            jobs.append((k, "train_dry_run", ["soup"] + soup_train_args(p, dry_run=True)))
    rows = []
    for case, what, args in jobs:
        name = f"fault_{case}_{what}"
        rc, out = run_cmd(name, args, timeout_s=timeout_s, quiet=True)
        # judge only what Soup printed: drop the echoed command, file paths and the case name itself
        body = "\n".join(ln for ln in out.splitlines() if not ln.lstrip().startswith("$ ")
                         and not ln.startswith("exit code") and "Loading config from" not in ln)
        body = body.replace(case, "<case>").replace("data/faults/", "")
        low = body.lower()
        hits = sorted({kw for kw in FAULT_KEYWORDS if kw in low})
        flagged = [re.sub(r"\s{3,}", "  ", ln.strip(" │|"))[:220] for ln in body.splitlines()
                   if re.search(r"warn|major|minor|error|invalid|refus|trunc|dup|rows valid|data ok|quant:|lora:|overall|"
                                r"validation|greater than|unknown", ln, re.I)]

        def _prio(ln):
            l_ = ln.lower()
            return (0 if "major" in l_ else 1 if "overall" in l_ or "would be truncated" in l_ else
                    3 if "generation_markers" in l_ else 2)

        flagged = sorted(flagged, key=_prio)[:10]
        rows.append({"case": case, "check": what, "rc": rc, "keywords": hits, "flagged_lines": flagged,
                     "log": rel(_cmd_paths(name)[0])})
        log.info(f"fault {case:<26} {what:<14} rc={rc} keywords={hits[:8]}")
    write_json(OUT / "fault_injection.json", rows)
    md = ["| case | Soup check | exit code | what Soup printed (verdict lines; read the log to judge) | log |",
          "|---|---|---|---|---|"]
    for r in rows:
        said = " / ".join(x.replace("|", "/") for x in r["flagged_lines"][:4]) or "(nothing flagged)"
        md.append(f"| {r['case']} | {r['check']} | {r['rc']} | {said[:400]} | {r['log']} |")
    (OUT / "fault_injection.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return rows


# ======================================================================================
# static artifact checks (CPU) -> outputs/silent_checks.json
# ======================================================================================
def _base_snapshot_dir():
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(MODEL_ID, allow_patterns=["*.json", "*.txt", "*.jinja"]))
    except Exception:
        return None


def _artifact_meta(d):
    d = Path(d) if d else None
    if not d or not d.exists():
        return None
    cfg = read_json(d / "config.json", {}) or {}
    tcfg = read_json(d / "tokenizer_config.json", {}) or {}
    tmpl = tcfg.get("chat_template")
    if not tmpl and (d / "chat_template.jinja").exists():
        tmpl = (d / "chat_template.jinja").read_text(encoding="utf-8")
    return {
        "dir": rel(d),
        "config": {k: cfg.get(k) for k in ("architectures", "torch_dtype", "dtype", "tie_word_embeddings", "vocab_size",
                                            "max_position_embeddings", "rope_theta", "use_cache", "eos_token_id",
                                            "bos_token_id", "pad_token_id")},
        "quantization_config": cfg.get("quantization_config"),
        "generation_config": read_json(d / "generation_config.json"),
        "adapter_config": read_json(d / "adapter_config.json"),
        "tokenizer_json_sha": sha256_file(d / "tokenizer.json"),
        "chat_template_sha": hashlib.sha256(tmpl.encode()).hexdigest()[:16] if isinstance(tmpl, str) else None,
        "eos_token": tcfg.get("eos_token"), "pad_token": tcfg.get("pad_token"),
        "files": sorted(p.name for p in d.iterdir())[:60],
        "weights_gib": sum(p.stat().st_size for p in d.glob("*.safetensors")) / 2**30,
    }


def static_artifact_checks() -> dict:
    P = paths()
    arts = {"base": _base_snapshot_dir(), "adapter": P.get("adapter"), "merged": P.get("merged"),
            "merged_4bit": P.get("merged_4bit"), "nc_adapter": P.get("nc_adapter"), "nc_merged": P.get("nc_merged")}
    meta = {k: _artifact_meta(v) for k, v in arts.items()}
    checks = []

    def add(cid, status, what, value):
        checks.append({"id": cid, "status": status, "what": what, "value": value})

    b = meta.get("base") or {}
    for k in ("adapter", "merged", "merged_4bit"):
        m = meta.get(k)
        if not m:
            add(f"S-{k}-exists", "MISSING", f"{k} artifact present", None)
            continue
        if b:
            same_t = m["chat_template_sha"] == b.get("chat_template_sha")
            add(f"S-{k}-template", "OK" if same_t else "DIFF", f"{k} chat template == base",
                {"artifact": m["chat_template_sha"], "base": b.get("chat_template_sha")})
            add(f"S-{k}-tokenizer", "OK" if m["tokenizer_json_sha"] == b.get("tokenizer_json_sha") else "DIFF",
                f"{k} tokenizer.json == base", {"artifact": m["tokenizer_json_sha"], "base": b.get("tokenizer_json_sha")})
            add(f"S-{k}-eos-pad", "OK" if (m["eos_token"], m["pad_token"]) == (b.get("eos_token"), b.get("pad_token")) else "DIFF",
                f"{k} eos/pad tokens == base", {"artifact": [m["eos_token"], m["pad_token"]], "base": [b.get("eos_token"), b.get("pad_token")]})
            if k != "adapter":
                strip = lambda g: {kk: vv for kk, vv in (g or {}).items() if kk != "transformers_version"}  # noqa: E731
                add(f"S-{k}-genconfig", "OK" if strip(m["generation_config"]) == strip(b.get("generation_config")) else "DIFF",
                    f"{k} generation_config.json == base (users get these defaults)",
                    {"artifact": m["generation_config"], "base": b.get("generation_config")})
    ac = (meta.get("adapter") or {}).get("adapter_config") or {}
    if ac:
        tm = ac.get("target_modules")
        tm_set = set(tm) if isinstance(tm, list) else {tm}
        add("S-adapter-targets", "OK" if tm_set == set(TARGET_MODULES) else "DIFF", "adapter target_modules == requested",
            {"adapter": sorted(map(str, tm_set)), "requested": TARGET_MODULES})
        add("S-adapter-rank", "OK" if (ac.get("r"), ac.get("lora_alpha")) == (HP["lora_r"], HP["lora_alpha"]) else "DIFF",
            "adapter r/alpha == requested", {"r": ac.get("r"), "alpha": ac.get("lora_alpha")})
        add("S-adapter-base", "OK" if ac.get("base_model_name_or_path") == MODEL_ID else "DIFF",
            "adapter base_model_name_or_path == base we evaluate", ac.get("base_model_name_or_path"))
        add("S-adapter-extras", "OK" if not ac.get("modules_to_save") and not ac.get("use_dora") else "DIFF",
            "no modules_to_save / DoRA surprises", {"modules_to_save": ac.get("modules_to_save"), "use_dora": ac.get("use_dora")})
    q = (meta.get("merged_4bit") or {}).get("quantization_config")
    if q is not None:
        add("S-4bit-config", "OK" if q.get("bnb_4bit_quant_type") == "nf4" else "DIFF", "4-bit quantization_config", q)
    mcfg = (meta.get("merged") or {}).get("config") or {}
    if meta.get("merged"):
        add("S-merged-dtype", "OK" if str(mcfg.get("torch_dtype") or mcfg.get("dtype")) in ("float16", "torch.float16") else "DIFF",
            "merged checkpoint dtype is float16", mcfg.get("torch_dtype") or mcfg.get("dtype"))
    # training run consistency
    h = training_history(P.get("train_root") or SOUP_RUN)
    s = summarize_history(h)
    exp_steps = math.ceil(math.ceil(N_TRAIN / HP["batch_size"]) / HP["grad_accum"]) * HP["epochs"]
    gs = s.get("global_step")
    add("S-train-steps", "OK" if isinstance(gs, int) and abs(gs - exp_steps) <= HP["epochs"] else "DIFF",
        "optimizer steps ~= ceil(ceil(N/bs)/ga)*epochs, +-1 per epoch for floor/ceil (else rows were dropped)",
        {"expected": exp_steps, "logged_global_step": s.get("global_step"), "max_steps": s.get("max_steps")})
    add("S-train-finite", "OK" if s.get("nonfinite_entries") == 0 and s.get("n_logged") else "CHECK",
        "all logged loss/grad_norm finite", {"nonfinite_entries": s.get("nonfinite_entries"), "n_logged": s.get("n_logged")})
    out = {"created": now_iso(), "artifacts": meta, "checks": checks, "training_summary": s,
           "soup_train_log_flags": grep_log(P.get("train_cmd_name") or "soup_train")[:120]}
    write_json(OUT / "silent_checks.json", out)
    return out


# ======================================================================================
# measured memory, report numbers, bundle
# ======================================================================================
STAGE_FOR_COMPARE = {"train": ["soup_train", "fallback_train"], "merge": ["soup_merge", "fallback_merge"],
                     "quantize_4bit": ["soup_quantize_4bit", "fallback_quantize"], "verify": ["verify"]}


def collect_measured() -> dict:
    recs = stage_records()
    meas = {}
    for key, names in STAGE_FOR_COMPARE.items():
        cand = [r for r in recs if r.get("stage") in names and r.get("rc") in (0, 2)]  # 2 = negative verdict, not an error
        if not cand:
            continue
        r = cand[-1]
        pk = r.get("subprocess_peak") or {}
        meas[key] = {"source": r.get("stage"), "seconds": r.get("seconds"),
                     "torch_max_allocated_gib": pk.get("torch_max_allocated_gib"),
                     "torch_max_reserved_gib": pk.get("torch_max_reserved_gib"),
                     "cuda_initialized": pk.get("cuda_initialized"),
                     "ru_maxrss_gib": pk.get("ru_maxrss_gib"),
                     "nvsmi_max_used_gib": r.get("nvsmi_max_used_gib"),
                     "nvsmi_baseline_gib": r.get("nvsmi_baseline_gib"),
                     "nvsmi_samples": r.get("nvsmi_samples")}
    write_json(OUT / "measured_memory.json", meas)
    return meas


CUDA_CTX_CODE = r"""
import json, time, torch
torch.cuda.init()
a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
b = a @ a                     # forces cuBLAS handle + workspace
torch.cuda.synchronize()
print(json.dumps({"torch_allocated_gib": torch.cuda.memory_allocated() / 2**30,
                  "torch_reserved_gib": torch.cuda.memory_reserved() / 2**30}), flush=True)
time.sleep(6)                 # let the 1 s nvidia-smi sampler see it
"""


def measure_cuda_context() -> dict:
    """Calibrate the memory torch cannot see: nvidia-smi delta of a process holding ~4 MiB of tensors."""
    rc, out = run_cmd("cuda_context_probe", [sys.executable, "-c", CUDA_CTX_CODE], timeout_s=120, quiet=True)
    rec = next((r for r in reversed(stage_records()) if r.get("stage") == "cuda_context_probe"), {})
    res = {"rc": rc, "nvsmi_max_used_gib": rec.get("nvsmi_max_used_gib"), "nvsmi_baseline_gib": rec.get("nvsmi_baseline_gib")}
    m = re.search(r"\{.*\}", out or "")
    if m:
        res.update(json.loads(m.group(0)))
    if res.get("nvsmi_max_used_gib") is not None and res.get("nvsmi_baseline_gib") is not None:
        res["context_overhead_gib"] = res["nvsmi_max_used_gib"] - res["nvsmi_baseline_gib"] - res.get("torch_reserved_gib", 0.0)
    write_json(OUT / "cuda_context.json", res)
    log.info(f"CUDA context calibration: {res}")
    return res


def _g(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def report_numbers() -> Path:
    """Collect key numbers FROM RUN OUTPUTS into outputs/report_numbers.md (no number is typed by hand)."""
    V = read_json(VERIFY_DIR / "verify_results.json", {}) or {}
    V2 = read_json(VERIFY_NC2_DIR / "verify_results.json", {}) or {}
    MB = read_json(OUT / "memory_budget.json", {}) or {}
    MC = read_json(OUT / "memory_compare.json", []) or []
    DR = read_json(DATA / "data_report.json", {}) or {}
    SC = read_json(OUT / "silent_checks.json", {}) or {}
    SH = read_json(OUT / "ship_verdict.json")
    SHO = read_json(OUT / "ship_verdict_offline.json")
    caps = read_json(OUT / "soup_capabilities.json", {}) or {}
    P = paths()
    NA = "<<NOT AVAILABLE: missing in run outputs>>"

    def v(x, fmt="{}"):
        if x is None:
            return NA
        try:
            return fmt.format(x)
        except Exception:
            return str(x)

    L = ["# Numbers for the report (auto-collected from run outputs)", "",
         f"Generated {now_iso()} from files under {WORK}. Copy values into report_template.md; "
         "anything marked NOT AVAILABLE did not run or failed: say so in the report.", ""]
    L += ["## Setup", f"- soup version: {v(caps.get('version'))}; train backend used: {v(P.get('train_backend'))}",
          f"- versions (verify): {v(V.get('versions'))}",
          f"- dataset {v(DR.get('dataset'))}@{v(DR.get('dataset_revision'))}: raw {v(DR.get('n_raw'))}, after filters "
          f"{v(DR.get('n_pool_after_filters'))}, train {v(DR.get('n_train'))}, eval {v(DR.get('n_eval'))}",
          f"- drop reasons: {v(DR.get('drop_reasons'))}", f"- leakage audit: {v(DR.get('leakage_audit'))}",
          f"- train token length (total): {v(_g(DR, 'token_stats', 'train_total'))}", ""]
    L += ["## Memory"]
    for name, t in (MB.get("train") or {}).items():
        L.append(f"- estimate train [{name}] T={t.get('seq_len')}: alloc {v(t.get('est_torch_allocated_gib'), '{:.2f}')} GiB, "
                 f"nvidia-smi {v(t.get('est_nvidia_smi_gib'), '{:.2f}')} GiB, logits bwd {v(t.get('logits_bwd_gib'), '{:.2f}')} GiB, peak set by {t.get('peak_component')}")
    L.append(f"- estimate 4-bit model {v(_g(MB, 'quantize_4bit', 'final_model_gib'), '{:.2f}')} GiB; merge RAM "
             f"{v(_g(MB, 'merge', 'ram_low_gib'), '{:.2f}')}-{v(_g(MB, 'merge', 'ram_high_gib'), '{:.2f}')} GiB")
    for r in MC:
        L.append(f"- measured {r.get('stage')}: torch alloc {v(r.get('meas_alloc_gib'), '{:.2f}')}, reserved "
                 f"{v(r.get('meas_reserved_gib'), '{:.2f}')}, nvidia-smi max {v(r.get('meas_nvsmi_gib'), '{:.2f}')}, "
                 f"gap {v(r.get('gap_alloc_pct'), '{:+.0f}')}%  hints: {r.get('hints')}")
    L += ["", "## Proof of training (verify_trained.py)"]
    for n, s in (V.get("heldout") or {}).items():
        L.append(f"- NLL {n}: {v(s.get('nll_per_token'), '{:.4f}')} (ci95 {v(s.get('ci95'))}), ppl {v(s.get('ppl'), '{:.3f}')}, nonfinite {s.get('nonfinite_logits')}")
    for k, d in (V.get("heldout_diffs") or {}).items():
        L.append(f"- diff {k}: {v(d.get('point'), '{:+.4f}')} ci95 {v(d.get('ci95'))}")
    for k, d in (V.get("kl") or {}).items():
        L.append(f"- KL {k}: {v(d.get('kl_mean'), '{:.5f}')}, top-1 {v(d.get('top1_agreement'), '{:.4f}')}")
    L.append(f"- adapter B norms: {v(_g(V, 'adapter_main', 'B_fro'))}; control: {v(_g(V, 'adapter_control', 'B_fro'))}")
    L.append(f"- adapter rel dW: {v(_g(V, 'adapter_main', 'rel_dW'))}; fp16 rounding loss: {v(_g(V, 'adapter_main', 'fp16_rounding_loss'))}")
    L.append(f"- 4-bit codes: {v(V.get('quant4_codes', {}).get('frac_changed_merged_vs_base') if V.get('quant4_codes') else None)}")
    L.append(f"- generalisation gap: {v(V.get('generalisation_gap'))}")
    L.append(f"- generations: {v(V.get('generations'))}")
    L.append(f"- VERDICT main: {v(V.get('verdict'))}")
    L.append(f"- VERDICT NC2 (merged := base): {v(V2.get('verdict'))}")
    L += ["", "## Checks table (main run)"]
    for c in V.get("checks", []):
        L.append(f"- [{c['status']}] {c['id']} {c['name']}: {json.dumps(c['value'], ensure_ascii=False)[:300]}")
    L += ["", "## Static artifact checks"]
    for c in SC.get("checks", []):
        L.append(f"- [{c['status']}] {c['id']} {c['what']}: {json.dumps(c['value'], ensure_ascii=False, default=str)[:300]}")
    L.append(f"- training summary: {v(SC.get('training_summary'))}")
    L += ["", "## soup ship", f"- live verdict file: {v(SH)}"[:3000], f"- offline (our NLL metric) verdict: {v(SHO)}"[:2000]]
    p = OUT / "report_numbers.md"
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    return p


BUNDLE_EXCLUDE_SUFFIX = {".safetensors", ".bin", ".pt", ".pth", ".gguf", ".ckpt", ".tmp"}
BUNDLE_EXCLUDE_NAMES = {"optimizer.pt", "scheduler.pt", "rng_state.pth"}


def bundle() -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    zp = WORK / f"soup_task_results_{stamp}.zip"
    n = 0
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for root in (LOGS, OUT, DATA):
            for f in root.rglob("*"):
                if not f.is_file() or f.suffix in BUNDLE_EXCLUDE_SUFFIX or f.name in BUNDLE_EXCLUDE_NAMES:
                    continue
                if f.stat().st_size > 50 * 2**20:
                    continue
                z.write(f, f.relative_to(WORK))
                n += 1
        for f in list(WORK.glob("*.py")) + list(WORK.glob("*.md")) + list(WORK.glob("*.yaml")) + list(WORK.glob("*.ipynb")):
            z.write(f, f.relative_to(WORK))
            n += 1
    log.info(f"bundle {rel(zp)}: {n} files, {zp.stat().st_size / 2**20:.1f} MiB")
    return zp


# ======================================================================================
# CLI
# ======================================================================================
def _main(argv):
    if not argv:
        print(__doc__)
        return 0
    sub, rest = argv[0], argv[1:]
    if sub == "bg-runner":
        return _bg_runner_main(rest)
    if sub == "peak":
        return _peak_main(rest)
    if sub == "fallback-train":
        return _fallback_train_main(rest)
    if sub == "fallback-merge":
        return _fallback_merge_main(rest)
    if sub == "fallback-quantize":
        return _fallback_quantize_main(rest)
    print(f"unknown subcommand {sub!r}")
    return 2


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]) or 0)
