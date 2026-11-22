"""Background jobs for the long tools (analyze, review, render).

Over MCP a single call that runs for many minutes is fragile: the client may time
out or the connection may drop although the work goes on. So a long tool starts a
job, waits a little (short work still answers in one call), and otherwise returns a
``job_id`` at once; the agent polls ``job_status``.

A job is a file, ``<ws>/jobs/<job_id>.json`` (spec, status, result), plus a log
``<job_id>.log``. In ``process`` mode it runs in its own Python process, so a crash in
ffmpeg, Whisper or a detector cannot take the tool server down, and a job survives a
server restart. ``thread`` mode (tests, injected fakes) runs it in this process.
One long job per workspace runs at a time: they share the same checkpoints.
"""

from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

LONG_TOOLS = ("analyze", "review", "render")
MAX_WAIT = 55.0  # seconds a single call may block (well under client timeouts)
_PKG_ROOT = Path(__file__).resolve().parents[2]  # the repository / site-packages dir


def jobs_dir(workspace: Path) -> Path:
    d = Path(workspace) / "jobs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _path(workspace: Path, job_id: str) -> Path:
    if not job_id or "/" in job_id or "\\" in job_id or job_id.startswith("."):
        raise ValueError(f"bad job_id {job_id!r}")
    return jobs_dir(workspace) / f"{job_id}.json"


def _read(p: Path) -> dict:
    return json.loads(p.read_text())


def _write(p: Path, data: dict) -> None:
    tmp = p.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(data, indent=1, default=str))
    tmp.replace(p)


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _log_tail(p: Path, lines: int = 15) -> str:
    log = p.with_suffix(".log")
    if not log.exists():
        return ""
    return "\n".join(log.read_text(errors="replace").splitlines()[-lines:])


def status(workspace: Path, job_id: str) -> dict:
    """The job record; a job whose process died without a result is reported failed."""
    p = _path(workspace, job_id)
    if not p.exists():
        raise FileNotFoundError(f"no job {job_id} in {workspace}")
    job = _read(p)
    pid = job.get("pid")
    pidfile = p.with_suffix(".pid")
    if not pid and pidfile.exists():  # written by the server when it spawned the process
        pid = int(pidfile.read_text().strip() or 0)
    if job["status"] in ("queued", "running") and pid and not _alive(pid):
        job.update(status="failed", finished_at=time.time(),
                   error="the job process exited without a result (crashed or killed)",
                   log_tail=_log_tail(p))
        _write(p, job)
    return job


def running(workspace: Path) -> list[dict]:
    out = []
    for p in sorted(jobs_dir(workspace).glob("*.json")):
        try:
            job = status(workspace, p.stem)
        except (ValueError, json.JSONDecodeError):
            continue
        if job["status"] in ("queued", "running"):
            out.append(job)
    return out


def run_job(path: Path, ctx: Any | None = None) -> None:
    """Execute the job described by ``path`` and store its result (the runner)."""
    from .tools import Context, call

    job = _read(path)
    job.update(status="running", pid=os.getpid(), started_at=time.time())
    _write(path, job)
    ctx = ctx or Context(privacy=job["privacy"],
                         home=Path(job["home"]) if job.get("home") else None)
    try:
        res = call(job["tool"], job["args"], ctx)
        job.update(status="failed" if "error" in res else "done", result=res,
                   images=list(ctx.images_out))
        if "error" in res:
            job["error"] = res["error"]
    except BaseException as e:  # noqa: BLE001 - recorded for the agent, never swallowed silently
        job.update(status="failed", error=f"{type(e).__name__}: {e}",
                   traceback=traceback.format_exc()[-4000:])
    job["finished_at"] = time.time()
    _write(path, job)


def start(tool: str, args: dict, ctx: Any, workspace: Path, wait: float) -> dict:
    """Start ``tool`` as a job; return its result if it ends within ``wait`` seconds,
    otherwise the job id to poll."""
    busy = running(workspace)
    if busy:
        b = busy[0]
        raise ValueError(f"job {b['job_id']} ({b['tool']}) is still running in this "
                         "workspace: call job_status with that job_id and wait for it")
    job_id = f"{tool}-{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"
    path = _path(workspace, job_id)
    job = {"job_id": job_id, "tool": tool, "args": args, "status": "queued",
           "privacy": ctx.privacy, "home": str(ctx.home) if ctx.home else None,
           "created_at": time.time(), "mode": ctx.jobs}
    if ctx.jobs == "thread":
        import dataclasses

        _write(path, job)
        local = dataclasses.replace(ctx, jobs="off", images_out=[])
        threading.Thread(target=run_job, args=(path, local), daemon=True,
                         name=f"llmve-{job_id}").start()
    else:
        _write(path, job)  # before the child starts: it updates this file itself
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(_PKG_ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        log = open(path.with_suffix(".log"), "ab")  # noqa: SIM115 - handed to the child
        try:
            proc = subprocess.Popen(
                [sys.executable, "-m", "llm_video_editor.agent.jobs", str(path)],
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                cwd=str(_PKG_ROOT), env=env, start_new_session=True)
        finally:
            log.close()
        path.with_suffix(".pid").write_text(str(proc.pid))
    return wait_for(workspace, job_id, wait)


def wait_for(workspace: Path, job_id: str, wait: float) -> dict:
    deadline = time.monotonic() + max(0.0, min(wait, MAX_WAIT))
    while True:
        job = status(workspace, job_id)
        if job["status"] in ("done", "failed") or time.monotonic() >= deadline:
            return job
        time.sleep(0.25)


def summary(job: dict) -> dict:
    """What a tool call returns for a job: the result when finished, else progress."""
    if job["status"] == "done":
        return {**job["result"], "job_id": job["job_id"]}
    if job["status"] == "failed":
        out = {"error": job.get("error", "job failed"), "job_id": job["job_id"]}
        if job.get("log_tail"):
            out["log_tail"] = job["log_tail"]
        return out
    started = job.get("started_at") or job["created_at"]
    return {"job_id": job["job_id"], "status": job["status"], "tool": job["tool"],
            "elapsed_seconds": round(time.time() - started),
            "next": "still working: call job_status with this job_id (it waits up to "
                    f"{int(MAX_WAIT)} s per call)"}


if __name__ == "__main__":  # the job process: python -m llm_video_editor.agent.jobs <job.json>
    run_job(Path(sys.argv[1]))
