"""
Job queue blueprint.

Shows the Slurm queue (squeue): running jobs, and pending jobs with Slurm's
estimated start. The page can be customized; its settings live in foreman.conf.
"""
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from flask import Blueprint, current_app, render_template, request

from foreman import settings
from foreman.r3_extension import Repository
from foreman.settings_page import require_foreman_request, update_settings
from foreman.slurm import QUEUE_COLUMNS, QueueJob, get_slurm

job_queue = Blueprint("job_queue", __name__, url_prefix="/queue")

# Why pending jobs wait, for squeue's most common reason codes.
PENDING_REASONS = {
    "Priority": "Jobs with higher priority come first",
    "Resources": "Waiting for resources to become free",
    "Dependency": "Waiting for a job it depends on",
    "DependencyNeverSatisfied": "A job it depends on failed",
    "BeginTime": "Scheduled to start later",
    "JobHeldUser": "Held by the user",
    "JobHeldAdmin": "Held by an administrator",
    "ReqNodeNotAvail": "Requested nodes are not available",
    "PartitionDown": "The partition is down",
    "QOSMaxJobsPerUserLimit": "Limit of running jobs per user reached",
    "QOSMaxGRESPerUser": "Limit of GPUs per user reached",
    "AssocGrpGRES": "The account's GPU limit is reached",
    "AssocMaxJobsLimit": "The account's job limit is reached",
    "Reservation": "Waiting for its reservation",
}


@job_queue.route("/")
def view():
    return render_template("job_queue/view.html.jinja", **_context())


@job_queue.route("/jobs")
def jobs():
    """The queue tables alone, for refreshing them in place."""
    return render_template("job_queue/_jobs.html.jinja", **_context())


@job_queue.route("/settings", methods=["POST"])
def save_settings():
    require_foreman_request("queue")
    values, errors, status = update_settings("queue", request.form)
    return render_template(
        "job_queue/view.html.jinja", **_context(), errors=errors, saved=not errors,
    ), status


def _context() -> dict:
    config = current_app.config
    queue = get_slurm().get_queue(all_users=config.get("QUEUE_SCOPE") == "all")
    running = sorted(
        (job for job in queue if not job.is_pending),
        key=lambda job: job.start or datetime.max, reverse=True,
    )
    # Soonest estimated start first; jobs without an estimate last.
    pending = sorted(
        (job for job in queue if job.is_pending),
        key=lambda job: (job.start or datetime.max, -job.priority),
    )
    return {
        "running": running,
        "pending": pending,
        "columns": [c for c in config.get("QUEUE_COLUMNS", []) if c in QUEUE_COLUMNS],
        "all_columns": QUEUE_COLUMNS,
        "queue_settings": [s for s in settings.SETTINGS if s.page == "queue"],
        "config": config,
        "r3_jobs": _r3_jobs(queue),
        "reasons": PENDING_REASONS,
        "hosts": config.get("SLURM_HOSTS", []),
        "now": datetime.now(),
        "gpus_running": sum(job.gpus for job in running),
        "gpus_pending": sum(job.gpus for job in pending),
        "users": len({job.user for job in queue}),
    }


def _r3_jobs(queue: list[QueueJob]) -> dict[str, dict]:
    """Returns the R3 job (id and name) of each Slurm job that runs one."""
    root = os.environ.get("R3_REPOSITORY")
    if not root:
        return {}
    jobs_dir = Path(root) / "jobs"
    repository: Optional[Repository] = None
    matches = {}
    for slurm_job in queue:
        for key in slurm_job.keys:
            if not (jobs_dir / key).is_dir():
                continue
            repository = repository or Repository.from_environment()
            try:
                r3_job = repository.get(key)
            except (FileNotFoundError, ValueError):
                continue
            path = r3_job.virtual_path or ""
            # The last two folders usually tell jobs apart; the full path is a tooltip.
            short = path
            if path.count("/") > 1:
                short = "…/" + "/".join(path.split("/")[-2:])
            matches[slurm_job.job_id] = {
                "id": r3_job.id,
                "name": r3_job.name or short or r3_job.id[:8],
                "path": path or r3_job.id,
            }
            break
    return matches
