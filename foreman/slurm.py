"""Provides additional job information for SLURM jobs.

Slurm jobs are matched to R3 jobs by their job name or by their working
directory: tools may give jobs a readable `--job-name`, but they run in the R3
job directory, whose name is the job ID.
"""

import logging
import shlex
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import PurePosixPath
from typing import Dict, List, Optional

import flask
from diskcache import Cache
from executor.ssh.client import ExternalCommandFailed, RemoteCommand

LOGGER = logging.getLogger(__name__)

CACHE = Cache()

# Host name that runs Slurm commands on this machine instead of over SSH.
LOCAL_HOST = "local"

SLURM_DATE_FORMAT = r"%Y-%m-%dT%H:%M:%S"

SACCT_FIELDS = [
    "JobID", "JobName", "State", "Submit", "Start", "End",
    "ExitCode", "NodeList", "Partition", "WorkDir",
]

# Foreman status for each Slurm job state.
STATUS_BY_SLURM_STATE = {
    "PENDING": "queued",
    "CONFIGURING": "running",
    "RUNNING": "running",
    "COMPLETING": "running",
    "SUSPENDED": "running",
    "COMPLETED": "completed",
    "FAILED": "failed",
    "BOOT_FAIL": "failed",
    "CANCELLED": "cancelled",
    "DEADLINE": "timeout",
    "TIMEOUT": "timeout",
    "NODE_FAIL": "node failure",
    "OUT_OF_MEMORY": "out of memory",
    "PREEMPTED": "preempted",
    "REQUEUED": "requeued",
}


def get_slurm():
    if "slurm" not in flask.g:
        flask.g.slurm = Slurm(
            flask.current_app.config["SLURM_HOSTS"],
            cache_experiation=flask.current_app.config["SLURM_CACHE_EXPIRATION"],
            history_days=flask.current_app.config.get("SLURM_HISTORY_DAYS", 30),
        )

    return flask.g.slurm


@dataclass(frozen=True)
class SlurmRecord:
    """One Slurm allocation of a job, as reported by sacct.

    A job that was preempted and requeued has one record per attempt.
    """

    job_id: str
    name: str
    state: str
    submit: Optional[datetime]
    start: Optional[datetime]
    end: Optional[datetime]
    exit_code: str
    nodes: str
    partition: str
    work_dir: str
    host: str

    @property
    def keys(self) -> set[str]:
        """Returns the names this record can be matched to an R3 job by."""
        return {self.name, PurePosixPath(self.work_dir).name} - {""}

    @property
    def status(self) -> str:
        """Returns the Foreman status of this record."""
        return STATUS_BY_SLURM_STATE.get(self.state, self.state.lower())

    @property
    def is_active(self) -> bool:
        """Returns whether this allocation is still queued or running."""
        return self.status in ("queued", "running")


def parse_sacct(output: str, host: str) -> List[SlurmRecord]:
    """Parses `sacct --parsable2 --noheader` output with `SACCT_FIELDS`."""
    records = []
    for line in output.splitlines():
        fields = line.split("|", maxsplit=len(SACCT_FIELDS) - 1)
        if len(fields) != len(SACCT_FIELDS):
            continue
        values = dict(zip(SACCT_FIELDS, fields, strict=True))
        records.append(SlurmRecord(
            job_id=values["JobID"],
            name=values["JobName"],
            # e.g. "CANCELLED by 1234"
            state=values["State"].split(" ", maxsplit=1)[0],
            submit=_parse_date(values["Submit"]),
            start=_parse_date(values["Start"]),
            end=_parse_date(values["End"]),
            exit_code=values["ExitCode"],
            nodes=values["NodeList"],
            partition=values["Partition"],
            work_dir=values["WorkDir"],
            host=host,
        ))
    return records


def parse_squeue(output: str) -> Dict[str, str]:
    """Parses `squeue -h -o '%j|%T|%Z'` output into job states by matching key."""
    jobs = {}
    for line in output.splitlines():
        fields = line.split("|")
        if len(fields) != 3:
            continue
        name, state, work_dir = fields
        for key in {name, PurePosixPath(work_dir).name} - {""}:
            jobs[key] = state
    return jobs


def _parse_date(value: str) -> Optional[datetime]:
    try:
        return datetime.strptime(value, SLURM_DATE_FORMAT)
    except ValueError:
        return None  # "None" or "Unknown" for jobs that did not start or end


def run_command(host: str, args: List[str]) -> Optional[str]:
    """Runs a command locally or on a remote host; returns None on failure."""
    if host == LOCAL_HOST:
        try:
            result = subprocess.run(
                args, capture_output=True, text=True, timeout=30, check=True,
            )
        except (OSError, subprocess.SubprocessError) as exception:
            LOGGER.warning("Could not run %s locally: %s", args[0], exception)
            return None
        return result.stdout

    command = RemoteCommand(host, shlex.join(args), capture=True)
    try:
        command.start()
    except ExternalCommandFailed:
        LOGGER.warning("Could not run %s on %s.", args[0], host)
        return None
    return command.output


class Slurm:
    """Provides additional information from SLURM."""

    def __init__(
        self, hosts: list[str], cache_experiation: int = 5, history_days: int = 30,
    ) -> None:
        self.hosts = hosts
        self.history_days = history_days
        self._cache_expiration = cache_experiation
        # Per-instance results, so a request asking about many jobs queries once.
        self._jobs: Optional[Dict[str, str]] = None
        self._history: Dict[date, Dict[str, List[SlurmRecord]]] = {}

        if cache_experiation > 0:
            memoize = CACHE.memoize(expire=cache_experiation)
            self._get_jobs_from_host = memoize(self._get_jobs_from_host)  # type: ignore
            self._get_history_from_host = memoize(self._get_history_from_host)  # type: ignore

    @property
    def cache_experation(self) -> int:
        return self._cache_expiration

    def get_jobs(self) -> Dict[str, str]:
        """Returns the Slurm states of queued and running jobs by matching key."""
        if self._jobs is None:
            self._jobs = {}
            for host in self.hosts:
                self._jobs.update(self._get_jobs_from_host(host))
        return self._jobs

    def get_history(self, since: Optional[date] = None) -> Dict[str, List[SlurmRecord]]:
        """Returns Slurm records submitted since a date by matching key.

        Defaults to the last `history_days` days. Records are sorted by submit time.
        """
        if since is None:
            since = date.today() - timedelta(days=self.history_days)

        if since not in self._history:
            history: Dict[str, List[SlurmRecord]] = defaultdict(list)
            for host in self.hosts:
                for record in self._get_history_from_host(host, since.isoformat()):
                    for key in record.keys:
                        history[key].append(record)
            for records in history.values():
                records.sort(key=lambda record: record.submit or datetime.min)
            self._history[since] = dict(history)

        return self._history[since]

    def get_records(
        self, job_id: str, since: Optional[date] = None,
    ) -> List[SlurmRecord]:
        """Returns the Slurm records of an R3 job, oldest first."""
        default_since = date.today() - timedelta(days=self.history_days)
        if since is not None and since > default_since:
            since = default_since  # reuse the (cached) default history
        return self.get_history(since).get(job_id, [])

    def _get_jobs_from_host(self, host: str) -> Dict[str, str]:
        """Returns a dictionary of slurm jobs from a host."""
        LOGGER.info("Getting slurm jobs from %s.", host)
        output = run_command(host, ["squeue", "--me", "--noheader", "-o", "%j|%T|%Z"])
        return parse_squeue(output) if output is not None else {}

    def _get_history_from_host(self, host: str, since: str) -> List[SlurmRecord]:
        """Returns the Slurm records of jobs submitted since a date on a host."""
        LOGGER.info("Getting slurm history since %s from %s.", since, host)
        output = run_command(host, [
            "sacct", "--allocations", "--duplicates", "--parsable2", "--noheader",
            "--starttime", since, "--format", ",".join(SACCT_FIELDS),
        ])
        return parse_sacct(output, host) if output is not None else []
