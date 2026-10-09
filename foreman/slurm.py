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


# Columns of the job queue page that can be switched on or off, in display order.
QUEUE_COLUMNS = {
    "state": "State",
    "partition": "Partition",
    "submitted": "Submitted",
    "start": "Start",
    "elapsed": "Elapsed",
    "end": "Expected end",
    "resources": "Resources",
    "nodelist": "Nodes",
    "priority": "Priority",
    "user": "User",
    "account": "Account",
    "host": "Host",
}

# squeue fields, separated by a character that job names and paths never contain.
_QUEUE_SEPARATOR = "\x1f"
_QUEUE_FIELDS = {
    "job_id": "%i", "name": "%j", "state": "%T", "reason": "%r", "partition": "%P",
    "submit": "%V", "start": "%S", "end": "%e", "elapsed": "%M",
    "time_limit": "%l", "nodes": "%D", "nodelist": "%N", "cpus": "%C",
    "tres": "%b", "priority": "%Q", "user": "%u", "account": "%a", "qos": "%q",
    "work_dir": "%Z",
}


@dataclass(frozen=True)
class QueueJob:
    """A queued or running Slurm job, as reported by squeue."""

    host: str
    job_id: str
    name: str
    state: str
    # Why a pending job waits (e.g. "Priority"), "None" otherwise.
    reason: str
    partition: str
    submit: Optional[datetime]
    # Actual start, or Slurm's estimate for a pending job (None if unknown).
    start: Optional[datetime]
    end: Optional[datetime]
    elapsed: Optional[timedelta]
    time_limit: Optional[timedelta]
    nodes: int
    nodelist: str
    cpus: int
    gpus: int
    priority: int
    user: str
    account: str
    qos: str
    work_dir: str

    @property
    def keys(self) -> set[str]:
        """Returns the names this job can be matched to an R3 job by."""
        return {self.name, PurePosixPath(self.work_dir).name} - {""}

    @property
    def is_pending(self) -> bool:
        return self.state == "PENDING"

    @property
    def status(self) -> str:
        return STATUS_BY_SLURM_STATE.get(self.state, self.state.lower())

    @property
    def progress(self) -> Optional[float]:
        """Returns the fraction of the time limit used, if there is a limit."""
        if not self.elapsed or not self.time_limit:
            return None
        return min(self.elapsed / self.time_limit, 1.0)


def parse_duration(value: str) -> Optional[timedelta]:
    """Parses a Slurm duration such as `5:47:13`, `24:55` or `3-00:00:00`."""
    days, _, clock = value.strip().rpartition("-")
    parts = clock.split(":")
    if not 1 <= len(parts) <= 3 or not all(part.isdigit() for part in parts):
        return None  # "UNLIMITED", "INVALID", "N/A", ...
    if days and not days.isdigit():
        return None
    seconds = 0
    for part in parts:
        seconds = seconds * 60 + int(part)
    return timedelta(days=int(days or 0), seconds=seconds)


def _count_gpus(tres: str, nodes: int) -> int:
    """Counts GPUs in a per-node TRES/GRES string, e.g. `gres/gpu:a100:2`."""
    count = 0
    for entry in tres.split(","):
        if "gpu" not in entry:
            continue
        last = entry.rsplit(":", 1)[-1].split("(")[0]
        count += int(last) if last.isdigit() else 1
    return count * max(nodes, 1)


def _int(value: str) -> int:
    try:
        return int(value)
    except ValueError:
        return 0


def queue_format() -> str:
    """Returns the squeue output format for `parse_queue`."""
    return _QUEUE_SEPARATOR.join(_QUEUE_FIELDS.values())


def parse_queue(output: str, host: str) -> List[QueueJob]:
    """Parses `squeue --noheader -o <queue_format()>` output."""
    jobs = []
    for line in output.splitlines():
        fields = line.split(_QUEUE_SEPARATOR)
        if len(fields) != len(_QUEUE_FIELDS):
            continue
        values = dict(zip(_QUEUE_FIELDS, fields, strict=True))
        nodes = _int(values["nodes"])
        jobs.append(QueueJob(
            host=host,
            job_id=values["job_id"],
            name=values["name"],
            state=values["state"],
            reason=values["reason"],
            partition=values["partition"],
            submit=_parse_date(values["submit"]),
            start=_parse_date(values["start"]),
            end=_parse_date(values["end"]),
            elapsed=parse_duration(values["elapsed"]),
            time_limit=parse_duration(values["time_limit"]),
            nodes=nodes,
            nodelist=values["nodelist"],
            cpus=_int(values["cpus"]),
            gpus=_count_gpus(values["tres"], nodes),
            priority=_int(values["priority"]),
            user=values["user"],
            account=values["account"],
            qos=values["qos"],
            work_dir=values["work_dir"],
        ))
    return jobs


def check_host(host: str) -> tuple[bool, str]:
    """Runs `squeue --version` on a host; returns success and a message."""
    args = ["squeue", "--version"]
    if host == LOCAL_HOST:
        try:
            result = subprocess.run(
                args, capture_output=True, text=True, timeout=30, check=True,
            )
        except subprocess.CalledProcessError as error:
            return False, (error.stderr or str(error)).strip()
        except (OSError, subprocess.SubprocessError) as error:
            return False, str(error)
        return True, result.stdout.strip()

    command = RemoteCommand(host, shlex.join(args), capture=True, capture_stderr=True)
    try:
        command.start()
    except ExternalCommandFailed as error:
        return False, (command.stderr or str(error)).strip()
    return True, command.output.strip()


def get_slurm():
    if "slurm" not in flask.g:
        flask.g.slurm = Slurm(
            flask.current_app.config["SLURM_HOSTS"],
            cache_experiation=flask.current_app.config["SLURM_CACHE_EXPIRATION"],
            history_days=flask.current_app.config.get("SLURM_HISTORY_DAYS", 30),
            status_enabled=flask.current_app.config.get("SLURM_STATUS_ENABLED", True),
            history_enabled=flask.current_app.config.get("SLURM_HISTORY_ENABLED", True),
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
        status_enabled: bool = True, history_enabled: bool = True,
    ) -> None:
        self.hosts = hosts
        self.history_days = history_days
        # Switched off, the corresponding Slurm commands are never run.
        self.status_enabled = status_enabled
        self.history_enabled = history_enabled
        self._cache_expiration = cache_experiation
        # Per-instance results, so a request asking about many jobs queries once.
        self._jobs: Optional[Dict[str, str]] = None
        self._history: Dict[date, Dict[str, List[SlurmRecord]]] = {}

        if cache_experiation > 0:
            memoize = CACHE.memoize(expire=cache_experiation)
            self._get_jobs_from_host = memoize(self._get_jobs_from_host)  # type: ignore
            self._get_history_from_host = memoize(self._get_history_from_host)  # type: ignore
            self._get_queue_from_host = memoize(self._get_queue_from_host)  # type: ignore

    @property
    def cache_experation(self) -> int:
        return self._cache_expiration

    def get_jobs(self) -> Dict[str, str]:
        """Returns the Slurm states of queued and running jobs by matching key."""
        if not self.status_enabled:
            return {}
        if self._jobs is None:
            self._jobs = {}
            for host in self.hosts:
                self._jobs.update(self._get_jobs_from_host(host))
        return self._jobs

    def get_history(self, since: Optional[date] = None) -> Dict[str, List[SlurmRecord]]:
        """Returns Slurm records submitted since a date by matching key.

        Defaults to the last `history_days` days. Records are sorted by submit time.
        """
        if not self.history_enabled:
            return {}
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

    def get_queue(self, all_users: bool = False) -> List[QueueJob]:
        """Returns queued and running jobs on all hosts (own jobs by default)."""
        jobs = []
        for host in self.hosts:
            jobs.extend(self._get_queue_from_host(host, all_users))
        return jobs

    def _get_queue_from_host(self, host: str, all_users: bool) -> List[QueueJob]:
        LOGGER.info("Getting the slurm queue from %s.", host)
        args = ["squeue", "--noheader", "-o", queue_format()]
        if not all_users:
            args.insert(1, "--me")
        output = run_command(host, args)
        return parse_queue(output, host) if output is not None else []

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
