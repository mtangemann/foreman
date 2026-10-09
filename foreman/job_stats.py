"""Timeline and run statistics of a job's Slurm allocations."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import pairwise
from typing import List, Optional

from foreman.slurm import SlurmRecord

PREEMPTED_STATES = {"PREEMPTED", "REQUEUED"}
FAILURE_STATUSES = {"failed", "cancelled", "timeout", "node failure", "out of memory"}

END_DESCRIPTIONS = {
    "COMPLETED": "Completed",
    "FAILED": "Failed",
    "BOOT_FAIL": "Node failed to boot",
    "CANCELLED": "Cancelled",
    "DEADLINE": "Reached its deadline",
    "TIMEOUT": "Reached its time limit",
    "NODE_FAIL": "Node failed",
    "OUT_OF_MEMORY": "Ran out of memory",
    "PREEMPTED": "Preempted",
    "REQUEUED": "Requeued",
}


@dataclass
class TimelineEvent:
    """A change in a job's status."""

    datetime: datetime
    status: str
    description: str
    slurm_job_id: Optional[str] = None
    # Time until the next event, or until now for the job's current status.
    duration: Optional[timedelta] = None
    ongoing: bool = False


@dataclass
class JobStats:
    """How long a job waited and ran on Slurm, summed over all attempts."""

    attempts: int
    preemptions: int
    # From commit to first submission; None if the job has no commit timestamp.
    submit_delay: Optional[timedelta]
    queued: timedelta
    running: timedelta
    # From first submission to the end of the last attempt (or now).
    total: timedelta
    outcome: str
    is_active: bool
    # How long the last attempt ran before failing; None if it did not fail or start.
    failed_after: Optional[timedelta]
    last: SlurmRecord


def _start(record: SlurmRecord) -> Optional[datetime]:
    # Slurm reports an expected start time for pending jobs; ignore it.
    return None if record.state == "PENDING" else record.start


def build_timeline(
    committed: Optional[datetime], records: List[SlurmRecord], now: datetime,
) -> List[TimelineEvent]:
    """Returns the events of a job, oldest first."""
    events = []
    if committed is not None:
        events.append(TimelineEvent(committed, "ready", "Committed to R3"))

    seen_ids: set[str] = set()
    for record in records:
        if record.submit is not None:
            if record.job_id in seen_ids:
                description = "Requeued"
            elif seen_ids:
                description = "Resubmitted"
            else:
                description = "Submitted to Slurm"
            events.append(
                TimelineEvent(record.submit, "queued", description, record.job_id)
            )
        seen_ids.add(record.job_id)

        start = _start(record)
        if start is not None:
            description = f"Started on {record.nodes}" if record.nodes else "Started"
            events.append(TimelineEvent(start, "running", description, record.job_id))

        if record.end is not None and not record.is_active:
            description = END_DESCRIPTIONS.get(record.state, record.state.capitalize())
            if record.state == "FAILED":
                description += f" with exit code {record.exit_code}"
            events.append(
                TimelineEvent(record.end, record.status, description, record.job_id)
            )

    events.sort(key=lambda event: event.datetime)
    for event, next_event in pairwise(events):
        event.duration = next_event.datetime - event.datetime
    if events and records and records[-1].is_active:
        events[-1].duration = now - events[-1].datetime
        events[-1].ongoing = True

    return events


def compute_stats(
    committed: Optional[datetime], records: List[SlurmRecord], now: datetime,
) -> Optional[JobStats]:
    """Returns run statistics of a job, or None if it was never submitted."""
    records = [record for record in records if record.submit is not None]
    if not records:
        return None

    queued = timedelta()
    running = timedelta()
    for record in records:
        start = _start(record)
        assert record.submit is not None
        queued += (start or record.end or now) - record.submit
        if start is not None:
            running += (record.end or now) - start

    last = records[-1]
    first_submit = records[0].submit
    assert first_submit is not None
    end = now if last.is_active or last.end is None else last.end

    failed_after = None
    last_start = _start(last)
    if last.status in FAILURE_STATUSES and last_start and last.end:
        failed_after = last.end - last_start

    return JobStats(
        attempts=len(records),
        preemptions=sum(record.state in PREEMPTED_STATES for record in records),
        submit_delay=first_submit - committed if committed else None,
        queued=queued,
        running=running,
        total=end - first_submit,
        outcome=last.status,
        is_active=last.is_active,
        failed_after=failed_after,
        last=last,
    )


def format_duration(duration: Optional[timedelta]) -> str:
    """Formats a duration with its two largest units, e.g. `3h 12m`."""
    if duration is None:
        return "–"
    seconds = max(int(duration.total_seconds()), 0)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"
