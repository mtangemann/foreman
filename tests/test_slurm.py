from datetime import date, datetime, timedelta
from pathlib import Path

import flask
import pytest

import foreman.slurm as slurm_module
from foreman.job_stats import build_timeline, compute_stats, format_duration
from foreman.slurm import Slurm, parse_sacct, parse_squeue

UUID = "04d1ceec-f7dd-4ad8-969f-2b44d860fe03"
WORK_DIR = f"/r3_repo/jobs/{UUID}"
T0 = datetime(2026, 10, 9, 11, 0, 0)


def sacct_line(job_id="100", name="readable", state="COMPLETED",
               submit="2026-10-09T11:00:10", start="2026-10-09T11:05:10",
               end="2026-10-09T12:05:10", exit_code="0:0", work_dir=WORK_DIR):
    return "|".join([job_id, name, state, submit, start, end, exit_code,
                     "node-1", "gpu", work_dir])


def test_parse_sacct_matches_by_readable_name_or_work_dir():
    output = "\n".join([
        sacct_line(state="CANCELLED by 4726", start="None", end="Unknown"),
        sacct_line(job_id="101", name=UUID, work_dir="/elsewhere"),
        "malformed line",
    ])
    first, second = parse_sacct(output, host="local")
    assert first.state == "CANCELLED"
    assert first.status == "cancelled"
    assert first.start is None and first.end is None
    assert UUID in first.keys and "readable" in first.keys
    assert UUID in second.keys


def test_parse_squeue_matches_by_readable_name_or_work_dir():
    jobs = parse_squeue(f"decoding|RUNNING|{WORK_DIR}\n{UUID}|PENDING|/tmp\n")
    assert jobs == {"decoding": "RUNNING", UUID: "PENDING", "tmp": "PENDING"}


def test_local_host_runs_commands_without_ssh(monkeypatch):
    calls = []

    def run_command(host, args):
        calls.append((host, args[0]))
        return sacct_line() if args[0] == "sacct" else f"x|RUNNING|{WORK_DIR}"

    monkeypatch.setattr(slurm_module, "run_command", run_command)
    slurm = Slurm(["local"], cache_experiation=0)
    assert slurm.get_jobs() == {"x": "RUNNING", UUID: "RUNNING"}
    assert [record.job_id for record in slurm.get_records(UUID)] == ["100"]
    # Results are kept for the instance (one request), not re-queried per job.
    slurm.get_jobs()
    slurm.get_records(UUID)
    assert calls == [("local", "squeue"), ("local", "sacct")]


def test_get_records_extends_history_for_old_jobs(monkeypatch):
    since_dates = []

    def run_command(host, args):
        since_dates.append(args[args.index("--starttime") + 1])
        return ""

    monkeypatch.setattr(slurm_module, "run_command", run_command)
    slurm = Slurm(["local"], cache_experiation=0, history_days=30)
    slurm.get_records(UUID, since=date.today())
    slurm.get_records(UUID, since=date(2020, 1, 1))
    default_since = (date.today() - timedelta(days=30)).isoformat()
    assert since_dates == [default_since, "2020-01-01"]


def test_failed_job_stats_and_timeline():
    records = parse_sacct(sacct_line(state="FAILED", exit_code="1:0"), "local")
    stats = compute_stats(T0, records, now=T0 + timedelta(days=1))
    assert stats.outcome == "failed"
    assert stats.submit_delay == timedelta(seconds=10)
    assert stats.queued == timedelta(minutes=5)
    assert stats.running == stats.failed_after == timedelta(hours=1)
    assert stats.total == timedelta(hours=1, minutes=5)

    timeline = build_timeline(T0, records, now=T0 + timedelta(days=1))
    statuses = [event.status for event in timeline]
    assert statuses == ["ready", "queued", "running", "failed"]
    assert timeline[-1].description == "Failed with exit code 1:0"
    assert timeline[1].duration == timedelta(minutes=5)
    assert not timeline[-1].ongoing


def test_running_job_counts_until_now():
    records = parse_sacct(sacct_line(state="RUNNING", end="Unknown"), "local")
    now = datetime(2026, 10, 9, 13, 5, 10)
    stats = compute_stats(T0, records, now)
    assert stats.is_active and stats.failed_after is None
    assert stats.running == timedelta(hours=8 - 6)
    timeline = build_timeline(T0, records, now)
    assert timeline[-1].ongoing and timeline[-1].duration == timedelta(hours=2)


def test_pending_job_ignores_expected_start_time():
    records = parse_sacct(
        sacct_line(state="PENDING", start="2026-10-10T00:00:00", end="Unknown"), "local"
    )
    now = datetime(2026, 10, 9, 11, 30, 10)
    stats = compute_stats(T0, records, now)
    assert stats.queued == timedelta(minutes=30) and stats.running == timedelta()
    assert [event.status for event in build_timeline(T0, records, now)] == [
        "ready", "queued",
    ]


def test_preempted_and_requeued_attempts():
    records = parse_sacct("\n".join([
        sacct_line(state="PREEMPTED", end="2026-10-09T11:35:10"),
        sacct_line(state="COMPLETED", submit="2026-10-09T11:35:20",
                   start="2026-10-09T11:40:10"),
    ]), "local")
    stats = compute_stats(T0, records, now=T0 + timedelta(days=1))
    assert (stats.attempts, stats.preemptions) == (2, 1)
    assert stats.running == timedelta(minutes=30) + timedelta(minutes=25)
    descriptions = [event.description for event in build_timeline(T0, records, T0)]
    assert descriptions == [
        "Committed to R3", "Submitted to Slurm", "Started on node-1", "Preempted",
        "Requeued", "Started on node-1", "Completed",
    ]


def test_never_submitted_job_has_no_stats():
    assert compute_stats(T0, [], T0) is None
    assert [event.status for event in build_timeline(T0, [], T0)] == ["ready"]


@pytest.mark.parametrize(("seconds", "text"), [
    (None, "–"), (42, "42s"), (192, "3m 12s"), (11_520, "3h 12m"), (180_000, "2d 2h"),
])
def test_format_duration(seconds, text):
    duration = None if seconds is None else timedelta(seconds=seconds)
    assert format_duration(duration) == text


class FakeSlurm:
    def __init__(self, active, history):
        self.active, self.history = active, history

    def get_jobs(self):
        return self.active

    def get_history(self):
        return self.history


@pytest.mark.parametrize(("in_queue", "state", "status"), [
    (False, "OUT_OF_MEMORY", "out of memory"),
    (False, "TIMEOUT", "timeout"),
    (True, "PENDING", "queued"),
    (True, "RUNNING", "running"),
])
def test_status_falls_back_to_slurm_without_done_file(repo, in_queue, state, status):
    repository, ids = repo
    job_id = ids["a"]
    (Path(repository.get(job_id).path) / "output" / "done").unlink()
    active = {job_id: state} if in_queue else {}
    history = {job_id: parse_sacct(sacct_line(state=state, end="Unknown"), "local")}
    with flask.Flask(__name__).app_context():
        flask.g.slurm = FakeSlurm(active, history)
        assert repository.get(job_id).status == status
        flask.g.slurm = FakeSlurm({}, {})
        assert repository.get(job_id).status == "ready"
