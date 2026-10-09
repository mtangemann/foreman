import os
import time

from foreman import tracking
from foreman.tracking_index import TrackingIndex, default_cache_dir

RUN_URL = "https://mlflow.test/#/experiments/1/runs/" + "a" * 32


def job(root, name, logged=True):
    path = root / "jobs" / name
    (path / "output").mkdir(parents=True)
    if logged:
        (path / "output" / "slurm_1.log").write_text(f"View run {name} at: {RUN_URL}\n")
    return path


def wait_for(condition, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


def test_without_indexer_links_are_discovered_per_request(tmp_path):
    index = TrackingIndex()
    root = job(tmp_path, "a")
    assert [link["url"] for link in index.links(root)] == [RUN_URL]


def test_background_tables_only_read_indexed_links(tmp_path, monkeypatch):
    index = TrackingIndex()
    os.utime(job(tmp_path, "old"), (0, 0))
    job(tmp_path, "new")
    refreshed = []
    original = index.refresh

    def tracked_refresh(root, **settings):
        refreshed.append(root.name)
        return original(root, **settings)

    monkeypatch.setattr(index, "refresh", tracked_refresh)
    index.start(tmp_path / "jobs", lambda: (3600, {}))
    wait_for(lambda: len(refreshed) == 2)
    assert refreshed == ["new", "old"]  # most recently changed first
    assert index.background

    later = job(tmp_path, "later")
    assert index.links(later) == []  # not indexed yet: no scan in the request
    assert "later" not in refreshed
    assert [link["url"] for link in index.links(later, fresh=True)] == [RUN_URL]
    assert index.links(later) != []  # the fresh result is stored for tables


def test_disk_cache_keeps_links_and_log_scans_across_restarts(tmp_path, monkeypatch):
    # use_disk_cache replaces the module-wide log cache; restore it afterwards.
    monkeypatch.setattr(tracking, "_log_cache", {})
    root = job(tmp_path, "a")
    first = TrackingIndex()
    first.use_disk_cache(tmp_path / "cache")
    first.refresh(root)

    second = TrackingIndex()
    second.use_disk_cache(tmp_path / "cache")
    second._thread = type("Alive", (), {"is_alive": lambda self: True})()
    assert [link["url"] for link in second.links(root)] == [RUN_URL]

    def no_reads(*args, **kwargs):
        raise AssertionError("an unchanged log was read again")

    monkeypatch.setattr(tracking, "_links_in", no_reads)
    assert [link["url"] for link in second.refresh(root)] == [RUN_URL]


def test_settings_are_part_of_the_index_key(tmp_path):
    root = job(tmp_path, "a", logged=False)
    (root / "metadata.yaml").write_text(
        "tracking:\n  mlflow:\n    experiment_id: '1'\n    run_id: r\n"
    )
    index = TrackingIndex()
    assert index.refresh(root, mlflow_ui_url="https://one.test")[0]["url"].startswith(
        "https://one.test"
    )
    assert index.refresh(root, mlflow_ui_url="https://two.test")[0]["url"].startswith(
        "https://two.test"
    )


def test_default_cache_dir_follows_xdg(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert default_cache_dir() == tmp_path / "foreman" / "tracking"
