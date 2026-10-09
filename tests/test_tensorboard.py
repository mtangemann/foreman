import importlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace

import pytest
import tensorboard.program
from tensorboard.backend import application
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.test import Client
from werkzeug.wrappers import Response

from foreman.tensorboard import has_event_files


@pytest.fixture
def viewers(tmp_path, monkeypatch):
    """Keep real routing/filesystem checks; replace background TensorBoard loaders."""
    monkeypatch.setenv("R3_REPOSITORY", str(tmp_path))
    module = importlib.import_module("foreman.app")
    monkeypatch.setattr(module, "R3_REPOSITORY", tmp_path)
    monkeypatch.setitem(module.app.config, "TESTING", True)
    dispatcher = DispatcherMiddleware(module.app)
    monkeypatch.setattr(module, "joint_app", dispatcher)
    manager = type(module.tensorboard_viewers)(dispatcher.mounts)
    monkeypatch.setattr(module, "tensorboard_viewers", manager)

    class Program:
        def __init__(self):
            self.plugin_loaders = []
            self.assets_zip_provider = None

        def configure(self, **kwargs):
            self.flags = SimpleNamespace(**kwargs)

        def _make_data_provider(self):
            return None, None

    generation = 0

    def make_wsgi(flags, *args):
        nonlocal generation
        generation += 1
        # A viewer exposes its loader generation, detecting unwanted restarts.
        body = json.dumps({"logdir": flags.logdir, "generation": generation})
        return Response(body, mimetype="application/json")

    monkeypatch.setattr(tensorboard.program, "TensorBoard", Program)
    monkeypatch.setattr(application, "TensorBoardWSGIApp", make_wsgi)
    jobs = tmp_path / "jobs"
    for name in ("first", "second", "empty"):
        (jobs / name).mkdir(parents=True)
    for name in ("first", "second"):
        logs = jobs / name / "output" / "logs"
        logs.mkdir(parents=True)
        (logs / "events.out.tfevents.example").write_bytes(b"event data")
    return Client(dispatcher, Response), jobs, module


def test_empty_and_missing_jobs_do_not_launch_viewers(viewers):
    client, _, _ = viewers
    assert client.get("/tensorboard/open/empty").status_code == 404
    assert client.get("/tensorboard/open/missing").status_code == 404


def test_viewer_reopens_without_restarting_loader(viewers):
    client, jobs, _ = viewers
    response = client.get("/tensorboard/open/first")
    assert response.status_code == 302
    assert response.headers["Location"] == "/tensorboard/first/"
    first = client.get(response.headers["Location"]).get_json()
    client.get("/tensorboard/open/first")
    second = client.get("/tensorboard/first/").get_json()
    assert first == second
    assert first["logdir"] == str(jobs / "first")


def test_opening_second_job_keeps_first_viewer_working(viewers):
    client, jobs, _ = viewers
    client.get("/tensorboard/open/first")
    first = client.get("/tensorboard/first/").get_json()
    client.get("/tensorboard/open/second")
    response = client.get("/tensorboard/first/")
    assert response.status_code == 200
    assert response.get_json() == first
    assert client.get("/tensorboard/second/").get_json()["logdir"] == str(
        jobs / "second"
    )


def test_viewer_rechecks_data_before_reopen(viewers):
    client, jobs, _ = viewers
    client.get("/tensorboard/open/first")
    (jobs / "first/output/logs/events.out.tfevents.example").unlink()
    assert client.get("/tensorboard/open/first").status_code == 404


def test_viewer_rejects_paths_outside_jobs(viewers, tmp_path):
    client, jobs, _ = viewers
    (tmp_path / "events.out.tfevents.outside").write_bytes(b"event")
    (jobs / "linked").symlink_to(tmp_path, target_is_directory=True)
    assert client.get("/tensorboard/open/linked").status_code == 404
    assert client.get("/tensorboard/open/..").status_code == 404
    (jobs / "loop").symlink_to(jobs / "loop")
    assert client.get("/tensorboard/open/loop").status_code == 404


def test_concurrent_opens_share_one_viewer(viewers):
    client, _, _ = viewers
    barrier = Barrier(4)

    def open_viewer(_):
        barrier.wait(timeout=5)
        response = client.get("/tensorboard/open/first")
        assert response.status_code == 302
        return client.get(response.headers["Location"]).get_json()

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(open_viewer, range(4)))
    assert all(result == results[0] for result in results)


def test_initialization_failure_is_reported_and_retryable(viewers, monkeypatch):
    client, _, _ = viewers
    original = tensorboard.program.TensorBoard._make_data_provider

    def fail(_):
        raise RuntimeError("loader failed")

    monkeypatch.setattr(tensorboard.program.TensorBoard, "_make_data_provider", fail)
    response = client.get("/tensorboard/open/first")
    assert response.status_code == 503
    assert b"TensorBoard could not be started" in response.data
    assert client.get("/tensorboard/first/").status_code == 404
    monkeypatch.setattr(
        tensorboard.program.TensorBoard, "_make_data_provider", original
    )
    assert client.get("/tensorboard/open/first").status_code == 302
    assert client.get("/tensorboard/first/").status_code == 200


def test_detection_only_stats_nested_event_files(tmp_path, monkeypatch):
    logs = tmp_path / "artifacts/nested/train"
    logs.mkdir(parents=True)
    (logs / "events.out.tfevents.123.host.0.v2").write_bytes(b"data")

    def no_reads(*args, **kwargs):
        raise AssertionError("Discovery must not read event data")

    monkeypatch.setattr(Path, "open", no_reads)
    assert has_event_files(tmp_path)


@pytest.mark.parametrize(
    "filename", ["weights.pb", "events.out.tfevents", "events.out.tfevents."]
)
def test_non_event_filenames_are_not_detected(tmp_path, filename):
    (tmp_path / filename).write_bytes(b"not an event")
    assert not has_event_files(tmp_path)


def test_detection_ignores_empty_files_directories_and_symlinks(tmp_path):
    root = tmp_path / "job"
    root.mkdir()
    (root / "events.out.tfevents.empty").touch()
    (root / "events.out.tfevents.directory").mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    marker = outside / "events.out.tfevents.external"
    marker.write_bytes(b"event")
    (root / "events.out.tfevents.link").symlink_to(marker)
    (root / "linked-directory").symlink_to(outside, target_is_directory=True)
    assert not has_event_files(root)
    assert not has_event_files(root / "linked-directory")


def test_missing_and_inaccessible_directories_are_not_detected(tmp_path, monkeypatch):
    assert not has_event_files(tmp_path / "missing")
    original = Path.is_dir

    def inaccessible(path):
        if path == tmp_path:
            raise PermissionError("directory is inaccessible")
        return original(path)

    monkeypatch.setattr(Path, "is_dir", inaccessible)
    assert not has_event_files(tmp_path)


def test_detection_respects_depth_and_entry_budgets(tmp_path, monkeypatch):
    from foreman import tensorboard as backend

    logs = tmp_path / "output/logs"
    logs.mkdir(parents=True)
    (logs / "events.out.tfevents.nested").write_bytes(b"event")
    monkeypatch.setattr(backend, "MAX_DEPTH", 1)
    assert not has_event_files(tmp_path)
    monkeypatch.setattr(backend, "MAX_DEPTH", 2)
    assert has_event_files(tmp_path)
    monkeypatch.setattr(backend, "MAX_ENTRIES", 1)
    assert not has_event_files(tmp_path)
