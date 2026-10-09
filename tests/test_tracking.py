import json
from pathlib import Path

import pytest
import yaml

from foreman import tracking
from foreman.tracking import discover_tracking_links


def write(root, path, data):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(data))
    return target


def mlflow(root, run="run-a", experiment="12", prefix="outputs"):
    write(root, f"{prefix}/mlruns/{experiment}/meta.yaml",
          {"experiment_id": experiment})
    write(root, f"{prefix}/mlruns/{experiment}/{run}/meta.yaml",
          {"run_id": run, "experiment_id": experiment})


def wandb(root, name="run-20240101_120000-abc"):
    target = root / "nested" / "wandb" / name / "files" / "wandb-metadata.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"entity": "team", "project": "project"}))
    return target


def test_nested_mlflow_and_multiple_runs(tmp_path):
    mlflow(tmp_path, "second")
    mlflow(tmp_path, "first")
    mlflow(tmp_path, "first", prefix="duplicate")
    assert discover_tracking_links(tmp_path) == []
    links = discover_tracking_links(tmp_path, mlflow_ui_url="https://ml.test/ui/")
    assert [link["url"] for link in links] == [
        "https://ml.test/ui/#/experiments/12/runs/first",
        "https://ml.test/ui/#/experiments/12/runs/second",
    ]
    assert all(link["provider"] == "mlflow" for link in links)


def test_mlflow_legacy_run_uuid_and_experiment_metadata(tmp_path):
    write(tmp_path, "mlruns/1/meta.yaml", {"experiment_id": 1})
    write(tmp_path, "mlruns/1/legacy/meta.yaml", {"run_uuid": "legacy"})
    assert discover_tracking_links(tmp_path, mlflow_ui_url="http://localhost:5000") == [
        {"provider": "mlflow", "label": "MLflow: legacy",
         "url": "http://localhost:5000/#/experiments/1/runs/legacy"}
    ]


def test_online_wandb_requires_identity_and_offline_requires_url(tmp_path):
    online = wandb(tmp_path)
    wandb(tmp_path, "offline-run-20240101_120000-offline")
    links = discover_tracking_links(tmp_path)
    assert links == [
        {"provider": "wandb", "label": "W&B: project/abc",
         "url": "https://wandb.ai/team/project/runs/abc"}
    ]
    assert discover_tracking_links(tmp_path, wandb_ui_url="https://private.test")[0][
        "url"
    ] == "https://private.test/team/project/runs/abc"
    online.write_text("{}")
    assert discover_tracking_links(tmp_path, wandb_ui_url="https://private.test") == []


def test_wandb_saved_settings(tmp_path):
    directory = tmp_path / "wandb" / "run-20240101-id"
    directory.mkdir(parents=True)
    (directory.parent / "settings").write_text(
        "[default]\nentity = team\nproject = saved\nbase_url = https://private.test\n"
    )
    assert discover_tracking_links(tmp_path)[0]["url"] == (
        "https://private.test/team/saved/runs/id"
    )


def test_saved_wandb_cloud_api_host_and_configured_ui(tmp_path):
    marker = wandb(tmp_path)
    marker.write_text(json.dumps({
        "entity": "team", "project": "project", "base_url": "https://api.wandb.ai",
        "id": "saved-id",
    }))
    assert discover_tracking_links(tmp_path)[0]["url"] == (
        "https://wandb.ai/team/project/runs/saved-id"
    )
    assert discover_tracking_links(tmp_path, wandb_ui_url="https://ui.test")[0][
        "url"
    ] == "https://ui.test/team/project/runs/saved-id"


def test_offline_saved_explicit_url(tmp_path):
    marker = wandb(tmp_path, "offline-run-20240101-offline")
    marker.write_text(json.dumps({"url": "https://wandb.ai/team/project/runs/synced"}))
    assert discover_tracking_links(tmp_path)[0]["url"] == (
        "https://wandb.ai/team/project/runs/synced"
    )


def test_run_specific_wandb_json_settings(tmp_path):
    directory = tmp_path / "wandb" / "run-20240101-id" / "files"
    directory.mkdir(parents=True)
    (directory / "wandb-settings.json").write_text(json.dumps({
        "wandb": {"entity": "team", "project": "saved", "run_id": "specific"}
    }))
    assert discover_tracking_links(tmp_path)[0]["url"] == (
        "https://wandb.ai/team/saved/runs/specific"
    )


def test_explicit_provider_override_keeps_other_provider(tmp_path):
    mlflow(tmp_path)
    wandb(tmp_path)
    write(tmp_path, "metadata.yaml", {"tracking": {"mlflow": [
        {"tracking_uri": "https://ml.test", "experiment_id": 3, "run_id": "explicit"},
        {"url": "https://other.test/run"},
        {"url": "https://other.test/run"},
    ]}})
    links = discover_tracking_links(tmp_path, mlflow_ui_url="https://configured.test")
    assert [link["url"] for link in links] == [
        "https://configured.test/#/experiments/3/runs/explicit",
        "https://other.test/run",
        "https://wandb.ai/team/project/runs/abc",
    ]


def test_explicit_mlflow_host_precedence_and_local_uri(tmp_path):
    fields = {"experiment_id": 0, "run_id": "run", "tracking_uri": "file:///mlruns"}
    write(tmp_path, "metadata.yaml", {"tracking": {"mlflow": fields}})
    assert discover_tracking_links(tmp_path) == []
    fields["tracking_uri"] = "https://tracking.test"
    write(tmp_path, "metadata.yaml", {"tracking": {"mlflow": fields}})
    assert discover_tracking_links(tmp_path)[0]["url"].startswith(
        "https://tracking.test/"
    )
    fields["ui_url"] = "https://record-ui.test"
    write(tmp_path, "metadata.yaml", {"tracking": {"mlflow": fields}})
    assert discover_tracking_links(tmp_path, mlflow_ui_url="https://config.test")[
        0
    ]["url"] == "https://record-ui.test/#/experiments/0/runs/run"


@pytest.mark.parametrize("override", [None, [], {}, {"url": "file:///run"}])
def test_empty_or_invalid_override_suppresses_auto(tmp_path, override):
    mlflow(tmp_path)
    write(tmp_path, "metadata.yaml", {"tracking": {"mlflow": override}})
    assert discover_tracking_links(tmp_path, mlflow_ui_url="https://ml.test") == []


def test_explicit_offline_identity_and_escaped_segments(tmp_path):
    write(tmp_path, "metadata.yaml", {"tracking": {"wandb": {
        "entity": "a/b", "project": "my project", "run_id": "offline",
        "ui_url": "https://private.test/prefix/",
    }}})
    assert discover_tracking_links(tmp_path)[0]["url"] == (
        "https://private.test/prefix/a%2Fb/my%20project/runs/offline"
    )


@pytest.mark.parametrize("url", [
    "file:///run", "javascript:alert(1)", "https://user:secret@test/run",
    "https://user@test/run", "https:///run", "https://test:bad/run",
    "https://test:99999/run", "https://test/\nrun", " https://test/run",
    "https://test\\@evil/run",
])
def test_unsafe_urls_are_ignored(tmp_path, url):
    write(tmp_path, "metadata.yaml", {"tracking": {"wandb": {"url": url}}})
    assert discover_tracking_links(tmp_path) == []
    write(tmp_path, "metadata.yaml", {})
    mlflow(tmp_path)
    assert discover_tracking_links(tmp_path, mlflow_ui_url=url) == []


def test_malformed_unreadable_and_large_files_are_ignored(tmp_path, monkeypatch):
    write(tmp_path, "metadata.yaml", {})
    (tmp_path / "metadata.yaml").write_text("tracking: [broken")
    mlflow(tmp_path)
    meta = tmp_path / "outputs/mlruns/12/run-a/meta.yaml"
    meta.write_text("a" * (tracking.MAX_FILE_BYTES + 1))
    assert discover_tracking_links(tmp_path, mlflow_ui_url="https://ml.test") == []
    mlflow(tmp_path)
    original_open = Path.open

    def unreadable(path, *args, **kwargs):
        if path == meta:
            raise PermissionError("unreadable")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", unreadable)
    assert discover_tracking_links(tmp_path, mlflow_ui_url="https://ml.test") == []


def test_artifacts_checkpoints_and_external_symlinks_are_skipped(tmp_path):
    root = tmp_path / "job"
    root.mkdir()
    mlflow(root, prefix="artifacts")
    mlflow(root, prefix="checkpoints")
    mlflow(tmp_path, prefix="external")
    (root / "linked").symlink_to(tmp_path / "external", target_is_directory=True)
    (root / "metadata.yaml").symlink_to(tmp_path / "external/mlruns/12/run-a/meta.yaml")
    assert discover_tracking_links(root, mlflow_ui_url="https://ml.test") == []


def test_depth_entry_and_file_limits(tmp_path, monkeypatch):
    mlflow(tmp_path, prefix="a/b/c")
    monkeypatch.setattr(tracking, "MAX_DEPTH", 2)
    assert discover_tracking_links(tmp_path, mlflow_ui_url="https://ml.test") == []
    monkeypatch.setattr(tracking, "MAX_DEPTH", 8)
    monkeypatch.setattr(tracking, "MAX_ENTRIES", 1)
    assert discover_tracking_links(tmp_path, mlflow_ui_url="https://ml.test") == []
    monkeypatch.setattr(tracking, "MAX_ENTRIES", 10_000)
    monkeypatch.setattr(tracking, "MAX_FILES", 1)
    assert discover_tracking_links(tmp_path, mlflow_ui_url="https://ml.test") == []


def test_missing_root_and_non_mapping_metadata(tmp_path):
    assert discover_tracking_links(tmp_path / "missing") == []
    write(tmp_path, "metadata.yaml", ["not", "a", "mapping"])
    assert discover_tracking_links(tmp_path) == []


def test_explicit_experiment_link_and_distinct_url_labels(tmp_path):
    write(tmp_path, "metadata.yaml", {"tracking": {
        "mlflow": {"experiment_id": 12, "ui_url": "https://ml.test"},
        "wandb": [{"url": "https://wandb.ai/t/p/runs/one"},
                  {"url": "https://wandb.ai/t/p/runs/two"}],
    }})
    links = discover_tracking_links(tmp_path)
    assert links[0]["url"] == "https://ml.test/#/experiments/12"
    assert links[1]["label"] != links[2]["label"]


RUN = "6f7b5c0ea95b4defae584d63b5d2925b"
MLFLOW_RUN_URL = f"https://mlflow.test/#/experiments/42/runs/{RUN}"


def log(root, path, text, mode="w"):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, mode) as stream:
        stream.write(text)
    return target


def test_mlflow_run_url_printed_in_slurm_log(tmp_path):
    log(tmp_path, "output/slurm_1.log",
        f"epoch 3\n\U0001f3c3 View run cosmos_ae at: {MLFLOW_RUN_URL}\n")
    assert discover_tracking_links(tmp_path) == [
        {"provider": "mlflow", "label": "MLflow: cosmos_ae", "url": MLFLOW_RUN_URL}
    ]


def test_log_runs_from_several_files_and_without_names(tmp_path):
    other = "0" * 32
    log(tmp_path, "output/a/run.log", f"View run first at: {MLFLOW_RUN_URL}\n")
    log(tmp_path, "output/b/run.txt",
        f"see https://mlflow.test/#/experiments/42/runs/{other} for details\n")
    log(tmp_path, "output/c/stderr.err",
        "wandb: \U0001f680 View run lively at: https://wandb.ai/team/proj/runs/ab12cd34\n")
    links = discover_tracking_links(tmp_path)
    assert [link["label"] for link in links] == [
        f"MLflow: {other}", "MLflow: first", "W&B: lively",
    ]


def test_log_links_respect_overrides_and_url_safety(tmp_path):
    log(tmp_path, "output/slurm_1.log",
        f"View run a at: {MLFLOW_RUN_URL}\n"
        "View run b at: https://user:pw@wandb.ai/team/proj/runs/ab12cd34\n")
    assert [link["provider"] for link in discover_tracking_links(tmp_path)] == [
        "mlflow"
    ]
    write(tmp_path, "metadata.yaml", {"tracking": {"mlflow": []}})
    assert discover_tracking_links(tmp_path) == []


def test_log_links_skip_checkpoints_symlinks_and_other_files(tmp_path):
    line = f"View run a at: {MLFLOW_RUN_URL}\n"
    log(tmp_path, "checkpoints/train.log", line)
    log(tmp_path, "output/results.csv", line)
    outside = log(tmp_path.parent / f"{tmp_path.name}-outside", "x.log", line)
    (tmp_path / "linked.log").symlink_to(outside)
    assert discover_tracking_links(tmp_path) == []


def test_growing_log_is_rescanned_incrementally(tmp_path, monkeypatch):
    path = log(tmp_path, "output/slurm_1.log", "starting\n" * 1000)
    assert discover_tracking_links(tmp_path) == []
    reads = []
    original = Path.open

    def tracked_open(self, *args, **kwargs):
        stream = original(self, *args, **kwargs)
        if self == path:
            seek = stream.seek
            def tracked_seek(offset, *rest):
                reads.append(offset)
                return seek(offset, *rest)
            stream.seek = tracked_seek
        return stream

    monkeypatch.setattr(Path, "open", tracked_open)
    assert discover_tracking_links(tmp_path) == []  # unchanged: not read again
    assert reads == []
    size = path.stat().st_size
    log(tmp_path, "output/slurm_1.log", f"View run a at: {MLFLOW_RUN_URL}\n", mode="a")
    links = discover_tracking_links(tmp_path)
    assert [link["url"] for link in links] == [MLFLOW_RUN_URL]
    assert reads == [size - tracking._LOG_OVERLAP_BYTES]


def test_large_logs_only_read_their_end(tmp_path, monkeypatch):
    monkeypatch.setattr(tracking, "MAX_LOG_BYTES", 100)
    log(tmp_path, "output/early.log",
        f"View run a at: {MLFLOW_RUN_URL}\n" + "x" * 200 + "\n")
    assert discover_tracking_links(tmp_path) == []
