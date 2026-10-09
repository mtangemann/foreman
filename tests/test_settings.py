import os
from pathlib import Path

import pytest
from flask import Flask

import foreman.slurm as slurm_module
from foreman import settings
from foreman.blueprints import BLUEPRINTS
from foreman.job import tensorboard_available
from foreman.slurm import Slurm
from foreman.templating import autoescape
from foreman.tracking_index import TrackingIndex

HEADERS = {"X-Foreman-Request": "settings"}
RUN_URL = "https://mlflow.test/#/experiments/1/runs/" + "a" * 32


@pytest.fixture
def conf(tmp_path):
    return tmp_path / "foreman.conf"


@pytest.fixture
def app(repo, conf, monkeypatch):
    repository, _ = repo
    monkeypatch.setenv("R3_REPOSITORY", str(repository.r3_repository.path))
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).resolve().parents[1] / "foreman/templates"),
    )
    app.jinja_env.autoescape = autoescape
    app.config.from_object("foreman.config")
    app.config.update(TESTING=True, SLURM_HOSTS=[], SLURM_CACHE_EXPIRATION=0)
    settings.load(app.config, conf)
    for blueprint in BLUEPRINTS:
        app.register_blueprint(blueprint)
    return app


def form(**overrides):
    """A settings page form with every switch on and default values."""
    data = {s.key: "on" for s in settings.SETTINGS if s.is_switch}
    data.update(
        SLURM_HOSTS="galvani", SLURM_CACHE_EXPIRATION="5",
        SLURM_HISTORY_DAYS="30", TRACKING_REFRESH_INTERVAL="60",
    )
    for key, value in overrides.items():
        if value is None:
            data.pop(key)
        else:
            data[key] = value
    return data


def test_write_and_read_round_trip(conf):
    settings.write(conf, {"MLFLOW_ENABLED": False, "SLURM_HISTORY_DAYS": 7})
    text = conf.read_text()
    assert "[tracking]\nmlflow = off" in text and "[slurm]\nhistory_days = 7" in text
    assert settings.read(conf) == {"MLFLOW_ENABLED": False, "SLURM_HISTORY_DAYS": 7}


def test_invalid_and_unknown_entries_are_ignored(conf):
    conf.write_text(
        "[tracking]\nmlflow = maybe\nwandb = no\nunknown = 1\n"
        "[slurm]\nhistory_days = 0\n[indexing]\nrefresh_interval = 120\n"
    )
    assert settings.read(conf) == {
        "WANDB_ENABLED": False, "TRACKING_REFRESH_INTERVAL": 120,
    }
    conf.write_text("not an ini file")
    assert settings.read(conf) == {}


def test_load_records_sources(app, conf):
    assert app.config["SETTINGS_SOURCES"]["MLFLOW_ENABLED"] == "default"
    settings.write(conf, {"MLFLOW_ENABLED": False})
    settings.load(app.config, conf)
    assert app.config["MLFLOW_ENABLED"] is False
    assert app.config["SETTINGS_SOURCES"]["MLFLOW_ENABLED"] == "foreman.conf"


def test_hand_edits_apply_on_next_request(app, conf):
    client = app.test_client()
    conf.write_text("[tracking]\ntensorboard = off\n")
    client.get("/settings/")
    assert app.config["TENSORBOARD_ENABLED"] is False
    conf.write_text("# nothing\n")
    os.utime(conf, ns=(1, 1))  # a different mtime, even within the same tick
    client.get("/settings/")
    assert app.config["TENSORBOARD_ENABLED"] is True  # back to config.py


def test_page_shows_every_setting(app):
    html = app.test_client().get("/settings/").get_data(as_text=True)
    for setting in settings.SETTINGS:
        assert (f'name="{setting.key}"' in html) == (setting.page == "settings")
    assert "Tracking link index" in html


def test_saving_requires_foreman_header(app, conf):
    response = app.test_client().post("/settings/", data=form(MLFLOW_ENABLED=None))
    assert response.status_code == 403
    assert not conf.exists()


def test_saving_writes_only_changes(app, conf):
    response = app.test_client().post(
        "/settings/", data=form(MLFLOW_ENABLED=None, SLURM_HISTORY_DAYS="7"),
        headers=HEADERS,
    )
    assert response.status_code == 200
    assert "Saved" in response.get_data(as_text=True)
    assert settings.read(conf) == {"MLFLOW_ENABLED": False, "SLURM_HISTORY_DAYS": 7}
    assert app.config["MLFLOW_ENABLED"] is False
    assert app.config["SETTINGS_SOURCES"]["SLURM_HISTORY_DAYS"] == "foreman.conf"


def test_invalid_numbers_are_rejected(app, conf):
    response = app.test_client().post(
        "/settings/", data=form(TRACKING_REFRESH_INTERVAL="1"), headers=HEADERS
    )
    assert response.status_code == 400
    html = response.get_data(as_text=True)
    assert 'role="alert"' in html and "must be at least 10" in html
    assert not conf.exists()


def test_unwritable_settings_file_is_reported(app, tmp_path):
    app.config["SETTINGS_PATH"] = tmp_path / "missing" / "foreman.conf"
    response = app.test_client().post("/settings/", data=form(), headers=HEADERS)
    assert response.status_code == 500
    assert "Could not write" in response.get_data(as_text=True)


def test_switching_mlflow_off_hides_links_immediately(app, repo):
    repository, ids = repo
    output = Path(repository.get(ids["a"]).path) / "output"
    (output / "slurm_1.log").write_text(f"View run a at: {RUN_URL}\n")
    client = app.test_client()
    assert RUN_URL in client.get("/list/").get_data(as_text=True)
    client.post("/settings/", data=form(MLFLOW_ENABLED=None), headers=HEADERS)
    assert RUN_URL not in client.get("/list/").get_data(as_text=True)
    assert RUN_URL not in client.get(f"/job/view/{ids['a']}").get_data(as_text=True)


def test_switching_log_scan_off_skips_logs(app, repo):
    repository, ids = repo
    output = Path(repository.get(ids["a"]).path) / "output"
    (output / "slurm_1.log").write_text(f"View run a at: {RUN_URL}\n")
    client = app.test_client()
    client.post("/settings/", data=form(TRACKING_LOG_SCAN=None), headers=HEADERS)
    assert RUN_URL not in client.get(f"/job/view/{ids['a']}").get_data(as_text=True)


def test_switching_tensorboard_off(app, repo):
    repository, ids = repo
    job_root = Path(repository.get(ids["a"]).path)
    (job_root / "output" / "events.out.tfevents.1.host").write_bytes(b"data")
    with app.test_request_context():
        assert tensorboard_available(repository.get(ids["a"]))
        app.config["TENSORBOARD_ENABLED"] = False
        assert not tensorboard_available(repository.get(ids["a"]))


def test_switched_off_slurm_runs_no_commands(monkeypatch):
    def fail(*args):
        raise AssertionError("a Slurm command was run")

    monkeypatch.setattr(slurm_module, "run_command", fail)
    slurm = Slurm(["local"], cache_experiation=0,
                  status_enabled=False, history_enabled=False)
    assert slurm.get_jobs() == {}
    assert slurm.get_history() == {}
    assert slurm.get_records("any") == []


def test_cache_can_be_cleared_from_the_page(app, repo, monkeypatch):
    index = TrackingIndex()
    monkeypatch.setattr("foreman.settings_page.tracking_index", index)
    repository, ids = repo
    root = Path(repository.get(ids["a"]).path)
    (root / "output" / "slurm_1.log").write_text(f"View run a at: {RUN_URL}\n")
    index.refresh(root)
    assert index._store
    response = app.test_client().post("/settings/cache/clear", headers=HEADERS)
    assert response.status_code == 200
    assert "Cache cleared" in response.get_data(as_text=True)
    assert not index._store
    assert app.test_client().post("/settings/cache/clear").status_code == 403
