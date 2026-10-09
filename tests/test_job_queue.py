import re
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from flask import Flask

import foreman.slurm as slurm_module
from foreman import settings
from foreman.blueprints import BLUEPRINTS
from foreman.slurm import _QUEUE_FIELDS, _QUEUE_SEPARATOR, Slurm, parse_queue
from foreman.templating import autoescape

HEADERS = {"X-Foreman-Request": "queue"}


def squeue_line(**fields):
    later = (datetime.now() + timedelta(hours=3)).strftime("%Y-%m-%dT%H:%M:%S")
    values = {
        "job_id": "101", "name": "train", "state": "PENDING", "reason": "Priority",
        "partition": "a100,a100-fat", "submit": "2026-10-09T10:00:00",
        "start": later, "end": "N/A", "elapsed": "0:00", "time_limit": "8:00:00",
        "nodes": "1", "nodelist": "", "cpus": "24", "tres": "gres/gpu:a100:2",
        "priority": "4600", "user": "me", "account": "lab", "qos": "normal",
        "work_dir": "/elsewhere",
    }
    values.update(fields)
    return _QUEUE_SEPARATOR.join(values[name] for name in _QUEUE_FIELDS)


def test_parse_queue():
    running = squeue_line(
        job_id="7_3", name="a|b", state="RUNNING", reason="None",
        start="2026-10-09T11:00:00", end="2026-10-12T11:00:00", elapsed="1-02:00:00",
        time_limit="3-00:00:00", nodes="2", nodelist="cn[1-2]", tres="gres/gpu:4",
    )
    unlimited = squeue_line(start="N/A", time_limit="UNLIMITED", tres="N/A")
    first, second = parse_queue(f"{running}\n{unlimited}\nbroken line\n", "local")
    assert first.job_id == "7_3" and first.name == "a|b" and not first.is_pending
    assert first.elapsed == timedelta(days=1, hours=2)
    assert first.progress == pytest.approx(26 / 72)
    assert first.gpus == 8  # per node, on two nodes
    assert second.is_pending and second.start is None
    assert second.time_limit is None and second.progress is None and second.gpus == 0


@pytest.fixture
def app(repo, monkeypatch, tmp_path):
    repository, ids = repo
    monkeypatch.setenv("R3_REPOSITORY", str(repository.r3_repository.path))
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).resolve().parents[1] / "foreman/templates"),
    )
    app.jinja_env.autoescape = autoescape
    app.config.from_object("foreman.config")
    app.config.update(TESTING=True, SLURM_HOSTS=["local"], SLURM_CACHE_EXPIRATION=0)
    settings.load(app.config, tmp_path / "foreman.conf")
    for blueprint in BLUEPRINTS:
        app.register_blueprint(blueprint)

    work_dir = str(Path(repository.get(ids["a"]).path))
    calls = []

    def run_command(host, args):
        calls.append(args)
        if "-o" not in args:
            return ""  # squeue/sacct for job statuses
        return "\n".join([
            squeue_line(job_id="200", name="decode", state="RUNNING", reason="None",
                        start="2026-10-09T11:00:00", elapsed="2:00:00",
                        nodelist="cn7", work_dir=work_dir),
            squeue_line(job_id="201", name="<b>other</b>", reason="QOSMaxGRESPerUser"),
        ])

    monkeypatch.setattr(slurm_module, "run_command", run_command)
    app.calls = calls
    return app


def queue_calls(app):
    return [args for args in app.calls if "-o" in args]


def test_queue_page_shows_running_and_pending_jobs(app, repo):
    _, ids = repo
    html = app.test_client().get("/queue/").get_data(as_text=True)
    assert "Job queue" in html and 'aria-label="Running"' in html
    assert f'href="/job/view/{ids["a"]}"' in html  # matched by working directory
    assert 'title="test/a">test/a</a>' in html and "#200" in html
    assert "Limit of GPUs per user reached" in html  # readable pending reason
    # Pending: Slurm's estimated start, about three hours from now.
    assert "est.</span> <time" in html
    assert re.search(r">in (2h 59m|3h 0m)<", html)
    assert "&lt;b&gt;other&lt;/b&gt;" in html and "<b>other</b>" not in html
    assert "--me" in queue_calls(app)[0]  # own jobs by default


def test_queue_partial_for_refreshing(app):
    response = app.test_client().get("/queue/jobs")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "<html" not in html and "#201" in html


def test_columns_follow_settings(app):
    client = app.test_client()
    html = client.get("/queue/").get_data(as_text=True)
    assert "<th>Partition</th>" in html and "<th>Priority</th>" not in html
    response = client.post("/queue/settings", headers=HEADERS, data={
        "QUEUE_COLUMNS": ["priority", "user"], "QUEUE_SCOPE": "all",
        "QUEUE_REFRESH": "0",
    })
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "Queue settings saved" in html
    assert "<th>Priority</th>" in html and "<th>Partition</th>" not in html
    assert 'hx-trigger="every' not in html  # refresh 0: no polling
    assert "--me" not in queue_calls(app)[-1]  # all users
    assert settings.read(app.config["SETTINGS_PATH"]) == {
        "QUEUE_COLUMNS": ["priority", "user"], "QUEUE_SCOPE": "all",
        "QUEUE_REFRESH": 0,
    }


def test_invalid_queue_settings_are_rejected(app):
    client = app.test_client()
    response = client.post("/queue/settings", headers=HEADERS, data={
        "QUEUE_COLUMNS": ["nonsense"], "QUEUE_SCOPE": "me", "QUEUE_REFRESH": "30",
    })
    assert response.status_code == 400
    assert "is not one of" in response.get_data(as_text=True)
    assert client.post("/queue/settings", data={}).status_code == 403


@pytest.mark.parametrize("hosts", ["-oProxyCommand=evil", "a;rm", "x$(y)", ",", ""])
def test_unsafe_or_empty_hosts_are_rejected(hosts):
    with pytest.raises(ValueError):
        settings.SETTINGS_BY_KEY["SLURM_HOSTS"].parse(hosts)


def test_hosts_are_editable_on_the_settings_page(app):
    setting = settings.SETTINGS_BY_KEY["SLURM_HOSTS"]
    assert setting.parse("local, me@login.example.org,") == [
        "local", "me@login.example.org",
    ]
    html = app.test_client().get("/settings/").get_data(as_text=True)
    assert 'name="SLURM_HOSTS"' in html and 'value="local"' in html


def test_check_hosts_reports_each_host(app, monkeypatch):
    app.config["SLURM_HOSTS"] = ["local", "login"]
    monkeypatch.setattr(
        "foreman.settings_page.check_host",
        lambda host: (host == "local", "slurm 25.05" if host == "local" else "denied"),
    )
    client = app.test_client()
    headers = {"X-Foreman-Request": "settings"}
    html = client.post("/settings/slurm/check", headers=headers).get_data(as_text=True)
    assert "slurm 25.05" in html and "denied" in html
    assert client.post("/settings/slurm/check").status_code == 403


def test_check_host_reports_local_failure(monkeypatch):
    monkeypatch.setenv("PATH", "/nonexistent")
    ok, message = slurm_module.check_host("local")
    assert not ok and "squeue" in message


def test_scope_selects_whose_jobs(monkeypatch):
    calls = []
    monkeypatch.setattr(
        slurm_module, "run_command", lambda host, args: calls.append(args) or "",
    )
    slurm = Slurm(["local"], cache_experiation=0)
    slurm.get_queue()
    slurm.get_queue(all_users=True)
    assert ["--me" in args for args in calls] == [True, False]
