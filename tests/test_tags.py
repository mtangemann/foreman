import re
from pathlib import Path

import pytest
import r3
from flask import Flask, render_template_string
from werkzeug.datastructures import MultiDict

import foreman.job as job_module
from foreman.job_list import job_list
from foreman.job_tree import job_tree
from foreman.search import search
from foreman.settings_page import settings_page
from foreman.templating import autoescape

TAG_HEADERS = {"X-Foreman-Request": "tags"}


@pytest.fixture
def app(repo, monkeypatch):
    repository, _ = repo
    monkeypatch.setenv("R3_REPOSITORY", str(repository.r3_repository.path))
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).resolve().parents[1] / "foreman/templates"),
    )
    app.jinja_env.autoescape = autoescape
    app.config.update(TESTING=True, SLURM_HOSTS=[], SLURM_CACHE_EXPIRATION=0)
    for blueprint in (job_module.job, job_list, job_tree, search, settings_page):
        app.register_blueprint(blueprint)
    return app


def test_job_details_offer_tag_editor(app, repo):
    _, ids = repo
    response = app.test_client().get(f"/job/view/{ids['a']}")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'id="tag-section"' in html
    assert "Edit tags" in html
    assert ">a</span>" in html


def test_tag_editor_loads_current_tags(app, repo):
    _, ids = repo
    response = app.test_client().get(f"/job/view/{ids['a']}/tags?edit=1")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'name="tags"' in html
    assert "One tag per line" in html
    assert ">a</textarea>" in html
    assert "Save tags" in html
    assert "Cancel" in html


def test_idle_tag_editor_does_not_announce_saving(app, repo):
    _, ids = repo
    response = app.test_client().get(f"/job/view/{ids['a']}/tags?edit=1")
    live_status = re.search(
        r'<span\b[^>]*\brole="status"[^>]*>(.*?)</span>',
        response.get_data(as_text=True),
        re.DOTALL,
    )
    assert live_status is not None
    assert live_status.group(1).strip() == ""


def test_save_tags_persists_normalized_list_and_updates_search(app, repo):
    repository, ids = repo
    job_id = ids["a"]
    before = dict(repository.get(job_id).r3_job.metadata)
    response = app.test_client().post(
        f"/job/view/{job_id}/tags",
        data={"tags": " baseline \r\nnew tag\n\nbaseline\nNEW\n", "name": "overwrite"},
        headers=TAG_HEADERS,
    )
    assert response.status_code == 200
    fresh = r3.Repository(repository.r3_repository.path)
    metadata = fresh.get_job_by_id(job_id).metadata
    assert metadata == {**before, "tags": ["baseline", "new tag", "NEW"]}
    assert [job.id for job in fresh.find({"tags": {"$all": ["new tag"]}})] == [job_id]
    assert fresh.find({"tags": {"$all": ["a"]}}) == []
    html = response.get_data(as_text=True)
    assert ">new tag</span>" in html
    assert 'role="status"' in html
    assert 'name="tags"' not in html


def test_save_empty_tags_clears_list_and_shows_empty_state(app, repo):
    repository, ids = repo
    response = app.test_client().post(
        f"/job/view/{ids['a']}/tags", data={"tags": " \n"}, headers=TAG_HEADERS
    )
    assert response.status_code == 200
    assert repository.get(ids["a"]).tags == []
    assert "No tags yet" in response.get_data(as_text=True)


def test_cancel_does_not_save_unsent_changes(app, repo):
    repository, ids = repo
    response = app.test_client().get(f"/job/view/{ids['a']}/tags")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'name="tags"' not in html
    assert ">a</span>" in html
    assert repository.get(ids["a"]).tags == ["a"]


@pytest.mark.parametrize("data", [{}, MultiDict([("tags", "one"), ("tags", "two")])])
def test_malformed_tag_form_does_not_write(app, repo, data):
    repository, ids = repo
    response = app.test_client().post(
        f"/job/view/{ids['a']}/tags", data=data, headers=TAG_HEADERS
    )
    assert response.status_code == 400
    assert 'role="alert"' in response.get_data(as_text=True)
    assert repository.get(ids["a"]).tags == ["a"]


def test_tag_write_requires_non_simple_request_header(app, repo):
    repository, ids = repo
    # Cross-origin HTML forms cannot supply this header; no CORS is enabled.
    response = app.test_client().post(
        f"/job/view/{ids['a']}/tags", data={"tags": "unwanted"}
    )
    assert response.status_code == 403
    assert repository.get(ids["a"]).tags == ["a"]


def test_tag_write_failure_keeps_editor_and_submitted_text(app, repo, monkeypatch):
    repository, ids = repo

    def fail_save(job):
        raise PermissionError("private repository location")

    monkeypatch.setattr(r3.Job, "save_metadata", fail_save)
    response = app.test_client().post(
        f"/job/view/{ids['a']}/tags",
        data={"tags": 'new\n</textarea><script>alert("tag")</script>'},
        headers=TAG_HEADERS,
    )
    assert response.status_code == 500
    html = response.get_data(as_text=True)
    assert 'role="alert"' in html
    assert 'name="tags"' in html
    assert "&lt;/textarea&gt;&lt;script&gt;" in html
    assert "<script>alert" not in html
    assert "private repository location" not in html
    assert repository.get(ids["a"]).tags == ["a"]


@pytest.mark.parametrize(
    "job_id", ["not-a-job", "00000000-0000-0000-0000-000000000000"],
)
@pytest.mark.parametrize("method", ["get", "post"])
def test_tag_editor_handles_missing_and_invalid_job_ids(app, job_id, method):
    response = getattr(app.test_client(), method)(
        f"/job/view/{job_id}/tags", headers=TAG_HEADERS
    )
    assert response.status_code == 404


def test_tags_are_escaped_on_details_editor_and_shared_tables(app, repo):
    repository, ids = repo
    dangerous_tag = '<img src=x onerror="alert(1)">'
    job = repository.get(ids["a"])
    job.r3_job.metadata["tags"] = [dangerous_tag]
    job.r3_job.save_metadata()
    client = app.test_client()
    details = client.get(f"/job/view/{job.id}").get_data(as_text=True)
    editor = client.get(f"/job/view/{job.id}/tags?edit=1").get_data(as_text=True)
    with app.test_request_context():
        table = render_template_string(
            '{% import "macros.html.jinja" as m %}{{ m.jobs_table(jobs) }}', jobs=[job]
        )
    for html in (details, editor, table):
        assert "<img src=x" not in html
        assert "&lt;img src=x" in html
    assert 'class="tag-badge ' in table


def test_metadata_file_viewer_escapes_tag_text(app, repo):
    repository, ids = repo
    job = repository.get(ids["a"])
    job.r3_job.metadata["tags"] = ["</pre><script>alert(1)</script>"]
    job.r3_job.save_metadata()
    response = app.test_client().get(f"/job/view/{job.id}/file/metadata.yaml")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;/pre&gt;&lt;script&gt;alert(1)&lt;/script&gt;" in html
