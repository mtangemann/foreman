from pathlib import Path

import pytest
from flask import Flask

from foreman.job import job
from foreman.job_list import job_list
from foreman.job_tree import job_tree
from foreman.search import search
from foreman.settings_page import settings_page
from foreman.templating import autoescape


@pytest.fixture
def client(repo, monkeypatch):
    repository, _ = repo
    monkeypatch.setenv("R3_REPOSITORY", str(repository.r3_repository.path))
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).resolve().parents[1] / "foreman/templates"),
    )
    # No Slurm hosts: pages must render without any Slurm information.
    app.jinja_env.autoescape = autoescape
    app.config.update(TESTING=True, SLURM_HOSTS=[], SLURM_CACHE_EXPIRATION=0)
    for blueprint in (job, job_list, job_tree, search, settings_page):
        app.register_blueprint(blueprint)
    return app.test_client()


@pytest.mark.parametrize("url", [
    "/list/",
    "/job/view/{a}",
    "/job/view/{a}/dependencies",
    "/job/view/{b}/dependencies/upstream?recursive=true",
    "/job/view/{a}/directory/",
    "/job/view/{a}/file/metadata.yaml",
    "/job/status/{a}",
    "/tree/view",
    "/tree/view/test",
    "/tree/view_revisions/test/a",
    "/tree/directory_tree",
    "/search/search",
])
def test_page_renders(client, repo, url):
    _, ids = repo
    response = client.get(url.format(**ids))
    assert response.status_code == 200, response.get_data(as_text=True)


def test_job_page_without_slurm_records_says_so(client, repo):
    _, ids = repo
    html = client.get(f"/job/view/{ids['a']}").get_data(as_text=True)
    assert "No Slurm records found" in html
    assert "Committed to R3" in html
    # Title is the last segment of the virtual path, with its folder above.
    assert ">a</h1>" in html and "test/</p>" in html


def test_metadata_is_escaped(client, repo):
    repository, ids = repo
    job = repository.get(ids["a"])
    job.r3_job.metadata["description"] = '<script>alert("x")</script>'
    job.r3_job.metadata["tags"] = ['"><img src=x>']
    job.r3_job.save_metadata()
    for url in (f"/job/view/{job.id}", "/list/"):
        html = client.get(url).get_data(as_text=True)
        assert "<script>alert" not in html and "<img src=x>" not in html
    assert "&lt;script&gt;" in client.get(f"/job/view/{job.id}").get_data(as_text=True)


def test_search_errors_are_escaped(client):
    # YAML errors quote the offending input.
    response = client.post("/search/search", data={"query": "<b>: [", "format": "yaml"})
    html = response.get_data(as_text=True)
    assert 'role="alert"' in html and "Invalid YAML query" in html
    assert "&lt;b&gt;: [" in html and "<b>" not in html
