import importlib
from datetime import datetime
from html import unescape
from html.parser import HTMLParser
from types import SimpleNamespace

import pytest


@pytest.fixture
def views(tmp_path, monkeypatch):
    monkeypatch.setenv("R3_REPOSITORY", str(tmp_path))
    module = importlib.import_module("foreman.app")
    job_module = importlib.import_module("foreman.job")
    monkeypatch.setitem(module.app.config, "TESTING", True)
    monkeypatch.setitem(module.app.config, "MLFLOW_UI_URL", None)
    monkeypatch.setitem(module.app.config, "WANDB_UI_URL", None)
    monkeypatch.setitem(module.app.config, "SLURM_HOSTS", [])
    root = tmp_path / "jobs" / "test-job"
    root.mkdir(parents=True)
    job = SimpleNamespace(
        id="test-job", path=root, timestamp=datetime(2024, 1, 1),
        status="done", tags=[], leaf_tags=[], slurm_records=[],
        virtual_path="folder/example",
        r3_job=SimpleNamespace(path=root),
        has_report=True, name="Example", description=None, origin=None,
    )
    repository = SimpleNamespace(
        r3_repository=SimpleNamespace(path=tmp_path),
        jobs=lambda: [job], get=lambda _: job, find=lambda _: [job],
        find_dependencies=lambda _, recursive=False: [job],
        find_dependents=lambda _, recursive=False: [job],
    )
    monkeypatch.setattr(
        job_module.Repository, "from_environment", lambda: repository
    )
    return module.app.test_client(), root


@pytest.mark.parametrize(
    "url", ["/list/", "/tree/view/folder", "/tree/view_revisions/folder/example",
            "/job/view/test-job", "/search/search"]
)
def test_tracking_buttons_across_views(views, url):
    client, root = views
    (root / "metadata.yaml").write_text(
        "tracking:\n"
        "  mlflow:\n"
        "    url: https://mlflow.example/#/experiments/1/runs/abc\n"
        "  wandb:\n"
        "    url: https://wandb.ai/team/project/runs/xyz\n"
    )
    response = (
        client.post(url, data={"query": "{}", "format": "json"})
        if url == "/search/search"
        else client.get(url)
    )
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'href="https://mlflow.example/#/experiments/1/runs/abc"' in html
    assert 'href="https://wandb.ai/team/project/runs/xyz"' in html
    assert "logos/mlflow.svg" in html
    assert "logos/wandb.svg" in html
    assert 'rel="noopener noreferrer"' in html
    assert 'target="_blank"' in html
    assert "/tensorboard/open/test-job" not in html


class TensorBoardLinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.current = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "a" and attrs.get("href") == "/tensorboard/open/test-job":
            self.current = {"attrs": attrs, "text": "", "images": []}
            self.links.append(self.current)
        elif tag == "img" and self.current is not None:
            self.current["images"].append(attrs)

    def handle_data(self, data):
        if self.current is not None:
            self.current["text"] += data

    def handle_endtag(self, tag):
        if tag == "a":
            self.current = None


@pytest.mark.parametrize("has_events", [False, True], ids=["absent", "present"])
@pytest.mark.parametrize(
    "url,headers",
    [
        ("/list/", {}),
        ("/tree/view/folder", {}),
        ("/tree/view_revisions/folder/example", {}),
        ("/job/view/test-job", {}),
        ("/search/search", {}),
        ("/job/view/test-job/dependencies/upstream", {}),
        ("/job/view/test-job/dependencies/downstream", {}),
        ("/tree/view/folder", {"Hx-Target": "contents"}),
    ],
    ids=["list", "tree", "revisions", "job", "search",
         "upstream", "downstream", "htmx-tree"],
)
def test_tensorboard_link_requires_local_events(views, url, headers, has_events):
    client, root = views
    if has_events:
        output = root / "output" / "train"
        output.mkdir(parents=True)
        # Detection is filename/stat based; this is not a parsed event stream.
        (output / "events.out.tfevents.1704067200.test").write_bytes(b"marker")
    response = (
        client.post(url, data={"query": "{}", "format": "json"}, headers=headers)
        if url == "/search/search"
        else client.get(url, headers=headers)
    )
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    parser = TensorBoardLinkParser()
    parser.feed(html)
    if not has_events:
        assert not parser.links
        assert "logos/tensorboard.svg" not in html
        return
    assert len(parser.links) == 1
    link = parser.links[0]
    assert link["attrs"]["target"] == "_blank"
    assert set(link["attrs"]["rel"].split()) == {"noopener", "noreferrer"}
    assert link["text"].strip() == "TensorBoard"
    assert len(link["images"]) == 1
    assert link["images"][0]["src"] == "/static/logos/tensorboard.svg"
    assert link["images"][0]["alt"] == ""


def test_tensorboard_logo_is_served_locally(views):
    client, _ = views
    response = client.get("/static/logos/tensorboard.svg")
    assert response.status_code == 200
    assert response.mimetype == "image/svg+xml"


@pytest.mark.parametrize("has_tracking", [False, True])
def test_absent_tensorboard_leaves_no_empty_sidebar_row(views, has_tracking):
    client, root = views
    if has_tracking:
        (root / "metadata.yaml").write_text(
            "tracking:\n  wandb:\n    url: https://wandb.ai/team/project/runs/xyz\n"
        )

    class SidebarRows(HTMLParser):
        def __init__(self):
            super().__init__()
            self.in_sidebar = False
            self.current = None
            self.rows = []

        def handle_starttag(self, tag, attrs):
            if tag == "aside" and dict(attrs).get("id") == "logo-sidebar":
                self.in_sidebar = True
            elif tag == "li" and self.in_sidebar:
                self.current = ""

        def handle_data(self, data):
            if self.current is not None:
                self.current += data

        def handle_endtag(self, tag):
            if tag == "li" and self.current is not None:
                self.rows.append(self.current.strip())
                self.current = None
            elif tag == "aside":
                self.in_sidebar = False

    parser = SidebarRows()
    parser.feed(client.get("/job/view/test-job").get_data(as_text=True))
    assert parser.rows
    assert all(parser.rows), "Hidden controls must not leave empty sidebar rows"


def test_unresolved_tracking_is_hidden(views):
    client, _ = views
    html = client.get("/list/").get_data(as_text=True)
    assert "logos/mlflow.svg" not in html
    assert "logos/wandb.svg" not in html


def test_multiple_runs_have_chooser(views):
    client, root = views
    (root / "metadata.yaml").write_text(
        "tracking:\n"
        "  wandb:\n"
        "    - url: https://wandb.ai/team/project/runs/one\n"
        "    - url: https://wandb.ai/team/project/runs/two\n"
    )
    html = unescape(client.get("/list/").get_data(as_text=True))
    assert "<details" in html
    assert "W&B (2 runs)" in html
    assert "runs/one" in html
    assert "runs/two" in html


def test_unsafe_tracking_url_is_never_rendered(views):
    client, root = views
    (root / "metadata.yaml").write_text(
        "tracking:\n  wandb:\n    url: javascript:alert(document.domain)\n"
    )
    html = client.get("/list/").get_data(as_text=True)
    assert "javascript:" not in html
    assert "logos/wandb.svg" not in html


def test_htmx_tree_preserves_tracking_and_report_links(views):
    client, root = views
    (root / "metadata.yaml").write_text(
        "tracking:\n  wandb:\n    url: https://wandb.ai/team/project/runs/xyz\n"
    )
    html = client.get(
        "/tree/view/folder", headers={"Hx-Target": "contents"}
    ).get_data(as_text=True)
    assert "logos/wandb.svg" in html
    assert "report.html" in html
    assert 'draggable="true"' in html


def test_root_redirects_to_blueprint_list(views):
    client, _ = views
    response = client.get("/")
    assert response.status_code == 302
    assert response.headers["Location"].endswith("/list/")


def test_yaml_search_keeps_tracking_links(views):
    client, root = views
    (root / "metadata.yaml").write_text(
        "tracking:\n  wandb:\n    url: https://wandb.ai/team/project/runs/xyz\n"
    )
    response = client.post(
        "/search/search", data={"query": "path: folder/example", "format": "yaml"}
    )
    assert response.status_code == 200
    assert "logos/wandb.svg" in response.get_data(as_text=True)


@pytest.mark.parametrize("direction", ["upstream", "downstream"])
def test_dependency_tables_keep_tracking_links(views, direction):
    client, root = views
    (root / "metadata.yaml").write_text(
        "tracking:\n  wandb:\n    url: https://wandb.ai/team/project/runs/xyz\n"
    )
    response = client.get(f"/job/view/test-job/dependencies/{direction}")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "logos/wandb.svg" in html
    assert "report.html" in html


def test_application_escapes_jinja_file_previews(views):
    client, root = views
    (root / "notes.txt").write_text("<script>alert('artifact')</script>")
    response = client.get("/job/view/test-job/file/notes.txt")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "&lt;script&gt;" in html
    assert "<script>alert('artifact')</script>" not in html
