import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from flask import Flask
from PIL import Image

from foreman.job_list import job_list
from foreman.templating import autoescape


@pytest.fixture
def web_app(tmp_path, monkeypatch):
    module = importlib.import_module("foreman.job")
    root = tmp_path / "jobs" / "test-job"
    root.mkdir(parents=True)
    repository = SimpleNamespace(
        r3_repository=SimpleNamespace(path=tmp_path),
        get=lambda id: SimpleNamespace(r3_job=SimpleNamespace(path=root.parent / id)),
    )
    monkeypatch.setattr(module.Repository, "from_environment", lambda: repository)
    app = Flask(
        "preview-tests",
        template_folder=str(Path(module.__file__).parent / "templates"),
    )
    app.config["TESTING"] = True
    app.jinja_env.autoescape = autoescape
    app.register_blueprint(module.job)
    app.register_blueprint(job_list)  # file breadcrumbs link to the job list
    return app.test_client(), root


@pytest.mark.parametrize(
    ("extension", "language"),
    [
        ("json", "json"), ("txt", "plaintext"),
        ("toml", "plaintext"), ("ini", "ini"), ("cfg", "ini"),
        ("tsv", "plaintext"), ("xml", "xml"), ("csv", "csv"),
        ("log", "log"), ("md", "markdown"), ("py", "python"),
        ("sh", "shell"), ("yaml", "yaml"), ("yml", "yaml"),
        ("qmd", "markdown"),
    ],
)
@pytest.mark.parametrize("uppercase", [False, True])
def test_text_previews(web_app, extension, language, uppercase):
    client, root = web_app
    if uppercase:
        extension = extension.upper()
    (root / f"example.{extension}").write_text(
        'hello café <script>alert("test")</script>', encoding="utf-8"
    )
    response = client.get(f"/job/view/test-job/file/example.{extension}")
    assert response.status_code == 200
    text = response.get_data(as_text=True)
    assert f'language: "{language}"' in text
    assert "hello café" in text
    assert "&lt;script&gt;" in text
    assert '<script>alert("test")</script>' not in text
    assert "readOnly: true" in text


@pytest.mark.parametrize(
    ("extension", "format", "mime"),
    [
        ("png", "PNG", "image/png"), ("jpg", "JPEG", "image/jpeg"),
        ("jpeg", "JPEG", "image/jpeg"), ("gif", "GIF", "image/gif"),
        ("webp", "WEBP", "image/webp"),
    ],
)
@pytest.mark.parametrize("uppercase", [False, True])
def test_image_previews(web_app, extension, format, mime, uppercase):
    client, root = web_app
    if uppercase:
        extension = extension.upper()
    Image.new("RGB", (2, 2)).save(root / f"image.{extension}", format=format)
    response = client.get(f"/job/view/test-job/file/image.{extension}")
    assert response.status_code == 200
    assert f"data:{mime};base64," in response.get_data(as_text=True)


@pytest.mark.parametrize("content", [b"\xff\xfe", b"binary\x00data"])
def test_binary_text_falls_back(web_app, content):
    client, root = web_app
    (root / "binary.txt").write_bytes(content)
    response = client.get("/job/view/test-job/file/binary.txt")
    assert response.status_code == 200
    assert "preview this type of file" in response.get_data(as_text=True)


def test_unknown_missing_and_outside_files(web_app):
    client, root = web_app
    (root / "weights.bin").write_bytes(b"\x00")
    assert "preview this type of file" in client.get(
        "/job/view/test-job/file/weights.bin"
    ).get_data(as_text=True)
    assert client.get("/job/view/test-job/file/missing.txt").status_code == 404
    outside = root.parent / "outside.txt"
    outside.write_text("private")
    (root / "linked.txt").symlink_to(outside)
    assert client.get("/job/view/test-job/file/linked.txt").status_code == 404
    assert client.get("/job/view/test-job/file/../outside.txt").status_code == 404


def test_job_root_boundary(web_app):
    client, root = web_app
    (root.parent / "outside.txt").write_text("private")
    assert client.get("/job/view/./file/outside.txt").status_code == 404
    assert client.get("/job/view/../file/outside.txt").status_code == 404
    outside_root = root.parent.parent / "outside-job"
    outside_root.mkdir()
    (outside_root / "private.txt").write_text("private")
    (root.parent / "linked-job").symlink_to(outside_root, target_is_directory=True)
    assert client.get("/job/view/linked-job/file/private.txt").status_code == 404


def test_html_and_metadata_behavior(web_app):
    client, root = web_app
    (root / "page.html").write_text("<h1>hello</h1>")
    assert "/job/file_contents/test-job/page.html" in client.get(
        "/job/view/test-job/file/page.html"
    ).get_data(as_text=True)
    (root / "page.HTML").write_text("<h1>hello</h1>")
    assert "preview this type of file" in client.get(
        "/job/view/test-job/file/page.HTML"
    ).get_data(as_text=True)
    (root / "metadata.yaml").write_text("name: example")
    assert "readOnly: false" in client.get(
        "/job/view/test-job/file/metadata.yaml"
    ).get_data(as_text=True)


def test_nested_report_link(web_app):
    client, root = web_app
    (root / "output").mkdir()
    (root / "output" / "report.html").write_text("<h1>report</h1>")
    response = client.get("/job/view/test-job/file/output/report.html")
    assert response.status_code == 200
    assert "/job/file_contents/test-job/output/report.html" in response.get_data(
        as_text=True
    )
    report = client.get("/job/file_contents/test-job/output/report.html")
    assert report.status_code == 200
    assert report.get_data(as_text=True) == "<h1>report</h1>"
