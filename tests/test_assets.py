import logging
import os

import pytest

from foreman.assets import stale_stylesheet_reason, warn_if_stylesheet_stale


@pytest.fixture
def package(tmp_path):
    """A minimal package layout with a stylesheet built after its sources."""
    (tmp_path / "static").mkdir()
    (tmp_path / "templates" / "job").mkdir(parents=True)
    for path in ["static/input.css", "static/tailwind.config.js",
                 "templates/base.html.jinja", "templates/job/view.html.jinja"]:
        (tmp_path / path).write_text("")
        os.utime(tmp_path / path, (1000, 1000))
    (tmp_path / "static/output.css").write_text("")
    os.utime(tmp_path / "static/output.css", (2000, 2000))
    return tmp_path


def test_fresh_stylesheet_is_not_stale(package):
    assert stale_stylesheet_reason(package) is None


def test_missing_stylesheet_is_stale(package):
    (package / "static/output.css").unlink()
    assert "does not exist" in stale_stylesheet_reason(package)


@pytest.mark.parametrize(
    "source", ["templates/job/view.html.jinja", "static/tailwind.config.js"]
)
def test_source_changed_after_build_is_stale(package, source):
    os.utime(package / source, (3000, 3000))
    assert source in stale_stylesheet_reason(package)


def test_stale_stylesheet_logs_rebuild_hint(package, caplog):
    os.utime(package / "templates/base.html.jinja", (3000, 3000))
    with caplog.at_level(logging.WARNING, logger="foreman.assets"):
        warn_if_stylesheet_stale(package)
    assert "make css" in caplog.text
