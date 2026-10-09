import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import r3
import yaml
from flask import Flask

from foreman import repositories
from foreman.blueprints import BLUEPRINTS
from foreman.repositories import branches_containing, repository_links
from foreman.templating import autoescape

COMMIT = "bb16a40e46e615a7d57206072703586dd673c460"


def git_job(*dependencies):
    return SimpleNamespace(dependencies=list(dependencies))


@pytest.mark.parametrize("url", [
    "git@github.com:enigma-brain/ntt.git",
    "https://github.com/enigma-brain/ntt.git",
    "https://github.com/enigma-brain/ntt",
])
def test_github_links(url):
    (link,) = repository_links(git_job(
        r3.GitDependency("code", url, commit=COMMIT, source="src/model"),
        r3.JobDependency("data", "some-job-id"),
    ))
    assert link.name == "enigma-brain/ntt"
    assert link.web_url == "https://github.com/enigma-brain/ntt"
    assert link.commit_url == f"https://github.com/enigma-brain/ntt/commit/{COMMIT}"
    assert link.tree_url == (
        f"https://github.com/enigma-brain/ntt/tree/{COMMIT}/src/model"
    )
    assert link.compare_url("main") == (
        f"https://github.com/enigma-brain/ntt/compare/{COMMIT}...main"
    )
    assert link.short_commit == "bb16a40" and link.destination == "code"


def test_whole_repository_and_unresolved_and_unknown_hosts():
    whole, unresolved = repository_links(git_job(
        r3.GitDependency("code", "git@github.com:o/r.git", commit=COMMIT, source="."),
        r3.GitDependency("other", "git@github.com:o/r.git", branch="main"),
    ))
    assert whole.source == "" and whole.tree_url.endswith(f"/tree/{COMMIT}")
    assert unresolved.commit is None and unresolved.tree_url is None
    unknown = repositories.RepositoryLink(
        "git@gitlab.com:o/r.git", COMMIT, "", "x", "git@gitlab.com:o/r.git", None, None,
    )
    assert unknown.commit_url is None and unknown.tree_url is None


def git(cwd, *args):
    subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True)


def make_mirror(root):
    """A bare mirror of o/r: `old` is on main and feature, `new` only on feature."""
    work = root.parent / "work"
    work.mkdir()
    git(work, "init", "-q", "-b", "main")
    commit = ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q",
              "--allow-empty"]
    git(work, *commit, "-m", "old")
    git(work, "branch", "feature")
    git(work, "checkout", "-q", "feature")
    git(work, *commit, "-m", "new")
    git(work, "checkout", "-q", "main")  # the mirror's default branch
    path = Path(root) / "git" / "github.com" / "o" / "r"
    subprocess.run(["git", "clone", "-q", "--bare", str(work), str(path)], check=True)

    def rev(name):
        return subprocess.run(["git", "-C", str(work), "rev-parse", name],
                              check=True, capture_output=True, text=True).stdout.strip()

    return path, rev("main"), rev("feature")


def test_branches_containing_lists_default_branch_first(tmp_path):
    path, old, new = make_mirror(tmp_path / "r3")
    assert branches_containing(path, old) == ["main", "feature"]
    assert branches_containing(path, new) == ["feature"]
    assert branches_containing(path, "0" * 40) == []  # unknown commit


def test_mirror_is_found_in_the_r3_repository(tmp_path):
    path, old, _ = make_mirror(tmp_path / "r3")
    (link,) = repository_links(
        git_job(r3.GitDependency("code", "git@github.com:o/r.git", commit=old)),
        tmp_path / "r3",
    )
    assert link.mirror == path


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
    for blueprint in BLUEPRINTS:
        app.register_blueprint(blueprint)
    return app


def add_git_dependency(repository, job_id, commit=COMMIT):
    """Adds a git dependency to a committed job's r3.yaml."""
    config_path = Path(repository.get(job_id).path) / "r3.yaml"
    config = yaml.safe_load(config_path.read_text()) or {}
    config.setdefault("dependencies", []).append({
        "repository": "git@github.com:o/r.git", "commit": commit,
        "source": ".", "destination": "code",
    })
    config_path.chmod(0o644)  # R3 commits files read-only
    config_path.write_text(yaml.safe_dump(config))


def test_sidebar_and_dependencies_tab(app, repo):
    repository, ids = repo
    add_git_dependency(repository, ids["a"])
    client = app.test_client()

    html = client.get(f"/job/view/{ids['a']}").get_data(as_text=True)
    assert "Repositories" in html
    assert f'href="https://github.com/o/r/tree/{COMMIT}"' in html
    assert ">bb16a40<" in html

    deps = client.get(f"/job/view/{ids['a']}/dependencies").get_data(as_text=True)
    assert "/dependencies/repositories" in deps
    url = f"/job/view/{ids['a']}/dependencies/repositories"
    html = client.get(url).get_data(as_text=True)
    assert f"https://github.com/o/r/commit/{COMMIT}" in html
    assert "No local mirror" in html


def test_branch_lookup_can_be_switched_off(app, repo, monkeypatch):
    repository, ids = repo
    _, old, _ = make_mirror(repository.r3_repository.path)
    add_git_dependency(repository, ids["a"], commit=old)
    client = app.test_client()
    url = f"/job/view/{ids['a']}/dependencies/repositories"
    html = client.get(url).get_data(as_text=True)
    assert ">main<" in html and f"compare/{old}...main" in html

    app.config["REPOSITORY_BRANCHES"] = False
    monkeypatch.setattr(
        "foreman.job.branches_containing",
        lambda *args: pytest.fail("branches were looked up"),
    )
    assert "switched off in the settings" in client.get(url).get_data(as_text=True)


def test_jobs_without_repositories(app, repo):
    _, ids = repo
    client = app.test_client()
    url = f"/job/view/{ids['a']}/dependencies/repositories"
    assert "does not depend on any git" in client.get(url).get_data(as_text=True)
    page = client.get(f"/job/view/{ids['a']}").get_data(as_text=True)
    assert "Repositories" not in page
