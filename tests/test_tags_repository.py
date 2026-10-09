import sqlite3
from copy import deepcopy

import pytest
import r3

from foreman.r3_extension import Job


def repository_snapshot(repository):
    """Capture persisted files and index rows without relying on cached jobs."""
    root = repository.r3_repository.path
    files = {
        str(path.relative_to(root)): path.read_bytes()
        for path in (root / "jobs").rglob("*")
        if path.is_file()
    }
    with sqlite3.connect(root / "index.sqlite") as connection:
        jobs = connection.execute("SELECT * FROM jobs ORDER BY id").fetchall()
        dependencies = connection.execute(
            "SELECT * FROM job_dependencies ORDER BY child_id, parent_id"
        ).fetchall()
    return files, jobs, dependencies


@pytest.mark.parametrize(
    ("tags", "expected"),
    [
        (["b", "new"], ["b", "new"]),
        (["new"], ["new"]),
        ([], []),
        (
            [" b ", "", "  ", "new", "b", "New", "\tnew\t", "two words"],
            ["b", "new", "New", "two words"],
        ),
    ],
    ids=["add", "rename", "clear", "normalize"],
)
def test_update_tags_persists_and_refreshes_search(repo, tags, expected):
    repository, ids = repo
    result = repository.update_tags(ids["b"], tags)

    assert isinstance(result, Job)
    assert result.tags == expected
    assert result.status == "completed"
    assert repository.get(ids["b"]).tags == expected
    fresh = r3.Repository(repository.r3_repository.path)
    assert fresh.find({"path": "test/b"})[0].metadata["tags"] == expected
    for tag in ["b", "new", "New", "two words"]:
        matches = {job.id for job in fresh.find({"tags": tag})}
        assert (ids["b"] in matches) == (tag in expected)


def test_update_tags_removes_one_tag(repo):
    repository, ids = repo
    repository.update_tags(ids["b"], ["b", "keep"])

    result = repository.update_tags(ids["b"], ["keep"])

    assert result.tags == ["keep"]
    fresh = r3.Repository(repository.r3_repository.path)
    assert fresh.find({"tags": "b"}) == []
    assert [job.id for job in fresh.find({"tags": "keep"})] == [ids["b"]]


def test_update_tags_preserves_identity_files_and_dependency_graph(repo):
    repository, ids = repo
    before = repository.get(ids["b"])
    metadata = deepcopy(before.r3_job.metadata)
    before_hash = before.r3_job.hash(recompute=True)
    files, rows, dependencies = repository_snapshot(repository)

    result = repository.update_tags(ids["b"], ["changed"])

    assert result.id == before.id
    assert result.path == before.path
    assert result.timestamp == before.timestamp
    assert result.r3_job.hash(recompute=True) == before_hash
    assert result.r3_job.metadata == {**metadata, "tags": ["changed"]}
    after_files, after_rows, after_dependencies = repository_snapshot(repository)
    metadata_path = str((before.path / "metadata.yaml").relative_to(
        repository.r3_repository.path
    ))
    assert {k: v for k, v in files.items() if k != metadata_path} == {
        k: v for k, v in after_files.items() if k != metadata_path
    }
    assert [row for row in rows if row[0] != ids["b"]] == [
        row for row in after_rows if row[0] != ids["b"]
    ]
    assert len(rows) == len(after_rows) == 4
    assert after_dependencies == dependencies
    assert {job.id for job in repository.find_dependents(result)} == {
        ids["c"], ids["d"]
    }


def test_update_tags_loads_metadata_from_storage_not_search_cache(repo):
    repository, ids = repo
    cached = repository.find({"tags": "b"})[0]
    stored = repository.get(ids["b"]).r3_job
    stored.metadata.update({
        "tags": ["on-disk"],
        "name": "fresh name",
        "description": "fresh description",
        "origin": "fresh origin",
        "custom": {"nested": [1, "two"]},
    })
    stored.save_metadata()
    expected = deepcopy(stored.metadata)
    assert cached.tags == ["b"]
    assert repository.find({"tags": "b"})[0].tags == ["b"]

    result = repository.update_tags(ids["b"], ["edited"])

    assert result.r3_job.metadata == {**expected, "tags": ["edited"]}
    assert repository.get(ids["b"]).r3_job.metadata == result.r3_job.metadata
    fresh = r3.Repository(repository.r3_repository.path)
    assert fresh.find({"path": "test/b"})[0].metadata == result.r3_job.metadata
    assert fresh.find({"tags": "b"}) == []
    assert fresh.find({"tags": "on-disk"}) == []


@pytest.mark.parametrize(
    "tags",
    [
        None, "tag", ("tag",), {"tag"}, {"tags": ["tag"]},
        [1], ["valid", None], ["valid", ["nested"]],
        ["valid", "two\nlines"], ["valid", "two\rlines"],
        ["valid", "two\r\nlines"], ["\ntrimmed"], ["trimmed\r"],
    ],
)
def test_update_tags_rejects_invalid_input_without_persistence(repo, tags):
    repository, ids = repo
    before = repository_snapshot(repository)

    with pytest.raises(ValueError):
        repository.update_tags(ids["b"], tags)

    assert repository_snapshot(repository) == before


@pytest.mark.parametrize(
    ("job_id", "error"),
    [
        ("00000000-0000-0000-0000-000000000000", FileNotFoundError),
        ("not-a-job-id", ValueError),
        ("../jobs", ValueError),
        ("*", ValueError),
        ("", ValueError),
        (None, ValueError),
    ],
)
def test_update_tags_rejects_missing_or_invalid_id(repo, job_id, error):
    repository, _ = repo
    before = repository_snapshot(repository)

    with pytest.raises(error):
        repository.update_tags(job_id, ["valid"])

    assert repository_snapshot(repository) == before
