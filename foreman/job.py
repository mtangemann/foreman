"""
Job blueprint.

Displays information about individual jobs.
"""
import base64
from datetime import datetime

from flask import Blueprint, render_template, request, send_from_directory, url_for

from foreman.job_stats import build_timeline, compute_stats, format_duration
from foreman.r3_extension import Repository

job = Blueprint("job", __name__, url_prefix="/job")
job.add_app_template_filter(format_duration, "duration")


@job.route("/view/<id>")
def view(id: str):
    repository = Repository.from_environment()
    job = repository.get(id)
    records = job.slurm_records
    now = datetime.now()
    return render_template(
        "job/view.html.jinja",
        job=job,
        records=records,
        stats=compute_stats(job.timestamp, records, now),
        timeline=build_timeline(job.timestamp, records, now),
        open_file=request.args.get("file"),
    )


@job.route("/view/<id>/dependencies")
def dependencies(id: str):
    repository = Repository.from_environment()
    job = repository.get(id)
    return render_template("job/dependencies.html.jinja", job=job)


@job.route("/view/<id>/dependencies/upstream")
def dependencies_upstream(id: str):
    repository = Repository.from_environment()
    job = repository.get(id)
    recursive = request.args.get("recursive") == "true"
    jobs = repository.find_dependencies(job, recursive=recursive)
    return render_template(
        "job/_dependency_list.html.jinja",
        job=job, jobs=jobs, direction="upstream",
        recursive=recursive, table_id="deps-upstream",
    )


@job.route("/view/<id>/dependencies/downstream")
def dependencies_downstream(id: str):
    repository = Repository.from_environment()
    job = repository.get(id)
    recursive = request.args.get("recursive") == "true"
    jobs = repository.find_dependents(job, recursive=recursive)
    return render_template(
        "job/_dependency_list.html.jinja",
        job=job, jobs=jobs, direction="downstream",
        recursive=recursive, table_id="deps-downstream",
    )


@job.route("/status/<id>")
def status(id: str):
    repository = Repository.from_environment()
    job = repository.get(id)
    return render_template("job/status.html.jinja", job=job)


@job.route("/view/<id>/directory/")
@job.route("/view/<id>/directory/<path:path>")
def view_directory(id: str, path: str = ""):
    repository = Repository.from_environment()
    job = repository.get(id)
    complete_path = job.r3_job.path / path

    entries = []

    for child in complete_path.iterdir():
        if child.is_dir():
            url = url_for("job.view_directory", id=id, path=f"{path}/{child.name}")
        else:
            url = url_for("job.view_file", id=id, path=f"{path}/{child.name}")

        entries.append({ "name": child.name, "is_dir": child.is_dir(), "url": url })

    entries = sorted(
        entries,
        key = lambda entry: (not entry["is_dir"], entry["name"].lower())  # type: ignore
    )

    return render_template(
        "job/sidebar_files.html.jinja",
        entries=entries,
        name=complete_path.name,
        is_root = path == "",
    )


@job.route("/view/<id>/file/<path:path>")
def view_file(id: str, path: str = ""):
    repository = Repository.from_environment()
    job = repository.get(id)
    complete_path = job.r3_job.path / path

    if path.endswith(".png"):
        with open(complete_path, "rb") as stream:
            content = base64.b64encode(stream.read()).decode("utf-8")

        return render_template(
            "job/file_image.html.jinja",
            content=content,
            uuid=id,
            path=str(path).replace("/", " / "),
        )

    if path.endswith(".html"):
        return render_template(
            "job/file_website.html.jinja",
            content_path=url_for("job.file_contents", path=f"{id}/{path}"),
            uuid=id,
            path=str(path).replace("/", " / "),
        )

    known_extensions = {
        "csv": "csv",
        "json": "json",
        "log": "log",
        "md": "markdown",
        "qmd": "markdown",
        "py": "python",
        "sh": "shell",
        "yaml": "yaml",
        "yml": "yaml",
    }

    extension = path.rsplit(".", maxsplit=1)[-1]

    if extension not in known_extensions:
        return render_template(
            "job/file_unknown.html.jinja",
            uuid=id,
            path=str(path).replace("/", " / "),
        )

    else:
        with open(complete_path, "r") as stream:
            content = stream.read()

        return render_template(
            "job/file_text.html.jinja",
            content=content,
            language=known_extensions[extension],
            read_only=(path != "metadata.yaml"),
            uuid=id,
            path=str(path).replace("/", " / "),
        )


@job.route("/file_contents/<path:path>")
def file_contents(path):
    repository = Repository.from_environment()
    return send_from_directory(repository.r3_repository.path / "jobs", path)
