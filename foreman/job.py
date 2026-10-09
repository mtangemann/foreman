"""
Job blueprint.

Displays information about individual jobs.
"""
import base64
import sqlite3
from datetime import datetime

from flask import (
    Blueprint,
    abort,
    current_app,
    g,
    render_template,
    request,
    send_from_directory,
    url_for,
)

from foreman.job_stats import build_timeline, compute_stats, format_duration
from foreman.r3_extension import Repository
from foreman.repositories import branches_containing, repository_links
from foreman.settings import enabled
from foreman.tensorboard import has_event_files
from foreman.tracking_index import INDEX as tracking_index

job = Blueprint("job", __name__, url_prefix="/job")
job.add_app_template_filter(format_duration, "duration")


@job.app_template_global()
def tracking_links(job, fresh=False):
    """Resolve links once per job per request, including inside shared macros.

    Tables use indexed links; `fresh` (the job page) discovers them now.
    """
    if not hasattr(g, "tracking_links"):
        g.tracking_links = {}
    key = (str(job.path), fresh)
    if key not in g.tracking_links:
        settings = tracking_settings(current_app.config)
        links = []
        if settings["providers"]:
            links = tracking_index.links(job.path, fresh=fresh, **settings)
        # Filter as well, so that switching a provider off applies immediately
        # to links indexed before.
        g.tracking_links[key] = [
            link for link in links if link["provider"] in settings["providers"]
        ]
    return g.tracking_links[key]


def tracking_settings(config) -> dict:
    """Returns the tracking discovery settings from the app configuration."""
    return {
        "mlflow_ui_url": config.get("MLFLOW_UI_URL"),
        "wandb_ui_url": config.get("WANDB_UI_URL"),
        "providers": tuple(
            provider for provider, key in
            (("mlflow", "MLFLOW_ENABLED"), ("wandb", "WANDB_ENABLED"))
            if enabled(config, key)
        ),
        "scan_logs": enabled(config, "TRACKING_LOG_SCAN"),
    }


@job.app_template_global()
def tensorboard_available(job):
    """Cache lightweight event-file discovery only for the current request."""
    if not enabled(current_app.config, "TENSORBOARD_ENABLED"):
        return False
    if not hasattr(g, "tensorboard_available"):
        g.tensorboard_available = {}
    key = str(job.path)
    if key not in g.tensorboard_available:
        g.tensorboard_available[key] = has_event_files(job.path)
    return g.tensorboard_available[key]


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
        repositories=repository_links(job.r3_job),
    )


@job.route("/view/<id>/tags", methods=["GET", "POST"])
def tags(id: str):
    # A custom header prevents cross-origin HTML forms from submitting writes.
    # Foreman's same-origin HTMX form supplies it; this is not authentication.
    if request.method == "POST" and request.headers.get("X-Foreman-Request") != "tags":
        abort(403)

    repository = Repository.from_environment()
    try:
        job = repository.get(id)
    except (FileNotFoundError, ValueError):
        abort(404)

    if request.method == "GET":
        return render_template(
            "job/tags.html.jinja", job=job, editing=request.args.get("edit") == "1",
        )

    values = request.form.getlist("tags")
    tags_text = values[0] if values else ""
    error = None
    status = 400
    if len(values) != 1:
        error = "Submit tags as one list, with one tag per line."
    else:
        try:
            job = repository.update_tags(id, tags_text.splitlines())
        except ValueError as exception:
            error = str(exception)
        except (OSError, sqlite3.Error):
            current_app.logger.exception("Could not save tags for job %s", id)
            error = "Tags could not be fully saved. Try again or check the server logs."
            status = 500
        else:
            return render_template("job/tags.html.jinja", job=job, saved=True)

    return render_template(
        "job/tags.html.jinja", job=job, editing=True, tags_text=tags_text, error=error,
    ), status


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


@job.route("/view/<id>/dependencies/repositories")
def dependencies_repositories(id: str):
    """The job's git repositories, with the branches that contain their commits."""
    repository = Repository.from_environment()
    job = repository.get(id)
    links = repository_links(job.r3_job, repository.r3_repository.path)
    branches = {}
    if enabled(current_app.config, "REPOSITORY_BRANCHES"):
        branches = {
            link.repository: branches_containing(link.mirror, link.commit)
            for link in links if link.mirror and link.commit
        }
    return render_template(
        "job/_repository_list.html.jinja", repositories=links, branches=branches,
        branches_enabled=enabled(current_app.config, "REPOSITORY_BRANCHES"),
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
    jobs_root = (repository.r3_repository.path / "jobs").resolve()
    job_root = job.r3_job.path.resolve()
    complete_path = (job_root / path).resolve()
    if (
        not job_root.is_relative_to(jobs_root)
        or job_root == jobs_root
        or not complete_path.is_relative_to(job_root)
        or not complete_path.is_file()
    ):
        abort(404)

    extension = complete_path.suffix.lower().lstrip(".")
    image_types = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "gif": "image/gif",
        "webp": "image/webp",
    }
    if extension in image_types:
        with open(complete_path, "rb") as stream:
            content = base64.b64encode(stream.read()).decode("utf-8")

        return render_template(
            "job/file_image.html.jinja",
            content=content,
            mime_type=image_types[extension],
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
        "cfg": "ini",
        "csv": "csv",
        "ini": "ini",
        "json": "json",
        "log": "log",
        "md": "markdown",
        "qmd": "markdown",
        "py": "python",
        "sh": "shell",
        "toml": "plaintext",
        "tsv": "plaintext",
        "txt": "plaintext",
        "xml": "xml",
        "yaml": "yaml",
        "yml": "yaml",
    }

    if extension not in known_extensions:
        return render_template(
            "job/file_unknown.html.jinja",
            uuid=id,
            path=str(path).replace("/", " / "),
        )

    else:
        try:
            with open(complete_path, "r", encoding="utf-8") as stream:
                content = stream.read()
            if "\x00" in content:
                raise UnicodeError("Binary content")
        except UnicodeError:
            return render_template(
                "job/file_unknown.html.jinja",
                uuid=id,
                path=str(path).replace("/", " / "),
            )

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
