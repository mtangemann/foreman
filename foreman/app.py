import logging
import os
import re
from pathlib import Path

import click
from flask import Flask, abort, redirect, url_for
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.serving import run_simple

from foreman.assets import warn_if_stylesheet_stale
from foreman.job import job, tracking_settings
from foreman.job_list import job_list
from foreman.job_tree import job_tree
from foreman.search import search
from foreman.templating import autoescape
from foreman.tensorboard import TensorBoardViewers, has_event_files
from foreman.tracking_index import INDEX as tracking_index
from foreman.tracking_index import default_cache_dir

R3_REPOSITORY = Path(os.environ["R3_REPOSITORY"])

logging.basicConfig(level=logging.INFO)
warn_if_stylesheet_stale()

app = Flask(
    "app",
    template_folder=Path(__file__).parent / "templates",
    static_folder=Path(__file__).parent / "static",
)
app.jinja_env.autoescape = autoescape
app.register_blueprint(job)
app.register_blueprint(job_list)
app.register_blueprint(job_tree)
app.register_blueprint(search)

joint_app = DispatcherMiddleware(app)
tensorboard_viewers = TensorBoardViewers(joint_app.mounts)

app.config.from_object("foreman.config")
# Optional machine-specific overrides in foreman/config_local.py (gitignored).
app.config.from_pyfile(Path(__file__).parent / "config_local.py", silent=True)



@click.command()
@click.option("--host", type=str, default="localhost")
@click.option("--port", type=int, default=5000)
@click.option("--debug", type=bool, default=False)
def main(host, port, debug):
    # With the reloader, only the child process (WERKZEUG_RUN_MAIN) serves requests.
    if not debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        start_tracking_index()
    run_simple(
        host,
        port,
        joint_app,
        use_reloader=debug,
        use_debugger=debug,
        use_evalex=debug,
    )


def start_tracking_index():
    cache_dir = app.config["TRACKING_CACHE_DIR"]
    if cache_dir:
        tracking_index.use_disk_cache(
            default_cache_dir() if cache_dir == "default" else Path(cache_dir)
        )
    interval = app.config["TRACKING_REFRESH_INTERVAL"]
    if interval:
        tracking_index.start(
            R3_REPOSITORY / "jobs", interval, **tracking_settings(app.config)
        )


@app.route("/")
def index():
    return redirect(url_for("job_list.view"))


@app.route("/tensorboard/open/<uuid>")
def open_in_tensorboard(uuid):
    # Reject traversal and the reserved mount name that would shadow this route.
    if not re.fullmatch(r"[A-Za-z0-9_-]+", uuid) or uuid == "open":
        abort(404)
    try:
        jobs_root = (R3_REPOSITORY / "jobs").resolve()
        candidate = jobs_root / uuid
        if candidate.is_symlink():
            abort(404)
        job_root = candidate.resolve()
    except (OSError, RuntimeError):
        abort(404)
    if (
        not job_root.is_relative_to(jobs_root)
        or job_root == jobs_root
        or not has_event_files(job_root)
    ):
        abort(404, description="No TensorBoard event files were found for this job.")
    try:
        destination = tensorboard_viewers.open(uuid, job_root)
    except Exception:
        app.logger.exception("Could not initialize TensorBoard for job %s", uuid)
        abort(503, description="TensorBoard could not be started. Please try again.")
    return redirect(destination)


if __name__ == "__main__":
    main()
