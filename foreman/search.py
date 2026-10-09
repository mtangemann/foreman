"""Job search blueprint."""
import json

import yaml
from flask import Blueprint, render_template, request

from foreman.r3_extension import Repository

search = Blueprint("search", __name__, url_prefix="/search")


@search.route("/search", methods=["GET"])
def index():
    return render_template("search/search.html.jinja")


@search.route("/search", methods=["POST"])
def search_post():
    query_str = request.form["query"]
    query_format = request.form['format']

    if query_format == 'json':
        try:
            query = json.loads(query_str)
        except json.JSONDecodeError as error:
            return _error("Invalid JSON query", error)
    elif query_format == 'yaml':
        try:
            query = yaml.safe_load(query_str)
        except yaml.YAMLError as error:
            return _error("Invalid YAML query", error)
    else:
        return _error(f"Invalid format: {query_format}")

    repository = Repository.from_environment()
    jobs = sorted(repository.find(query), key=lambda job: job.timestamp, reverse=True)

    return render_template("search/search_results.html.jinja", jobs=jobs)


def _error(title: str, detail: object = None) -> str:
    return render_template("search/search_error.html.jinja", title=title, detail=detail)
