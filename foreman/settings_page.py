"""Settings blueprint.

Switches integrations that cost time on every page on or off, and shows the
state of the tracking-link index. Changes are stored in foreman.conf.
"""
from flask import Blueprint, abort, current_app, render_template, request

from foreman import settings
from foreman.slurm import check_host
from foreman.tracking_index import INDEX as tracking_index

settings_page = Blueprint("settings_page", __name__, url_prefix="/settings")


@settings_page.before_app_request
def reload_settings():
    """Applies hand edits of foreman.conf without restarting Foreman."""
    if settings.reload_if_changed(current_app.config):
        tracking_index.request_sweep()


def require_foreman_request(name: str) -> None:
    # A custom header prevents cross-origin HTML forms from submitting writes.
    # Foreman's same-origin HTMX requests supply it; this is not authentication.
    if request.headers.get("X-Foreman-Request") != name:
        abort(403)


def _render(status=200, **context):
    config = current_app.config
    context.setdefault("values", {s.key: config.get(s.key) for s in settings.SETTINGS})
    sections = {section: title for section, title in settings.SECTIONS.items()
                if any(s.section == section and s.page == "settings"
                       for s in settings.SETTINGS)}
    return render_template(
        "settings/view.html.jinja",
        settings=[s for s in settings.SETTINGS if s.page == "settings"],
        sections=sections,
        sources=config.get("SETTINGS_SOURCES", {}),
        path=config.get("SETTINGS_PATH") or settings.settings_path(),
        index=tracking_index,
        **context,
    ), status


@settings_page.route("/")
def view():
    return _render()


@settings_page.route("/", methods=["POST"])
def save():
    require_foreman_request("settings")
    values, errors, status = update_settings("settings", request.form)
    if errors:
        return _render(status, values=values, errors=errors)
    return _render(saved=True)


def update_settings(page: str, form) -> tuple[dict, list[str], int]:
    """Validates and stores the settings a page edits.

    Returns the submitted values, errors, and the HTTP status for the response.
    Nothing is stored if any value is invalid. Other pages' settings are kept.
    """
    values, errors = {}, []
    for setting in settings.SETTINGS:
        if setting.page != page:
            continue
        if setting.is_switch:
            values[setting.key] = form.get(setting.key) == "on"
            continue
        # Lists with choices are checkboxes; other lists are typed comma-separated.
        if setting.type is list and setting.choices:
            raw = ",".join(form.getlist(setting.key))
        else:
            raw = form.get(setting.key, "")
        try:
            values[setting.key] = setting.parse(raw)
        except ValueError as error:
            errors.append(str(error))
            values[setting.key] = raw
    if errors:
        return values, errors, 400

    # Store only what differs from config.py/config_local.py, so that the file
    # records actual choices and the Python configuration still applies otherwise.
    path = current_app.config.get("SETTINGS_PATH") or settings.settings_path()
    stored = settings.read(path)
    python = settings.python_values()
    for key, value in values.items():
        if python.get(key) == value:
            stored.pop(key, None)
        else:
            stored[key] = value
    try:
        settings.write(path, stored)
    except OSError:
        current_app.logger.exception("Could not write settings to %s", path)
        return values, [f"Could not write {path}."], 500
    current_app.config["SETTINGS_PATH"] = path
    settings.reload(current_app.config)
    tracking_index.request_sweep()
    return values, [], 200


@settings_page.route("/slurm/check", methods=["POST"])
def check_slurm_hosts():
    """Checks that Slurm commands work on each configured host."""
    require_foreman_request("settings")
    results = [(host, *check_host(host)) for host in current_app.config["SLURM_HOSTS"]]
    return render_template("settings/_host_check.html.jinja", results=results)


@settings_page.route("/status")
def status():
    return render_template("settings/_status.html.jinja", index=tracking_index)


@settings_page.route("/index/refresh", methods=["POST"])
def refresh_index():
    require_foreman_request("settings")
    tracking_index.request_sweep()
    return render_template("settings/_status.html.jinja", index=tracking_index)


@settings_page.route("/cache/clear", methods=["POST"])
def clear_cache():
    require_foreman_request("settings")
    tracking_index.clear()
    return render_template(
        "settings/_status.html.jinja", index=tracking_index, cleared=True,
    )
