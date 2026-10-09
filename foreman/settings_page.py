"""Settings blueprint.

Switches integrations that cost time on every page on or off, and shows the
state of the tracking-link index. Changes are stored in foreman.conf.
"""
from flask import Blueprint, abort, current_app, render_template, request

from foreman import settings
from foreman.tracking_index import INDEX as tracking_index

settings_page = Blueprint("settings_page", __name__, url_prefix="/settings")


@settings_page.before_app_request
def reload_settings():
    """Applies hand edits of foreman.conf without restarting Foreman."""
    if settings.reload_if_changed(current_app.config):
        tracking_index.request_sweep()


def _require_foreman_request(name: str) -> None:
    # A custom header prevents cross-origin HTML forms from submitting writes.
    # Foreman's same-origin HTMX requests supply it; this is not authentication.
    if request.headers.get("X-Foreman-Request") != name:
        abort(403)


def _render(status=200, **context):
    config = current_app.config
    context.setdefault("values", {s.key: config.get(s.key) for s in settings.SETTINGS})
    return render_template(
        "settings/view.html.jinja",
        settings=settings.SETTINGS,
        sections=settings.SECTIONS,
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
    _require_foreman_request("settings")
    values, errors = {}, []
    for setting in settings.SETTINGS:
        if setting.type is bool:
            values[setting.key] = request.form.get(setting.key) == "on"
            continue
        try:
            values[setting.key] = setting.parse(request.form.get(setting.key, ""))
        except ValueError as error:
            errors.append(str(error))
            values[setting.key] = request.form.get(setting.key, "")
    if errors:
        return _render(400, values=values, errors=errors)

    # Store only what differs from config.py/config_local.py, so that the file
    # records actual choices and the Python configuration still applies otherwise.
    python = settings.python_values()
    changed = {key: value for key, value in values.items() if python.get(key) != value}
    path = current_app.config.get("SETTINGS_PATH") or settings.settings_path()
    try:
        settings.write(path, changed)
    except OSError:
        current_app.logger.exception("Could not write settings to %s", path)
        return _render(500, values=values, errors=[f"Could not write {path}."])
    current_app.config["SETTINGS_PATH"] = path
    settings.reload(current_app.config)
    tracking_index.request_sweep()
    return _render(saved=True)


@settings_page.route("/status")
def status():
    return render_template("settings/_status.html.jinja", index=tracking_index)


@settings_page.route("/index/refresh", methods=["POST"])
def refresh_index():
    _require_foreman_request("settings")
    tracking_index.request_sweep()
    return render_template("settings/_status.html.jinja", index=tracking_index)


@settings_page.route("/cache/clear", methods=["POST"])
def clear_cache():
    _require_foreman_request("settings")
    tracking_index.clear()
    return render_template(
        "settings/_status.html.jinja", index=tracking_index, cleared=True,
    )
