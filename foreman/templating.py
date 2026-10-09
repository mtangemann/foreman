"""Template configuration shared by the app and tests."""

from typing import Optional


def autoescape(template_name: Optional[str]) -> bool:
    """Returns whether to escape HTML in a template.

    Flask only escapes templates ending in `.html` and similar, which leaves
    Foreman's `*.html.jinja` templates unescaped.
    """
    if template_name is None:
        return True  # render_template_string
    return template_name.endswith((".html", ".htm", ".xml", ".html.jinja"))
