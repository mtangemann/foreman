"""
Stylesheet freshness check.

Tailwind only emits CSS for the classes it finds in the templates when
`static/output.css` is built. A class added to a template has no effect until
the stylesheet is rebuilt with `make css`, so warn when it looks out of date.
"""
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

PACKAGE = Path(__file__).parent


def stale_stylesheet_reason(package: Path = PACKAGE) -> Optional[str]:
    """Returns why `static/output.css` needs rebuilding, or None if it is current."""
    static = package / "static"
    output = static / "output.css"
    if not output.exists():
        return "static/output.css does not exist"

    sources = [
        static / "input.css",
        static / "tailwind.config.js",
        *(package / "templates").rglob("*.html*"),
    ]
    built = output.stat().st_mtime
    newer = [path for path in sources if path.exists() and path.stat().st_mtime > built]
    if not newer:
        return None
    newest = max(newer, key=lambda path: path.stat().st_mtime)
    return (
        f"static/output.css is older than {len(newer)} source file(s), "
        f"e.g. {newest.relative_to(package)}"
    )


def warn_if_stylesheet_stale(package: Path = PACKAGE) -> None:
    """Logs a warning if Tailwind classes may be missing from the stylesheet."""
    reason = stale_stylesheet_reason(package)
    if reason:
        logger.warning(
            "%s. New or changed styles will not render until you run `make css`.",
            reason,
        )
