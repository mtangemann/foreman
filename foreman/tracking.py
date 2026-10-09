"""Bounded, local discovery of experiment-tracking links."""

import configparser
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import MutableMapping
from urllib.parse import quote, urlsplit, urlunsplit

import yaml

# Bound both directory scanning and the individual text files we inspect.
MAX_DEPTH = 8
MAX_ENTRIES = 10_000
MAX_FILES = 256
MAX_FILE_BYTES = 256 * 1024
SKIP_DIRECTORIES = {
    ".git", "__pycache__", "node_modules", "artifact", "artifacts",
    "checkpoint", "checkpoints",
}

# Text logs that may print a run's URL, e.g. MLflow's "View run NAME at: URL".
LOG_SUFFIXES = {".log", ".out", ".err", ".txt"}
MAX_LOG_FILES = 64
# Only the end of larger logs is read; trackers print run URLs near the end.
MAX_LOG_BYTES = 256 * 1024 * 1024
_LOG_CHUNK_BYTES = 8 * 1024 * 1024
# Chunks overlap so that a URL split across a chunk boundary is still found.
_LOG_OVERLAP_BYTES = 4096
_URL_CHARS = rb"[^\s\"'<>]"
_LOG_LINK = re.compile(
    rb"(?:View run (?P<name>\S+) at: )?(?:"
    rb"(?P<mlflow>https?://" + _URL_CHARS
    + rb"+/#/experiments/[0-9]+/runs/[0-9a-f]{32})"
    rb"|(?P<wandb>https?://[^\s\"'<>#/]*wandb[^\s\"'<>#/]*"
    rb"(?:/[^\s\"'<>#/]+){2}/runs/[A-Za-z0-9]+)"
    rb")"
)


def _safe_url(value):
    if not isinstance(value, str) or not value:
        return None
    if "\\" in value or any(c.isspace() or ord(c) < 32 or ord(c) == 127
                            for c in value):
        return None
    try:
        parts = urlsplit(value)
        if (parts.scheme not in ("http", "https") or not parts.hostname
                or parts.username is not None or parts.password is not None):
            return None
        # Accessing port checks malformed/out-of-range ports.
        _ = parts.port
    except ValueError:
        return None
    return value


def _base_url(value):
    value = _safe_url(value)
    if value is None:
        return None
    parts = urlsplit(value)
    if parts.query:
        return None
    return urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip("/"), "", ""))


def _identity(value):
    if isinstance(value, (str, int)) and not isinstance(value, bool):
        value = str(value)
        if value and value not in (".", ".."):
            return value
    return None


def _link(provider, fields, mlflow_ui_url, wandb_ui_url):
    if not isinstance(fields, dict):
        return None
    if "url" in fields:
        url = _safe_url(fields["url"])
        if url is None:
            return None
        label = (
            _identity(fields.get("label"))
            or _identity(fields.get("run_id"))
            or url
        )
    elif provider == "mlflow":
        experiment = _identity(fields.get("experiment_id"))
        run = _identity(fields.get("run_id", fields.get("run_uuid")))
        base = _base_url(
            fields.get("ui_url") or mlflow_ui_url or fields.get("tracking_uri")
        )
        if not experiment or not base:
            return None
        url = f"{base}/#/experiments/{quote(experiment, safe='')}"
        if run:
            url += f"/runs/{quote(run, safe='')}"
        label = f"MLflow: {run}" if run else f"MLflow experiment: {experiment}"
    else:
        entity = _identity(fields.get("entity"))
        project = _identity(fields.get("project"))
        run = _identity(fields.get("run_id", fields.get("id")))
        base = _base_url(
            fields.get("ui_url") or fields.get("base_url")
            or wandb_ui_url or "https://wandb.ai"
        )
        if not entity or not project or not run or not base:
            return None
        url = (f"{base}/{quote(entity, safe='')}/{quote(project, safe='')}"
               f"/runs/{quote(run, safe='')}")
        label = f"W&B: {project}/{run}"
    if url is None:
        return None
    return {"provider": provider, "label": label, "url": url}


class _Reader:
    """Share a file budget across all candidates, including inherited settings."""

    def __init__(self, root):
        self.root = root
        self.cache = {}

    def read(self, path):
        if path in self.cache:
            return self.cache[path]
        if len(self.cache) >= MAX_FILES:
            return {}
        result = {}
        self.cache[path] = result
        try:
            if (path.is_symlink() or not path.resolve().is_relative_to(self.root)
                    or not path.is_file() or path.stat().st_size > MAX_FILE_BYTES):
                return result
            with path.open("rb") as stream:
                raw = stream.read(MAX_FILE_BYTES + 1)
            if len(raw) > MAX_FILE_BYTES:
                return result
            text = raw.decode("utf-8")
            if path.name == "settings":
                parser = configparser.ConfigParser(interpolation=None)
                parser.read_string(text)
                if parser.has_section("default"):
                    result = dict(parser["default"])
            elif path.suffix == ".json":
                result = json.loads(text)
            else:
                result = yaml.safe_load(text)
            if not isinstance(result, dict):
                result = {}
        except (OSError, RuntimeError, ValueError, UnicodeError, yaml.YAMLError,
                configparser.Error, RecursionError):
            result = {}
        self.cache[path] = result
        return result


def _directories(root):
    """Yield a sorted, bounded tree, never following directory symlinks."""
    pending = [(root, 0)]
    remaining = MAX_ENTRIES
    while pending and remaining:
        directory, depth = pending.pop()
        yield directory
        if depth >= MAX_DEPTH:
            continue
        children = []
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    remaining -= 1
                    if (entry.name.lower() not in SKIP_DIRECTORIES
                            and entry.is_dir(follow_symlinks=False)):
                        children.append(Path(entry.path))
                    if not remaining:
                        break
        except OSError:
            continue
        pending.extend((child, depth + 1) for child in sorted(children, reverse=True))


def _wandb_fields(directory, reader):
    fields = {}
    # Shared wandb/settings supplies project/entity in many local layouts.
    paths = [
        directory.parent / "settings",
        directory / "settings",
        directory / "files" / "settings",
        directory / "files" / "wandb-settings.json",
        directory / "files" / "wandb-metadata.json",
        directory / "wandb-metadata.json",
    ]
    for path in paths:
        data = reader.read(path)
        fields.update(data)
        if isinstance(data.get("wandb"), dict):
            fields.update(data["wandb"])
    if "run_id" not in fields and "id" not in fields:
        fields["run_id"] = directory.name.rsplit("-", 1)[-1]
    return fields


@dataclass(frozen=True)
class _ScannedLog:
    identity: tuple[int, int]
    size: int
    mtime_ns: int
    # (provider, url, run name or None), in order of appearance.
    links: tuple[tuple[str, str, str | None], ...]


# Logs are scanned once and then only when they change; a log that grew (e.g.
# of a running job) is scanned from where the last scan ended. The cache is in
# memory unless `set_log_cache` provides a persistent mapping.
_log_cache: MutableMapping = {}
_log_cache_lock = Lock()


def set_log_cache(cache: MutableMapping) -> None:
    """Stores log scans in `cache` (e.g. a diskcache.Cache), keyed by path."""
    global _log_cache
    _log_cache = cache


def _log_files(root):
    """Yield text logs below root, bounded like other discovery."""
    count = 0
    for directory in _directories(root):
        try:
            with os.scandir(directory) as entries:
                names = sorted(
                    entry.name for entry in entries
                    if entry.is_file(follow_symlinks=False)
                    and os.path.splitext(entry.name)[1].lower() in LOG_SUFFIXES
                )
        except OSError:
            continue
        for name in names:
            yield directory / name
            count += 1
            if count >= MAX_LOG_FILES:
                return


def _links_in(data):
    """Yield (provider, url, name) for run URLs in a chunk of log text."""
    # Every run URL contains "/runs/". Searching for it is much faster than
    # running the full pattern over every byte, so only matching lines are parsed.
    end = 0
    while (hit := data.find(b"/runs/", end)) != -1:
        start = data.rfind(b"\n", 0, hit) + 1
        end = data.find(b"\n", hit)
        end = len(data) if end == -1 else end
        for match in _LOG_LINK.finditer(data, start, end):
            provider = "mlflow" if match["mlflow"] else "wandb"
            name = match["name"]
            yield (
                provider,
                match[provider].decode("utf-8", "replace"),
                name.decode("utf-8", "replace") if name else None,
            )


def _scan_log(path):
    """Return run links printed in a log, reusing earlier scans of the file."""
    try:
        stat = path.stat()
    except OSError:
        return ()
    identity = (stat.st_dev, stat.st_ino)
    key = f"log:{path}"
    with _log_cache_lock:
        cached = _log_cache.get(key)
    if (cached and cached.identity == identity and cached.size == stat.st_size
            and cached.mtime_ns == stat.st_mtime_ns):
        return cached.links

    start, links = 0, []
    if cached and cached.identity == identity and stat.st_size > cached.size:
        start, links = cached.size - _LOG_OVERLAP_BYTES, list(cached.links)
    start = max(start, stat.st_size - MAX_LOG_BYTES, 0)
    try:
        with path.open("rb") as stream:
            stream.seek(start)
            previous = b""
            while chunk := stream.read(_LOG_CHUNK_BYTES):
                data = previous + chunk
                for link in _links_in(data):
                    if link not in links:
                        links.append(link)
                previous = data[-_LOG_OVERLAP_BYTES:]
    except OSError:
        return cached.links if cached else ()
    scanned = _ScannedLog(identity, stat.st_size, stat.st_mtime_ns, tuple(links))
    with _log_cache_lock:
        _log_cache[key] = scanned
    return scanned.links


def _log_links(root, providers):
    """Turn URLs printed in the job's logs into links for the given providers."""
    links = []
    for path in _log_files(root):
        for provider, url, name in _scan_log(path):
            if provider not in providers or _safe_url(url) is None:
                continue
            run = url.rstrip("/").rsplit("/", 1)[-1]
            label = f"{'MLflow' if provider == 'mlflow' else 'W&B'}: {name or run}"
            links.append({"provider": provider, "label": label, "url": url})
    return links


def discover_tracking_links(
    root: Path, *, mlflow_ui_url=None, wandb_ui_url=None
) -> list[dict[str, str]]:
    """Discover safe links without network access or parsing artifacts.

    Run URLs that trackers print into the job's text logs are found as well.

    A provider key in root metadata.yaml's tracking mapping replaces that
    provider's automatic discoveries, even when its value is empty or invalid.
    Host configuration alone never supplies experiment or run identity.
    """
    try:
        root = Path(root).resolve()
        if not root.is_dir():
            return []
    except (OSError, RuntimeError):
        return []
    reader = _Reader(root)
    metadata = reader.read(root / "metadata.yaml")
    overrides = metadata.get("tracking", {})
    if not isinstance(overrides, dict):
        overrides = {}
    links = []

    def add(provider, fields):
        link = _link(provider, fields, mlflow_ui_url, wandb_ui_url)
        if link:
            links.append(link)

    for provider in ("mlflow", "wandb"):
        if provider in overrides:
            records = overrides[provider]
            for record in records if isinstance(records, list) else [records]:
                add(provider, record)

    for directory in _directories(root):
        if ("mlflow" not in overrides and directory.parent.parent.name == "mlruns"):
            fields = dict(reader.read(directory / "meta.yaml"))
            experiment = reader.read(directory.parent / "meta.yaml")
            fields.setdefault("experiment_id", experiment.get("experiment_id"))
            if _identity(fields.get("run_id", fields.get("run_uuid"))):
                add("mlflow", fields)
        if ("wandb" not in overrides
                and directory.name.startswith(("run-", "offline-run-"))):
            fields = _wandb_fields(directory, reader)
            # Offline identity is not evidence that this run was synced.
            if directory.name.startswith("offline-run-") and "url" not in fields:
                continue
            # Saved W&B settings often name the API host, not the browser UI.
            if wandb_ui_url:
                fields["ui_url"] = wandb_ui_url
            elif fields.get("base_url") in (
                "https://api.wandb.ai", "https://api.wandb.ai/"
            ):
                fields["base_url"] = "https://wandb.ai"
            add("wandb", fields)

    links.extend(_log_links(root, {"mlflow", "wandb"} - set(overrides)))

    # Stable ordering and deduplication also make repeated discovery cacheable.
    unique: dict[tuple[str, str], dict[str, str]] = {}
    for link in sorted(links, key=lambda item: (
        item["provider"], item["url"], item["label"]
    )):
        unique.setdefault((link["provider"], link["url"]), link)
    return list(unique.values())
