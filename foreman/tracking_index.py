"""Cached tracking links per job, refreshed in the background.

Discovering a job's links walks its directory and scans its logs, which is too
slow to do for every row of a table with thousands of jobs. While the
background indexer runs, tables only read stored links; the job page refreshes
its own job. Without the indexer, links are discovered on every request.
"""

import logging
import os
import time
from pathlib import Path
from threading import Lock, Thread
from typing import Optional

from diskcache import Cache

from foreman import tracking

LOGGER = logging.getLogger(__name__)


def default_cache_dir() -> Path:
    """Returns `$XDG_CACHE_HOME/foreman/tracking` (default `~/.cache/...`)."""
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "foreman" / "tracking"


class TrackingIndex:
    """Stores discovered links per job, in memory or in an on-disk cache."""

    def __init__(self) -> None:
        self._store: dict | Cache = {}
        self._lock = Lock()
        self._thread: Optional[Thread] = None

    @property
    def background(self) -> bool:
        """Returns whether the background indexer is running."""
        return self._thread is not None and self._thread.is_alive()

    def use_disk_cache(self, directory: Path) -> None:
        """Keeps links and log scans in `directory`, across Foreman restarts."""
        cache = Cache(str(directory))
        self._store = cache
        tracking.set_log_cache(cache)

    def links(self, root: Path, fresh: bool = False, **settings) -> list[dict]:
        """Returns a job's links: stored ones while indexing in the background.

        `settings` are the keyword arguments of `discover_tracking_links`.
        """
        if fresh or not self.background:
            return self.refresh(root, **settings)
        return self._store.get(self._key(root, settings)) or []

    def refresh(self, root: Path, **settings) -> list[dict]:
        """Discovers and stores a job's links."""
        links = tracking.discover_tracking_links(root, **settings)
        self._store[self._key(root, settings)] = links
        return links

    def start(self, jobs_root: Path, interval: float, **settings) -> None:
        """Refreshes all jobs in a background thread, sweeping every `interval`s."""
        with self._lock:
            if self.background:
                return
            self._thread = Thread(
                target=self._run, args=(Path(jobs_root), interval, settings),
                name="tracking-index", daemon=True,
            )
            self._thread.start()

    def _run(self, jobs_root: Path, interval: float, settings: dict) -> None:
        while True:
            started = time.monotonic()
            roots = self._job_roots(jobs_root)
            for root in roots:
                try:
                    self.refresh(root, **settings)
                except Exception:
                    LOGGER.exception("Could not index tracking links of %s", root)
            LOGGER.info(
                "Indexed tracking links of %d jobs in %.1fs.",
                len(roots), time.monotonic() - started,
            )
            time.sleep(interval)

    @staticmethod
    def _job_roots(jobs_root: Path) -> list[Path]:
        """Returns job directories, most recently changed (likely viewed) first."""
        roots = []
        try:
            with os.scandir(jobs_root) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            roots.append((entry.stat().st_mtime, Path(entry.path)))
                    except OSError:
                        continue
        except OSError:
            LOGGER.warning("Could not list jobs in %s.", jobs_root)
        return [root for _, root in sorted(roots, reverse=True)]

    @staticmethod
    def _key(root: Path, settings: dict) -> str:
        # Configured UI hosts change the links, so they are part of the key.
        return "job:" + repr((str(Path(root).resolve()), sorted(settings.items())))


INDEX = TrackingIndex()
