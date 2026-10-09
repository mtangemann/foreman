"""Cached tracking links per job, refreshed in the background.

Discovering a job's links walks its directory and scans its logs, which is too
slow to do for every row of a table with thousands of jobs. While the
background indexer runs, tables only read stored links; the job page refreshes
its own job. Without the indexer, links are discovered on every request.
"""

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Callable, Optional

from diskcache import Cache

from foreman import tracking

LOGGER = logging.getLogger(__name__)


def default_cache_dir() -> Path:
    """Returns `$XDG_CACHE_HOME/foreman/tracking` (default `~/.cache/...`)."""
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "foreman" / "tracking"


# Settings that change the links themselves; other settings (which providers to
# look for, whether to search logs) only change what a refresh discovers.
KEY_SETTINGS = ("mlflow_ui_url", "wandb_ui_url")


@dataclass(frozen=True)
class Sweep:
    """The last completed refresh of all jobs."""

    finished: datetime
    seconds: float
    jobs: int


class TrackingIndex:
    """Stores discovered links per job, in memory or in an on-disk cache."""

    def __init__(self) -> None:
        self._store: dict | Cache = {}
        self._lock = Lock()
        self._thread: Optional[Thread] = None
        self._wake = Event()
        self.cache_dir: Optional[Path] = None
        self.sweeping = False
        self.last_sweep: Optional[Sweep] = None

    @property
    def background(self) -> bool:
        """Returns whether the background indexer is running."""
        return self._thread is not None and self._thread.is_alive()

    def use_disk_cache(self, directory: Path) -> None:
        """Keeps links and log scans in `directory`, across Foreman restarts."""
        cache = Cache(str(directory))
        self._store = cache
        self.cache_dir = Path(directory)
        tracking.set_log_cache(cache)

    def cache_size(self) -> Optional[int]:
        """Returns the on-disk cache size in bytes, if a disk cache is used."""
        return self._store.volume() if isinstance(self._store, Cache) else None

    def clear(self) -> None:
        """Forgets all links and log scans, and refreshes all jobs."""
        self._store.clear()
        if not isinstance(self._store, Cache):
            tracking.set_log_cache({})
        self.request_sweep()

    def request_sweep(self) -> None:
        """Starts the next refresh of all jobs now instead of after the pause."""
        self._wake.set()

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

    def start(
        self, jobs_root: Path, configuration: Callable[[], tuple[float, dict]],
    ) -> None:
        """Refreshes all jobs in a background thread.

        `configuration` returns the pause between sweeps (seconds) and the
        discovery settings; it is called before every sweep, so that changed
        settings apply from the next sweep on.
        """
        with self._lock:
            if self.background:
                return
            self._thread = Thread(
                target=self._run, args=(Path(jobs_root), configuration),
                name="tracking-index", daemon=True,
            )
            self._thread.start()

    def _run(self, jobs_root: Path, configuration) -> None:
        while True:
            interval, settings = configuration()
            self._wake.clear()
            self.sweeping = True
            started = time.monotonic()
            roots = self._job_roots(jobs_root)
            for root in roots:
                try:
                    self.refresh(root, **settings)
                except Exception:
                    LOGGER.exception("Could not index tracking links of %s", root)
            seconds = time.monotonic() - started
            self.last_sweep = Sweep(datetime.now(), seconds, len(roots))
            self.sweeping = False
            LOGGER.info(
                "Indexed tracking links of %d jobs in %.1fs.", len(roots), seconds
            )
            self._wake.wait(interval)

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
        key_settings = sorted((name, settings.get(name)) for name in KEY_SETTINGS)
        return "job:" + repr((str(Path(root).resolve()), key_settings))


INDEX = TrackingIndex()
