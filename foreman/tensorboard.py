"""Local TensorBoard event detection and reusable embedded viewers."""

import os
from pathlib import Path
from threading import Lock

# `from tensorboard import program` can bind TensorBoard's lazy proxy instead of
# the module, depending on import order; import the submodule itself.
import tensorboard.program as program
from tensorboard.backend import application

MAX_DEPTH = 8
MAX_ENTRIES = 10_000
EVENT_PREFIX = "events.out.tfevents."


def has_event_files(root: Path) -> bool:
    """Find a non-empty standard event file without parsing data or symlinks.

    Discovery is best-effort and bounded like other tracking discovery. A match
    means data is present, not that TensorBoard can parse every event in it.
    """
    root = Path(root)
    try:
        if root.is_symlink() or not root.is_dir():
            return False
    except OSError:
        return False
    pending = [(root, 0)]
    remaining = MAX_ENTRIES
    while pending and remaining > 0:
        directory, depth = pending.pop()
        children = []
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    remaining -= 1
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            if depth < MAX_DEPTH:
                                children.append(Path(entry.path))
                        elif (
                            entry.name.startswith(EVENT_PREFIX)
                            and entry.name != EVENT_PREFIX
                            and entry.is_file(follow_symlinks=False)
                            and entry.stat(follow_symlinks=False).st_size > 0
                        ):
                            return True
                    except OSError:
                        # A running job may move or remove files during a scan.
                        pass
                    if remaining <= 0:
                        break
        except OSError:
            continue
        pending.extend((child, depth + 1) for child in sorted(children, reverse=True))
    return False


class TensorBoardViewers:
    """Own one TensorBoard loader and WSGI mount per opened job per process.

    Intended for trusted repositories: TensorBoard's ongoing loader follows its
    own filesystem rules, including nested symlinks. Discovery/path validation
    is not a sandbox. Programs are retained until the Foreman process exits.
    """

    def __init__(self, mounts) -> None:
        self._mounts = mounts
        self._programs: dict[str, program.TensorBoard] = {}
        self._lock = Lock()

    def open(self, job_id: str, logdir: Path) -> str:
        mount = f"/tensorboard/{job_id}"
        with self._lock:
            if mount not in self._programs:
                tensorboard = program.TensorBoard()
                tensorboard.configure(logdir=str(logdir))
                # TensorBoard has no public embedded-WSGI factory. Keep the
                # version-sensitive provider setup confined to this adapter.
                provider, multiplexer = tensorboard._make_data_provider()
                viewer = application.TensorBoardWSGIApp(
                    tensorboard.flags,
                    tensorboard.plugin_loaders,
                    provider,
                    tensorboard.assets_zip_provider,
                    multiplexer,
                )
                # The program owns its ingester (including Windows file handles).
                # Retain it for as long as the mounted viewer is in use.
                self._programs[mount] = tensorboard
                self._mounts[mount] = viewer
        # Relative asset/API URLs resolve correctly at this canonical URL.
        return mount + "/"
