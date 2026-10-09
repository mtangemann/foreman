"""Settings that can be changed from the settings page.

They are stored in `foreman.conf` (INI, gitignored) at the repository root, or
at `$FOREMAN_CONFIG`, and take precedence over `config.py` and
`config_local.py`. Values are validated strictly before they are stored:
Foreman has no login, and e.g. Slurm host names end up in `ssh` commands.
Tracking server URLs stay in the Python configuration.
"""

import configparser
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

from flask import Config

from foreman.slurm import QUEUE_COLUMNS

LOGGER = logging.getLogger(__name__)

PACKAGE = Path(__file__).parent
# The machine-specific Python configuration (gitignored), loaded by the app.
LOCAL_CONFIG = PACKAGE / "config_local.py"


@dataclass(frozen=True)
class Setting:
    """A setting: an app config key, stored as `option` in a `foreman.conf` section."""

    key: str
    section: str
    option: str
    label: str
    help: str
    type: type = bool
    minimum: Optional[int] = None
    maximum: Optional[int] = None
    # For str and list settings: allowed values, or a pattern each value matches.
    choices: tuple[tuple[str, str], ...] = ()
    pattern: Optional[str] = None
    pattern_help: str = ""
    # The page that edits the setting: "settings", or "queue" for the job queue.
    page: str = "settings"

    @property
    def is_switch(self) -> bool:
        return self.type is bool

    @property
    def kind(self) -> str:
        """Returns how a form edits it: switch, number, choices or text."""
        if self.type is bool:
            return "switch"
        if self.type is int:
            return "number"
        return "choices" if self.choices else "text"

    def parse(self, value: str) -> Union[bool, int, str, list[str]]:
        """Parses a value from foreman.conf or the settings form; raises ValueError."""
        if self.type is bool:
            states = configparser.ConfigParser.BOOLEAN_STATES
            if value.strip().lower() not in states:
                raise ValueError(f"{self.label}: expected on or off.")
            return states[value.strip().lower()]
        if self.type is str:
            return self._check(value.strip())
        if self.type is list:
            items = [item for item in re.split(r"[,\s]+", value) if item]
            if not items:
                raise ValueError(f"{self.label}: choose at least one.")
            # Keep the order of choices, so that e.g. columns appear in a fixed order.
            items = list(dict.fromkeys(self._check(item) for item in items))
            if self.choices:
                order = [name for name, _ in self.choices]
                items.sort(key=order.index)
            return items
        try:
            number = int(value)
        except ValueError:
            raise ValueError(f"{self.label}: expected a whole number.") from None
        if self.minimum is not None and number < self.minimum:
            raise ValueError(f"{self.label}: must be at least {self.minimum}.")
        if self.maximum is not None and number > self.maximum:
            raise ValueError(f"{self.label}: must be at most {self.maximum}.")
        return number

    def _check(self, item: str) -> str:
        if self.choices and item not in dict(self.choices):
            names = ", ".join(name for name, _ in self.choices)
            raise ValueError(f"{self.label}: {item!r} is not one of {names}.")
        if self.pattern and not re.fullmatch(self.pattern, item):
            raise ValueError(
                f"{self.label}: {item!r} is not valid. {self.pattern_help}"
            )
        return item

    @staticmethod
    def format(value: Any) -> str:
        if isinstance(value, bool):
            return "on" if value else "off"
        if isinstance(value, (list, tuple)):
            return ", ".join(value)
        return str(value)


SECTIONS = {
    "tracking": "Tracking & TensorBoard",
    "slurm": "Slurm",
    "indexing": "Indexing",
    "repositories": "Code repositories",
    "queue": "Job queue",
}

# "local", or an SSH destination such as "login" or "me@login.example.org".
# A leading "-" is impossible, so a host can never be read as an ssh option.
HOST_PATTERN = r"local|[A-Za-z0-9_][A-Za-z0-9_.-]*(@[A-Za-z0-9_][A-Za-z0-9_.-]*)?"

SETTINGS = [
    Setting("MLFLOW_ENABLED", "tracking", "mlflow", "MLflow links",
            "Show links to MLflow runs and look for them in job directories."),
    Setting("WANDB_ENABLED", "tracking", "wandb", "W&B links",
            "Show links to Weights & Biases runs and look for them in job "
            "directories."),
    Setting("TRACKING_LOG_SCAN", "tracking", "log_scan", "Search logs for run URLs",
            "Find run URLs printed into job logs, e.g. MLflow's "
            "\"View run … at: URL\". "
            "Reads job logs; the most expensive part of finding runs."),
    Setting("TENSORBOARD_ENABLED", "tracking", "tensorboard", "TensorBoard",
            "Look for TensorBoard event files and show links to open them."),
    Setting("SLURM_HOSTS", "slurm", "hosts", "Slurm hosts",
            "SSH destinations of Slurm login nodes, comma-separated, or \"local\" to "
            "run Slurm commands on this machine (e.g. when Foreman runs on a "
            "cluster node). SSH must work without a password prompt.",
            list, pattern=HOST_PATTERN,
            pattern_help="Use \"local\" or an SSH host such as login or me@login."),
    Setting("SLURM_CACHE_EXPIRATION", "slurm", "cache_expiration",
            "Reuse Slurm answers for (seconds)",
            "How long answers from squeue and sacct are reused between pages.",
            int, 0, 3600),
    Setting("SLURM_STATUS_ENABLED", "slurm", "status", "Live job status",
            "Ask Slurm (squeue) which jobs are queued or running."),
    Setting("SLURM_HISTORY_ENABLED", "slurm", "history", "Job history and run stats",
            "Ask Slurm (sacct) for finished jobs' outcomes, timelines and run "
            "summaries."),
    Setting("SLURM_HISTORY_DAYS", "slurm", "history_days", "History window (days)",
            "How far back job history is used for the status of finished jobs.",
            int, 1, 3650),
    Setting("TRACKING_REFRESH_INTERVAL", "indexing", "refresh_interval",
            "Refresh interval (seconds)",
            "Pause between background refreshes of all jobs' tracking links.",
            int, 10, 86400),
    Setting("REPOSITORY_BRANCHES", "repositories", "branches",
            "Branches containing the commit",
            "On a job's Dependencies tab, look up which branches contain each "
            "repository's commit, using git in R3's local mirror."),
    Setting("QUEUE_COLUMNS", "queue", "columns", "Columns",
            "Columns shown on the job queue page.",
            list, choices=tuple(QUEUE_COLUMNS.items()), page="queue"),
    Setting("QUEUE_SCOPE", "queue", "scope", "Show jobs of",
            "Whose jobs the job queue page shows.",
            str, choices=(("me", "Me"), ("all", "All users")), page="queue"),
    Setting("QUEUE_REFRESH", "queue", "refresh", "Refresh every (seconds)",
            "How often the job queue page updates itself; 0 to update only on reload.",
            int, 0, 3600, page="queue"),
]

SETTINGS_BY_KEY = {setting.key: setting for setting in SETTINGS}


def settings_path() -> Path:
    """Returns `$FOREMAN_CONFIG`, or `foreman.conf` at the repository root."""
    return Path(os.environ.get("FOREMAN_CONFIG") or PACKAGE.parent / "foreman.conf")


def read(path: Path) -> dict[str, Union[bool, int]]:
    """Returns the valid settings in a foreman.conf; invalid ones are logged."""
    parser = configparser.ConfigParser(interpolation=None)
    try:
        if not parser.read(path, encoding="utf-8"):
            return {}
    except configparser.Error as error:
        LOGGER.warning("Ignoring %s: %s", path, error)
        return {}
    values = {}
    for setting in SETTINGS:
        if parser.has_option(setting.section, setting.option):
            try:
                values[setting.key] = setting.parse(
                    parser.get(setting.section, setting.option)
                )
            except ValueError as error:
                LOGGER.warning("Ignoring invalid setting in %s: %s", path, error)
    return values


def write(path: Path, values: dict[str, Any]) -> None:
    """Writes settings to a foreman.conf, replacing it atomically."""
    parser = configparser.ConfigParser(interpolation=None)
    for setting in SETTINGS:
        if setting.key in values:
            if not parser.has_section(setting.section):
                parser.add_section(setting.section)
            value = setting.format(values[setting.key])
            parser.set(setting.section, setting.option, value)
    temporary = path.with_name(f".{path.name}.tmp")
    with open(temporary, "w", encoding="utf-8") as stream:
        stream.write("# Foreman settings, written by the settings page.\n\n")
        parser.write(stream)
    os.replace(temporary, path)


def load(config: Config, path: Optional[Path] = None) -> None:
    """Applies foreman.conf to an app config, remembering where values come from.

    Call after `config.py` and `config_local.py` have been loaded.
    """
    path = path or settings_path()
    local = Config(str(PACKAGE))
    local.from_pyfile(LOCAL_CONFIG, silent=True)
    values = read(path)
    config.update(values)
    config["SETTINGS_PATH"] = path
    config["SETTINGS_SOURCES"] = {
        setting.key: (
            "foreman.conf" if setting.key in values
            else "config_local.py" if setting.key in local
            else "default"
        )
        for setting in SETTINGS
    }
    try:
        config["SETTINGS_MTIME"] = path.stat().st_mtime_ns
    except OSError:
        config["SETTINGS_MTIME"] = None


def python_values() -> dict[str, Any]:
    """Returns the settings as configured without foreman.conf."""
    python = Config(str(PACKAGE))
    python.from_object("foreman.config")
    python.from_pyfile(LOCAL_CONFIG, silent=True)
    return {key: python[key] for key in SETTINGS_BY_KEY if key in python}


def reload(config: Config) -> None:
    """Reloads foreman.conf into an app config loaded with `load`."""
    # Fall back to the Python configuration for settings removed from the file.
    config.update(python_values())
    load(config, config.get("SETTINGS_PATH"))


def reload_if_changed(config: Config) -> bool:
    """Reloads foreman.conf if it was edited since it was loaded."""
    path = config.get("SETTINGS_PATH")
    if path is None:
        return False
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:
        mtime = None
    if mtime == config.get("SETTINGS_MTIME"):
        return False
    reload(config)
    return True


def enabled(config, key: str) -> bool:
    """Returns whether a switch is on; switches are on unless configured off."""
    return bool(config.get(key, True))
