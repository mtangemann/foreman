"""Settings that can be changed from the settings page.

They are stored in `foreman.conf` (INI, gitignored) at the repository root, or
at `$FOREMAN_CONFIG`, and take precedence over `config.py` and
`config_local.py`. Only switches and bounded numbers are editable here: host
names and URLs stay in the Python configuration, as Foreman has no login.
"""

import configparser
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

from flask import Config

LOGGER = logging.getLogger(__name__)

PACKAGE = Path(__file__).parent


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

    @property
    def is_switch(self) -> bool:
        return self.type is bool

    def parse(self, value: str) -> Union[bool, int]:
        """Parses a value from foreman.conf or the settings form; raises ValueError."""
        if self.type is bool:
            states = configparser.ConfigParser.BOOLEAN_STATES
            if value.strip().lower() not in states:
                raise ValueError(f"{self.label}: expected on or off.")
            return states[value.strip().lower()]
        try:
            number = int(value)
        except ValueError:
            raise ValueError(f"{self.label}: expected a whole number.") from None
        if self.minimum is not None and number < self.minimum:
            raise ValueError(f"{self.label}: must be at least {self.minimum}.")
        if self.maximum is not None and number > self.maximum:
            raise ValueError(f"{self.label}: must be at most {self.maximum}.")
        return number

    @staticmethod
    def format(value: Any) -> str:
        if isinstance(value, bool):
            return "on" if value else "off"
        return str(value)


SECTIONS = {
    "tracking": "Tracking & TensorBoard",
    "slurm": "Slurm",
    "indexing": "Indexing",
}

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
    local.from_pyfile(PACKAGE / "config_local.py", silent=True)
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
    python.from_pyfile(PACKAGE / "config_local.py", silent=True)
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
