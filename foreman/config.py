import os

# Default configuration. For machine-specific overrides, create `config_local.py`
# next to this file (gitignored); its UPPERCASE names take precedence.
# See config_local.example.py. Settings changed on the settings page are stored
# in foreman.conf (see foreman/settings.py) and take precedence over both.

# SSH host names of the SLURM head nodes. Use "local" to run `squeue` and `sacct`
# on the machine running Foreman instead, e.g. when it runs on a cluster node.
SLURM_HOSTS = ["galvani"]

# Cache expiration time in seconds.
SLURM_CACHE_EXPIRATION = 5

# How many days of Slurm history (`sacct`) to use for the status of finished jobs
# that did not write `output/done`, e.g. because they ran out of memory.
SLURM_HISTORY_DAYS = 30

# Browser-accessible tracking UI hosts; these do not identify a job's run.
MLFLOW_UI_URL = os.environ.get("FOREMAN_MLFLOW_UI_URL")
WANDB_UI_URL = os.environ.get("FOREMAN_WANDB_UI_URL")

# Where to keep discovered tracking links and log scans across restarts
# (None: in memory only). Defaults to $XDG_CACHE_HOME/foreman/tracking.
TRACKING_CACHE_DIR = os.environ.get("FOREMAN_TRACKING_CACHE_DIR", "default")

# Seconds between background refreshes of all jobs' tracking links. Job tables
# show indexed links; 0 disables the indexer and discovers links per request.
TRACKING_REFRESH_INTERVAL = 60

# Switches for integrations that cost time on every page (settings page).
MLFLOW_ENABLED = True
WANDB_ENABLED = True
TRACKING_LOG_SCAN = True
TENSORBOARD_ENABLED = True
SLURM_STATUS_ENABLED = True
SLURM_HISTORY_ENABLED = True
REPOSITORY_BRANCHES = True

# Job queue page (editable on the page): shown columns (see QUEUE_COLUMNS in
# foreman/slurm.py), whose jobs to show ("me" or "all"), and how often the page
# updates itself in seconds (0: only on reload).
QUEUE_COLUMNS = [
    "state", "partition", "submitted", "start", "elapsed", "resources", "nodelist",
]
QUEUE_SCOPE = "me"
QUEUE_REFRESH = 30
