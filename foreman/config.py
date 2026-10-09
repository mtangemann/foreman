# Default configuration. For machine-specific overrides, create `config_local.py`
# next to this file (gitignored); its UPPERCASE names take precedence. See config_local.example.py.

# SSH host names of the SLURM head nodes. Use "local" to run `squeue` and `sacct`
# on the machine running Foreman instead, e.g. when it runs on a cluster node.
SLURM_HOSTS = ["galvani"]

# Cache expiration time in seconds.
SLURM_CACHE_EXPIRATION = 5

# How many days of Slurm history (`sacct`) to use for the status of finished jobs
# that did not write `output/done`, e.g. because they ran out of memory.
SLURM_HISTORY_DAYS = 30
