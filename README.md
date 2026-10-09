# Foreman

Foreman is a Web UI for browsing and managing the jobs in an R3 repository.


## Installation
```bash
make install
```

Usually, foreman will run on an external server. I found the following alias useful to
start foreman from my local machine:

```bash
# Replace `server` with the hostname of the server running foreman
alias foreman="ssh -t -L ${PORT:-5000}:localhost:${PORT:-5000} ${HOST:-server} python -m foreman.app --port ${PORT:-5000}"
```


## Development
Foreman is styled with [Tailwind CSS](https://tailwindcss.com/). The stylesheet
`foreman/static/output.css` is generated and not committed, and it only contains the
classes Tailwind found in the templates when it was built. **Rebuild it whenever you
add or change Tailwind classes in a template, or after pulling changes that do:**

```bash
make css        # rebuild once
make css-watch  # rebuild automatically while editing templates
```

Otherwise new classes silently have no effect and pages look unstyled. Foreman logs a
warning at startup if `output.css` is missing or older than any template.


## Settings
The **Settings** page (top right) switches off integrations you don't use, to
make pages faster:

- **Tracking & TensorBoard:** MLflow links, W&B links, searching job logs for
  run URLs (the most expensive part of finding runs), and TensorBoard detection.
- **Slurm:** live job status (`squeue`), job history and run stats (`sacct`), and
  how many days of history to use.
- **Indexing:** the pause between background refreshes of tracking links.
- **Code repositories:** whether to look up which branches contain a job's
  commits.

The page also shows the state of the tracking-link index and its cache, which
can be refreshed or cleared there.

Changes apply immediately and are stored in `foreman.conf` at the repository
root (gitignored), or in the file named by `FOREMAN_CONFIG`. It is an INI file
that can also be edited by hand; edits apply on the next request. Its values
take precedence over `config_local.py`, which takes precedence over the
defaults in `foreman/config.py`. The page only stores values that differ from
the Python configuration, and shows where each value comes from.

```ini
[tracking]
mlflow = on
wandb = off
log_scan = on
tensorboard = off

[slurm]
status = on
history = on
history_days = 30

[indexing]
refresh_interval = 60
```

Host names and tracking server URLs are only configured in the Python
configuration, since Foreman has no login and host names are used in `ssh`
commands. Setting `TRACKING_REFRESH_INTERVAL = 0` there disables the background
indexer until Foreman restarts.


## Code repositories
A job's sidebar lists the git repositories it depends on under **Repositories**,
each linking to the code on GitHub at exactly the commit the job used (and at
the checked-out path, if only part of the repository was used). The job's
**Dependencies** tab lists them below the jobs it depends on, with links to the
repository, the commit, and the changes since then on the default branch.

R3 records each repository's exact commit, not the branch it came from, so the
Dependencies tab shows the branches that *contain* the commit, the default
branch first. They are looked up with git in R3's local mirror
(`$R3_REPOSITORY/git/github.com/<owner>/<name>`), cached for ten minutes, and
can be switched off on the settings page.


## Editing job tags
On a job's detail page, choose **Edit tags**, enter one tag per line, and select
**Save tags**. Add, remove, or rename tags by changing the lines; **Cancel**
discards unsaved edits. Whitespace, blank lines, and duplicate tags are removed,
while order and case are preserved. Saving an empty list clears all tags.

Changes apply only to that job's tags and immediately refresh R3's search index.
The Foreman process needs write access to both the job's `metadata.yaml` and the
repository index. This uses R3's existing metadata and private index-update APIs;
file and index writes are not atomic, and concurrent edits can overwrite changes.


## File previews

The file browser previews UTF-8 `.json`, `.txt`, `.toml`, `.ini`, `.cfg`, `.tsv`,
and `.xml` files in addition to Python and YAML files. TOML and TSV are displayed
as plain text rather than parsed or rendered as tables. Image previews support
PNG, JPEG (`.jpg` and `.jpeg`), GIF, and WebP. Text and image extension matching
is case-insensitive; existing lowercase `.html` handling is unchanged.
Invalid UTF-8 or binary text content falls back to the
unsupported-file view instead of raising a decoding error.

## Experiment tracking links

Foreman discovers MLflow and Weights & Biases runs within each job directory,
including nested output folders. Discovery does not contact tracking servers,
parse binary files, or establish whether a run still exists.

- **Run URLs in logs:** text logs (`*.log`, `*.out`, `*.err`, `*.txt`, e.g. the
  Slurm log) are searched for MLflow run URLs (`…/#/experiments/<id>/runs/<id>`)
  and W&B run URLs (on a host containing `wandb`). Trackers print these, e.g.
  MLflow's `View run NAME at: URL` when a run ends, which also names the link.
  This needs no configuration, so it works for runs logged to a tracking server.
  Up to 64 logs per job are searched, and only the last 256 MiB of each.
- **MLflow:** `mlruns/<experiment>/<run>/meta.yaml` supplies run identity;
  experiment `meta.yaml` can supply the experiment ID. Configure
  `FOREMAN_MLFLOW_UI_URL` with the HTTP(S) browser-facing MLflow base URL
  (including a deployment path prefix if needed). A local `file:` tracking URI
  is not a browser link.
- **W&B:** `run-*` directories and saved `wandb/settings`, run settings, and
  `wandb-metadata.json` supply entity, project, and run identity. Online runs
  default to `https://wandb.ai`; configure `FOREMAN_WANDB_UI_URL` for a private
  browser-facing installation. The configured UI host takes precedence over
  inferred saved hosts; the standard saved `https://api.wandb.ai` host maps to
  `https://wandb.ai`. Host configuration alone never supplies missing identity.
  `offline-run-*` directories are not assumed to have been synced:
  use explicit job metadata for a known remote run, or a saved explicit run URL.

Set environment variables before starting Foreman. The discovery backend itself
accepts these host values as keyword arguments rather than reading environment
variables.

### Caching and background indexing

Scanning every job's directory and logs is too slow to do for each row of a job
table, so `python -m foreman.app` indexes all jobs' links in a background thread,
most recently changed jobs first, and refreshes them every
`TRACKING_REFRESH_INTERVAL` seconds (default 60; `0` disables the indexer and
discovers links on every request). Tables show indexed links, so right after the
first start, links appear as the indexer reaches each job. A job's own page
always discovers its links when opened.

Log scans and discovered links are kept in `TRACKING_CACHE_DIR` (default
`$XDG_CACHE_HOME/foreman/tracking`, i.e. `~/.cache/foreman/tracking`; override
with `FOREMAN_TRACKING_CACHE_DIR`, or set it to `None` in `config_local.py` to
keep them in memory only). Unchanged logs are not read again, also after a
restart, and a growing log is only read from where the last scan ended.

### Explicit links

In the job root's `metadata.yaml`, a provider entry under `tracking` overrides
all automatic discoveries for that provider. Each provider accepts one mapping
or a list of mappings. An empty or invalid provider entry suppresses automatic
links rather than falling back to filesystem inference.

```yaml
tracking:
  mlflow:
    - experiment_id: "12"
      run_id: "abc123"
      ui_url: "https://mlflow.example.org"
    - url: "https://mlflow.example.org/#/experiments/12/runs/other"
  wandb:
    entity: my-team
    project: training
    run_id: abc123
    base_url: "https://wandb.example.org"
```

MLflow field-based links accept `tracking_uri` (HTTP(S) only), `ui_url`,
`experiment_id`, and `run_id` (omit the run ID for an explicit experiment link).
`ui_url` takes precedence over the configured
MLflow UI host, which takes precedence over `tracking_uri`. W&B accepts
`entity`, `project`, `run_id`, and `ui_url` or `base_url`; a per-record host takes
precedence over configuration. For either provider, `url` specifies the complete
run URL instead of constructing one from fields; an optional `label` names the
link in the multiple-run chooser. Only HTTP(S) URLs without
embedded credentials, whitespace, or control characters are emitted.

### Discovery limits

Discovery is best-effort: malformed, unreadable, oversized, or insufficient
metadata is ignored. The scan is bounded to eight directory levels, 10,000
directory entries, 256 candidate text files, and 256 KiB per file. Artifact and
checkpoint directories are skipped, as are directory symlinks; file symlinks
and paths resolving outside the job root are not read. Very large trees can
therefore yield incomplete results. Multiple runs are returned in stable order,
deduplicated by provider and URL.

The Python API is
`discover_tracking_links(root: Path, *, mlflow_ui_url=None, wandb_ui_url=None)`
in `foreman.tracking`, returning dictionaries with `provider`, `label`, and
`url`. Missing or inaccessible roots return an empty list.


### TensorBoard

The TensorBoard link appears only when a bounded scan below the job directory
finds a non-empty regular file named `events.out.tfevents.*`, including nested
output folders. Detection checks filenames and file metadata; it does not parse
event contents or guarantee that TensorBoard can read them. No TensorFlow
dependency is required.

Opening the link creates a local TensorBoard viewer for that job. The viewer is
reused on subsequent opens of the same job within one Foreman server process;
different jobs have separate viewers. Viewers and their loaders remain allocated
until Foreman restarts.

Embedded TensorBoard viewers are intended for **trusted local repositories**.
The button-detection scan skips symlinks, and the open endpoint validates the job
root, but TensorBoard's ongoing loader can follow nested symlinks or filesystem
changes outside that root. These checks are not a filesystem sandbox. Isolation
through a private event-file mirror is a separate follow-up.


## Versioning
Updates to Foreman happen sporadically based on my needs and time. The versioning is
based on the date of the last update in the format `YYYY.MM.DD`.

Have a look at the [HISTORY.md](HISTORY.md) for a list of changes.
