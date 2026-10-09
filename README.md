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


## Editing job tags
On a job's detail page, choose **Edit tags**, enter one tag per line, and select
**Save tags**. Add, remove, or rename tags by changing the lines; **Cancel**
discards unsaved edits. Whitespace, blank lines, and duplicate tags are removed,
while order and case are preserved. Saving an empty list clears all tags.

Changes apply only to that job's tags and immediately refresh R3's search index.
The Foreman process needs write access to both the job's `metadata.yaml` and the
repository index. This uses R3's existing metadata and private index-update APIs;
file and index writes are not atomic, and concurrent edits can overwrite changes.


## Versioning
Updates to Foreman happen sporadically based on my needs and time. The versioning is
based on the date of the last update in the format `YYYY.MM.DD`.

Have a look at the [HISTORY.md](HISTORY.md) for a list of changes.
