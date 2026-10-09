"""Links to the git repositories (code) a job depends on.

R3 records the exact commit of each git dependency, but not the branch it came
from. Branches that contain the commit are looked up in R3's local mirror.
"""

import logging
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from threading import Lock
from typing import Optional

import r3

LOGGER = logging.getLogger(__name__)

# The repository URLs R3 supports (see r3.GitDependency.repository_path).
_GITHUB_URL = re.compile(
    r"^(?:https://github\.com/|git@github\.com:)([^/]+)/([^/]+?)(?:\.git)?$"
)
# Branches move, so lookups are only reused for a while.
BRANCH_CACHE_SECONDS = 600


@dataclass(frozen=True)
class RepositoryLink:
    """A git dependency of a job, with links to view it on GitHub."""

    repository: str
    commit: Optional[str]
    # Path within the repository that was checked out, and where to in the job.
    source: str
    destination: str
    # "owner/name" and "https://github.com/owner/name", if hosted on GitHub.
    name: str
    web_url: Optional[str]
    # R3's local mirror of the repository, if present.
    mirror: Optional[Path]

    @property
    def short_commit(self) -> str:
        return (self.commit or "")[:7]

    @property
    def commit_url(self) -> Optional[str]:
        if not self.web_url or not self.commit:
            return None
        return f"{self.web_url}/commit/{self.commit}"

    @property
    def tree_url(self) -> Optional[str]:
        """The code at the job's commit, at the checked-out path."""
        if not self.web_url or not self.commit:
            return None
        url = f"{self.web_url}/tree/{self.commit}"
        if self.source not in ("", "."):
            url += f"/{self.source}"
        return url

    def compare_url(self, branch: str) -> Optional[str]:
        """Changes on a branch since the job's commit."""
        if not self.web_url or not self.commit:
            return None
        return f"{self.web_url}/compare/{self.commit}...{branch}"


def repository_links(
    r3_job: r3.Job, r3_root: Optional[Path] = None,
) -> list[RepositoryLink]:
    """Returns the git dependencies of a job, in the order R3 lists them."""
    links = []
    for dependency in r3_job.dependencies:
        if not isinstance(dependency, r3.GitDependency):
            continue
        match = _GITHUB_URL.match(dependency.repository)
        owner, name = match.groups() if match else (None, None)
        mirror = None
        if r3_root is not None and match:
            candidate = Path(r3_root) / "git" / "github.com" / owner / name
            mirror = candidate if candidate.is_dir() else None
        source = str(PurePosixPath(dependency.source))
        links.append(RepositoryLink(
            repository=dependency.repository,
            commit=dependency.commit,
            source="" if source == "." else source,
            destination=str(dependency.destination),
            name=f"{owner}/{name}" if match else dependency.repository,
            web_url=f"https://github.com/{owner}/{name}" if match else None,
            mirror=mirror,
        ))
    return links


_branch_cache: dict[tuple[Path, str], tuple[float, list[str]]] = {}
_branch_cache_lock = Lock()


def branches_containing(mirror: Path, commit: str) -> list[str]:
    """Returns the branches that contain a commit, the default branch first."""
    key = (mirror, commit)
    with _branch_cache_lock:
        cached = _branch_cache.get(key)
    if cached and time.monotonic() - cached[0] < BRANCH_CACHE_SECONDS:
        return cached[1]

    try:
        result = subprocess.run(
            ["git", "-C", str(mirror), "for-each-ref", "--contains", commit,
             "--format=%(refname:short)", "refs/heads"],
            capture_output=True, text=True, timeout=30, check=True,
        )
        default = subprocess.run(
            ["git", "-C", str(mirror), "symbolic-ref", "--short", "HEAD"],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        LOGGER.warning("Could not find branches of %s in %s: %s", commit, mirror, error)
        return []
    branches = sorted(
        result.stdout.split(), key=lambda branch: (branch != default, branch),
    )
    with _branch_cache_lock:
        _branch_cache[key] = (time.monotonic(), branches)
    return branches
