"""Project-local training artifact paths and reproducibility metadata."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import subprocess
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
LOG_ROOT = PROJECT_ROOT / "logs"
CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints"
EXPORT_ROOT = PROJECT_ROOT / "exported"

ISAAC_LAB_ROOT = Path(
    os.environ.get("ISAACLAB_ROOT", Path.home() / "IsaacLab")
).expanduser().resolve()
TARGET_ISAAC_LAB_VERSION = "v2.3.2"
TARGET_ISAAC_SIM_VERSION = "5.1.0"
ROLLBACK_ISAAC_LAB_COMMIT = "cbf51abb5e98d1b3d497c8c73dc989e9f3628b89"


def assert_project_local_path(path: str | Path) -> Path:
    """Resolve a path and fail if it escapes the project root."""
    resolved = Path(path).expanduser().resolve()
    if resolved != PROJECT_ROOT and PROJECT_ROOT not in resolved.parents:
        raise ValueError(f"Artifact path escapes project root: {resolved}")
    return resolved


def git_revision(repository: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        capture_output=True,
        check=False,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _git_output(repository: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(repository), *args],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode(errors="replace").strip())
    return result.stdout


def _project_worktree_fingerprint(repository: Path) -> dict[str, Any]:
    """Fingerprint tracked edits and project-owned untracked source files."""
    status = _git_output(repository, "status", "--porcelain=v1", "--untracked-files=all")
    tracked_diff = _git_output(repository, "diff", "--binary", "HEAD", "--")
    relevant_status: list[str] = []
    untracked: dict[str, str] = {}
    ignored_runtime_roots = {"build", "install", "log", "logs", "outputs"}
    for line in status.decode(errors="surrogateescape").splitlines():
        if not line.startswith("?? "):
            relevant_status.append(line)
            continue
        relative = line[3:]
        if Path(relative).parts[0] in ignored_runtime_roots:
            continue
        relevant_status.append(line)
        path = repository / relative
        if path.is_file():
            untracked[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "dirty": bool(relevant_status),
        "status_sha256": hashlib.sha256("\n".join(relevant_status).encode()).hexdigest(),
        "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
        "untracked_file_sha256": untracked,
    }


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def build_run_manifest(*, task_id: str, seed: int, log_dir: str | Path) -> dict[str, Any]:
    """Build metadata saved beside every future RSL-RL run."""
    local_log_dir = assert_project_local_path(log_dir)
    project_revision = git_revision(PROJECT_ROOT)
    if project_revision is None:
        raise RuntimeError("Training requires a committed project Git revision")
    return {
        "task_id": task_id,
        "seed": seed,
        "project_root": str(PROJECT_ROOT),
        "log_dir": str(local_log_dir),
        "project_git_commit": project_revision,
        "project_worktree": _project_worktree_fingerprint(PROJECT_ROOT),
        "isaac_lab_target": TARGET_ISAAC_LAB_VERSION,
        "isaac_lab_git_commit": git_revision(ISAAC_LAB_ROOT),
        "isaac_sim_target": TARGET_ISAAC_SIM_VERSION,
        "rollback_isaac_lab_commit": ROLLBACK_ISAAC_LAB_COMMIT,
        "installed_distributions": {
            "isaaclab": _package_version("isaaclab"),
            "isaaclab_tasks": _package_version("isaaclab-tasks"),
            "isaaclab_rl": _package_version("isaaclab-rl"),
            "rsl_rl": _package_version("rsl-rl-lib"),
        },
    }
