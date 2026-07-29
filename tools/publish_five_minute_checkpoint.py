"""Publish the small five-minute paper checkpoint without publishing market data.

The active endpoint cache can be tens of megabytes and belongs in Actions
cache storage.  The checkpoint is deliberately tiny and is committed after a
new five-minute event so an interrupted handoff can resume without submitting
the same target twice.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any


def _read_checkpoint(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"checkpoint {path} must contain a JSON object")
    if not isinstance(data.get("completed_events", []), list) or not isinstance(data.get("ledger", []), list):
        raise ValueError(f"checkpoint {path} has an invalid shape")
    return data


def _unique_entries(entries: list[object]) -> list[object]:
    result: list[object] = []
    seen: set[str] = set()
    for entry in entries:
        key = json.dumps(entry, sort_keys=True, separators=(",", ":"), default=str)
        if key not in seen:
            result.append(entry)
            seen.add(key)
    return result


def merge_checkpoints(remote: dict[str, Any], local: dict[str, Any]) -> dict[str, Any]:
    """Merge same-day event ledgers without overwriting a newer session."""
    remote_day = str(remote.get("session_date", ""))
    local_day = str(local.get("session_date", ""))
    if remote_day != local_day:
        return local if local_day >= remote_day else remote
    return {
        "format_version": max(int(remote.get("format_version", 1)), int(local.get("format_version", 1))),
        "session_date": local_day or remote_day,
        "completed_events": sorted(set(remote.get("completed_events", [])) | set(local.get("completed_events", []))),
        "ledger": _unique_entries(list(remote.get("ledger", [])) + list(local.get("ledger", []))),
    }


def _run(command: list[str], *, cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def publish_checkpoint(
    source: Path,
    *,
    repository: Path,
    branch: str,
    retries: int = 3,
) -> bool:
    """Merge and push one checkpoint, returning false when no state exists."""
    if not source.is_file():
        print("FIVE MINUTE CHECKPOINT PUBLISH SKIPPED | no local checkpoint", flush=True)
        return False
    local = _read_checkpoint(source)
    remote_relative = Path("var/five_minute_paper_24x7_state.json")
    for attempt in range(1, retries + 1):
        staging_root = Path(tempfile.mkdtemp(prefix="cufolio-checkpoint-publish-"))
        worktree = staging_root / "checkout"
        try:
            _run(["git", "fetch", "origin", branch], cwd=repository)
            _run(["git", "worktree", "add", "--detach", str(worktree), f"origin/{branch}"], cwd=repository)
            remote_path = worktree / remote_relative
            remote = _read_checkpoint(remote_path) if remote_path.is_file() else {}
            merged = merge_checkpoints(remote, local)
            remote_path.parent.mkdir(parents=True, exist_ok=True)
            remote_path.write_text(json.dumps(merged, indent=2, default=str) + "\n", encoding="utf-8")
            _run(["git", "add", "-f", str(remote_relative)], cwd=worktree)
            changed = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=worktree).returncode != 0
            if not changed:
                print("FIVE MINUTE CHECKPOINT PUBLISH CURRENT | remote already contains this state", flush=True)
                return True
            _run(["git", "config", "user.name", "cufolio five-minute paper runner"], cwd=worktree)
            _run(["git", "config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com"], cwd=worktree)
            _run(["git", "commit", "-m", "chore: checkpoint five-minute paper session [skip ci]"], cwd=worktree)
            _run(["git", "push", "origin", f"HEAD:{branch}"], cwd=worktree)
            print(
                "FIVE MINUTE CHECKPOINT PUBLISHED | "
                f"session={merged.get('session_date', 'unknown')} completed_events={len(merged['completed_events'])}",
                flush=True,
            )
            return True
        except subprocess.CalledProcessError as error:
            if attempt == retries:
                raise RuntimeError(f"checkpoint publish failed after {attempt} attempts") from error
            print(
                f"FIVE MINUTE CHECKPOINT PUBLISH RETRY | attempt={attempt}/{retries} reason={error}",
                flush=True,
            )
            time.sleep(float(attempt))
        finally:
            if worktree.exists():
                subprocess.run(["git", "worktree", "remove", "--force", str(worktree)], cwd=repository, check=False)
            shutil.rmtree(staging_root, ignore_errors=True)
            subprocess.run(["git", "worktree", "prune"], cwd=repository, check=False)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish a five-minute paper checkpoint to a Git branch.")
    parser.add_argument("--source", default="var/five_minute_paper_24x7_state.json")
    parser.add_argument("--repository", default=".")
    parser.add_argument("--branch", default="main")
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()
    if args.retries < 1:
        parser.error("--retries must be positive")
    publish_checkpoint(Path(args.source), repository=Path(args.repository).resolve(), branch=args.branch, retries=args.retries)


if __name__ == "__main__":
    main()
