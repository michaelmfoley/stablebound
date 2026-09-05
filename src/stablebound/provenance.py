"""Record what an artifact was built from, so it can be traced back.

A file handed to FEWS is opaque: six months later, "which lineage produced
this?" has no answer beyond the modification date. This records the package
version, the git commit, and a fingerprint of the inputs, so any shipped
artifact can be tied to an exact source state.

**Written as a sidecar, never embedded.** Stamping a version string into the
deliverables themselves would change their bytes, and India's FEWS bundle is
verified by byte-for-byte comparison against a frozen reference — embedding
provenance would break the strongest correctness guard in the project in order
to add a weaker one. So :func:`write_provenance` drops a ``provenance.json``
*next to* the artifacts and leaves them untouched.

    from stablebound.provenance import write_provenance
    write_provenance(out_dir, inputs={"lineage": path, "baseline": path})

The record deliberately contains no timestamp. A timestamp would make every
run's provenance differ, which defeats using it to tell whether two runs are
the same. Use the file's own mtime if you need "when".
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Mapping


def _git(*args: str, cwd: Path | None = None) -> str | None:
    """A git command's output, or None outside a repo / without git."""
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=str(cwd) if cwd else None,
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def git_state(repo: Path | None = None) -> dict:
    """Commit, branch, and whether the tree was dirty.

    ``dirty`` matters more than the SHA: an artifact built from a modified tree
    cannot be reproduced from the commit alone, and silently recording only the
    SHA would imply otherwise.
    """
    root = repo or Path(__file__).resolve().parents[2]
    sha = _git("rev-parse", "HEAD", cwd=root)
    # Not a checkout at all — an installed copy, most likely. Record that
    # nothing is known rather than leaving the fields out, so a reader can
    # tell "not tracked" apart from "we forgot to look".
    if sha is None:
        return {"commit": None, "branch": None, "dirty": None}
    # Whether anything was edited but not committed. This matters more than
    # the commit itself: a result built from a modified copy cannot be
    # reproduced from that commit, and recording only the commit would
    # quietly claim it could.
    status = _git("status", "--porcelain", cwd=root)
    return {
        "commit": sha,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD", cwd=root),
        "dirty": bool(status),
    }


def _sha256(path: Path, *, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def fingerprint_inputs(inputs: Mapping[str, Path | str]) -> dict:
    """``{label: sha256}`` for each input file, ``"MISSING"`` when absent.

    Plus ``_combined``: a hash over the sorted (label, digest) pairs, so two
    runs can be compared with one string instead of a dict.
    """
    out: dict[str, str] = {}
    for label, p in sorted(inputs.items()):
        path = Path(p)
        out[label] = _sha256(path) if path.is_file() else "MISSING"
    combined = hashlib.sha256(
        "\n".join(f"{k}={v}" for k, v in sorted(out.items())).encode()
    ).hexdigest()
    out["_combined"] = combined
    return out


def provenance_record(
    *,
    inputs: Mapping[str, Path | str] | None = None,
    extra: Mapping[str, object] | None = None,
    repo: Path | None = None,
) -> dict:
    """The full record. Deterministic: same code + same inputs, same record."""
    try:
        from . import __version__ as version
    except ImportError:  # pragma: no cover
        version = None
    # Everything needed to trace a published file back to what produced it:
    # which version of the package, which state of the source, and which
    # interpreter and system it ran on. Deliberately no timestamp — two runs
    # of the same code over the same inputs should produce identical records,
    # and a clock reading would make every run look different.
    record: dict = {
        "stablebound_version": version,
        "git": git_state(repo),
        "python": platform.python_version(),
        "platform": platform.system(),
    }
    # Plus a fingerprint of the input files themselves, which is what catches
    # someone quietly editing a boundary record between runs.
    if inputs:
        record["inputs"] = fingerprint_inputs(inputs)
    if extra:
        record.update(dict(extra))
    return record


def write_provenance(
    out_dir: Path | str,
    *,
    inputs: Mapping[str, Path | str] | None = None,
    extra: Mapping[str, object] | None = None,
    filename: str = "provenance.json",
) -> Path:
    """Write the record beside the artifacts. Returns the path written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / filename
    target.write_text(
        json.dumps(provenance_record(inputs=inputs, extra=extra),
                   indent=1, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target
