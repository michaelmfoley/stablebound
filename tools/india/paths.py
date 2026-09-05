"""One place that knows where the external data lives.

The canonical India sources (the hand-authored lineage workbooks, the DESAGRI
statistics, the Geolocet shapefile) live outside this repository. Every script
under ``tools/india/`` resolves them from ``$STABLEBOUND_DATA_ROOT``; with the
variable unset, the first missing input raises :class:`MissingDataError`
naming the variable rather than a path on someone else's machine.

    export STABLEBOUND_DATA_ROOT=/path/to/the/source/tree
    python tools/india/bundle_data.py --check


Why the data is not simply vendored: the India statistics alone are 376k rows,
GAUL L2 is 43,819 global features, and the FEWS relationship tables are not ours
to redistribute. So the contract is *pinned, not bundled* — see
``sources.toml``, which records a sha256 and the expected columns for every
external input, and ``verify_sources()`` below, which fails loudly with both
hashes when a file drifts.

Scripts should call :func:`require` rather than touching :data:`DATA_ROOT`, so a
missing input produces one clear message naming the environment variable instead
of a ``FileNotFoundError`` from three frames deep.
"""

from __future__ import annotations

import hashlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path

try:  # 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - 3.10
    import tomli as tomllib  # type: ignore[no-redef]

#: Repo root — the directory containing `src/`, `tools/`, `examples/`. This
#: file lives two levels down, in tools/india/.
REPO = Path(__file__).resolve().parents[2]

#: Where the external (non-bundled) source data lives.
DATA_ROOT = Path(os.environ.get("STABLEBOUND_DATA_ROOT", "STABLEBOUND_DATA_ROOT-unset")).expanduser()

SOURCES_TOML = Path(__file__).resolve().parent / "sources.toml"


class MissingDataError(FileNotFoundError):
    """An external input is absent. Names the env var rather than the frame."""


class SourceDriftError(RuntimeError):
    """A pinned source file no longer matches its recorded checksum."""


def resolve(relative: str | Path) -> Path:
    """A path under the data root. Does not check existence."""
    return DATA_ROOT / Path(relative)


def require(relative: str | Path, *, what: str = "") -> Path:
    """A path under the data root that must exist.

    Raises :class:`MissingDataError` naming ``STABLEBOUND_DATA_ROOT`` and the
    missing file, so someone on a fresh checkout learns what to set rather than
    what line crashed.
    """
    p = resolve(relative)
    if not p.exists():
        raise MissingDataError(
            f"required input not found: {p}\n"
            f"  {'(' + what + ')' if what else ''}\n"
            f"  STABLEBOUND_DATA_ROOT is currently {DATA_ROOT}\n"
            f"  Set it to the directory containing 'inbox/', 'data/' and "
            f"'scripts/', or place the file at the path above."
        )
    return p


def available(relative: str | Path) -> bool:
    """True when an optional input is present, for skip messages."""
    return resolve(relative).exists()


def sha256(path: Path, *, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class Source:
    """One pinned external input."""

    key: str
    path: str
    sha256: str | None
    columns: tuple[str, ...]
    description: str

    @property
    def resolved(self) -> Path:
        return resolve(self.path)


def load_sources(toml_path: Path | None = None) -> dict[str, Source]:
    """Parse ``sources.toml`` into ``{key: Source}``."""
    p = toml_path or SOURCES_TOML
    if not p.exists():
        return {}
    with open(p, "rb") as fh:
        raw = tomllib.load(fh)
    out: dict[str, Source] = {}
    for key, spec in raw.get("source", {}).items():
        out[key] = Source(
            key=key,
            path=spec["path"],
            sha256=spec.get("sha256") or None,
            columns=tuple(spec.get("columns", ())),
            description=spec.get("description", ""),
        )
    return out


def verify_sources(
    keys: list[str] | None = None,
    *,
    toml_path: Path | None = None,
    strict: bool = True,
) -> list[str]:
    """Check every pinned source against its checksum and column contract.

    Returns a list of human-readable problems (empty when clean). With
    ``strict``, an unpinned source (``sha256 = ""``) is reported so a file
    cannot sit un-pinned indefinitely; missing files are always reported.
    """
    problems: list[str] = []
    sources = load_sources(toml_path)
    for key, src in sorted(sources.items()):
        if keys and key not in keys:
            continue
        p = src.resolved
        if not p.exists():
            problems.append(f"{key}: MISSING at {p}")
            continue
        if not src.sha256:
            if strict:
                problems.append(
                    f"{key}: not pinned (sha256 empty) — current value is "
                    f"{sha256(p)}"
                )
            continue
        actual = sha256(p)
        if actual != src.sha256:
            problems.append(
                f"{key}: CHECKSUM DRIFT at {p}\n"
                f"    expected {src.sha256}\n"
                f"    actual   {actual}"
            )
            continue
        if src.columns:
            missing = _missing_columns(p, src.columns)
            if missing:
                problems.append(f"{key}: missing expected column(s) {missing}")
    return problems


def _missing_columns(path: Path, expected: tuple[str, ...]) -> list[str]:
    """Header check without loading the file. Best-effort by extension."""
    import csv

    suffix = path.suffix.lower()
    try:
        if suffix in (".csv", ".txt"):
            with open(path, newline="", encoding="utf-8", errors="replace") as fh:
                header = next(csv.reader(fh), [])
        elif suffix in (".xlsx", ".xls"):
            import pandas as pd

            header = list(pd.read_excel(path, nrows=0).columns)
        else:
            return []
    except Exception:  # noqa: BLE001 - a read failure is reported by the caller
        return []
    have = {str(c).strip().lower() for c in header}
    return [c for c in expected if c.strip().lower() not in have]


def pin_all(toml_path: Path | None = None) -> str:
    """Recompute every checksum and return an updated ``sources.toml`` body.

    Used to (re)pin after a deliberate source update. Never called
    automatically — a silent re-pin would defeat the point of pinning.
    """
    p = toml_path or SOURCES_TOML
    lines = p.read_text(encoding="utf-8").splitlines()
    sources = load_sources(p)
    out: list[str] = []
    current: str | None = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[source."):
            current = stripped[len("[source.") : -1].strip('"')
        if stripped.startswith("sha256") and current in sources:
            src = sources[current]
            if src.resolved.exists():
                out.append(f'sha256 = "{sha256(src.resolved)}"')
                continue
        out.append(line)
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    """``python tools/india/paths.py [--pin]`` — report or refresh pins."""
    argv = sys.argv[1:] if argv is None else argv
    if "--pin" in argv:
        body = pin_all()
        SOURCES_TOML.write_text(body, encoding="utf-8")
        print(f"re-pinned {SOURCES_TOML}")
        return 0

    print(f"STABLEBOUND_DATA_ROOT = {DATA_ROOT}")
    print(f"exists: {DATA_ROOT.exists()}")
    sources = load_sources()
    print(f"\n{len(sources)} pinned source(s):")
    for key, src in sorted(sources.items()):
        state = "OK " if src.resolved.exists() else "MISS"
        pin = "pinned" if src.sha256 else "UNPINNED"
        print(f"  [{state}] {key:<28} {pin:<9} {src.path}")
    problems = verify_sources()
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nall sources verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
