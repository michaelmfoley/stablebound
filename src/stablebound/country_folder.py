"""Relationship-table folders described by a ``stablebound.json`` file.

A country's relationship tables are maintained, and published, by whoever
builds them: a data repository, a deposit, a collaborator's folder. They come
with their own file names and their own column headers, and they come without
the facts the tables cannot state about themselves — whether the units are
states or districts, from when the record can be trusted, how far it has been
kept current, and how the country's names have to be tidied before they will
match a statistics file.

A ``stablebound.json`` placed next to the tables records exactly those things,
so the folder can be loaded as it is shipped, with no conversion step::

    from stablebound import Lineage
    ln = Lineage.from_folder("path/to/the/folder")

The file is plain JSON. Every path in it is relative to the folder holding it.
Keys beginning with ``_`` are comments and are ignored; any other key the
format does not define is an error, so a misspelt key cannot be silently
dropped. Example (India, abridged)::

    {
      "format": "stablebound-country/1",
      "country_code": "IN",
      "admin_level": 2,
      "validity_start_year": 1991,
      "coverage_end_year": 2025,
      "notes": "District-level lineage, 1991 baseline.",
      "files": {
        "relationship_table": "lineage_files/ADM2_LINEAGE_NEW_IDS_complete.xlsx",
        "coarse_lineage": "lineage_files/ADM1_LINEAGE.xlsx",
        "baseline": "admin_snapshots/admin_snapshot_1991.csv",
        "name_change_log": "lineage_files/NAME_CHANGE_LOG.xlsx"
      },
      "columns": {
        "baseline": {"ADM2_ID": "unit_id", "ADM2_NAME": "name",
                     "ADM1_ID": "coarse_id", "ADM1_NAME": "coarse_name"},
        "name_change_log": {"CHANGE_YEAR": "event_year",
                            "OLD_OFFICIAL_NAME": "old_name",
                            "NEW_OFFICIAL_NAME": "new_name"}
      },
      "name_rules": {
        "strip_before_lowercase": ["\\\\s*\\\\(([A-Z]{2,3})\\\\)\\\\s*$"],
        "remove_substrings": [" district", " dist.", " dist"],
        "remove_patterns": ["\\\\bdr\\\\.\\\\s*", "\\\\bsri\\\\s+", "\\\\bshri\\\\s+"]
      }
    }

``columns`` maps a file's own header to the canonical name; matching ignores
case, so headers that differ from the canonical ones only by case need no
entry. ``name_rules`` are applied in the order shown — patterns stripped
before lowercasing (a state abbreviation recognised by its capitals), then
substrings removed, then patterns removed — and finished by the package's
default :func:`stablebound.match.normalize_name`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .match import normalize_name

#: The file name looked for in a folder.
DESCRIPTION_FILE = "stablebound.json"

#: The format string this version of the package reads and writes. A later,
#: incompatible format gets a new number so an old reader refuses it loudly.
FORMAT = "stablebound-country/1"

#: The tables a folder can hold, in the order they are documented.
TABLES = ("relationship_table", "coarse_lineage", "baseline", "name_change_log")

#: The three kinds of name rule, in the order they are applied.
NAME_RULE_KINDS = ("strip_before_lowercase", "remove_substrings", "remove_patterns")

_TOP_LEVEL_KEYS = {
    "format",
    "country_code",
    "admin_level",
    "validity_start_year",
    "coverage_end_year",
    "notes",
    "files",
    "columns",
    "name_rules",
}


class CountryFolderError(ValueError):
    """A ``stablebound.json`` that cannot be read as written.

    Raised for a missing or unreadable file, a wrong ``format``, an unknown
    key, a wrong type, a table named that the format does not define, a file
    the description points at that is not there, or a name rule that is not a
    valid regular expression. The message names the file and the problem.
    """


def _drop_comments(d: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if not str(k).startswith("_")}


def _expect(cond: bool, where: Path, msg: str) -> None:
    if not cond:
        raise CountryFolderError(f"{where}: {msg}")


def normalizer_from_rules(rules: Mapping[str, Any] | None) -> Callable[[str], str]:
    """Build a name normalizer from rules written as data.

    ``rules`` has up to three lists, applied in this order:

    1. ``strip_before_lowercase`` — regular expressions removed from the name
       *before* it is lowercased, for anything recognised by its capitals.
       India's statistics append the state abbreviation, "Villupuram (TN)";
       the pattern can only tell "(TN)" from a genuine parenthetical while the
       capitals are still there.
    2. ``remove_substrings`` — literal text removed after lowercasing, in the
       order listed (" district", " dist.", " dist": the longer forms first,
       so " dist." is not left as a stray ".").
    3. ``remove_patterns`` — regular expressions removed after that
       (honorifics such as ``\\bsri\\s+``).

    The result is then passed through :func:`stablebound.match.normalize_name`,
    the tidying every country gets, so the rules only describe what makes
    this country different. With no rules (``None`` or ``{}``) the function
    *is* :func:`normalize_name`.
    """
    if not rules:
        return normalize_name
    unknown = sorted(set(_drop_comments(rules)) - set(NAME_RULE_KINDS))
    if unknown:
        raise CountryFolderError(
            f"name_rules has unknown kind(s) {unknown}; the kinds are {list(NAME_RULE_KINDS)}"
        )
    before: list[re.Pattern[str]] = []
    for pat in rules.get("strip_before_lowercase", []) or []:
        try:
            before.append(re.compile(str(pat)))
        except re.error as exc:
            raise CountryFolderError(
                f"name_rules.strip_before_lowercase: {pat!r} is not a valid pattern ({exc})"
            ) from exc
    substrings = [str(s) for s in (rules.get("remove_substrings", []) or [])]
    after: list[re.Pattern[str]] = []
    for pat in rules.get("remove_patterns", []) or []:
        try:
            after.append(re.compile(str(pat)))
        except re.error as exc:
            raise CountryFolderError(
                f"name_rules.remove_patterns: {pat!r} is not a valid pattern ({exc})"
            ) from exc

    def _normalize(s: str) -> str:
        out = str(s).strip()
        for p in before:
            out = p.sub("", out)
        out = out.lower().strip()
        for sub in substrings:
            out = out.replace(sub, "")
        for p in after:
            out = p.sub("", out)
        return normalize_name(out)

    _normalize.__doc__ = f"Name normalizer built from rules: {dict(rules)!r}"
    return _normalize


@dataclass(frozen=True)
class CountryDescription:
    """What a ``stablebound.json`` says, with its paths resolved.

    ``files`` holds absolute paths (resolved against the folder the
    description was read from). ``columns`` maps table name to a
    ``{file header: canonical name}`` mapping. ``name_rules`` is kept as
    written; :attr:`normalizer` builds the function from it.
    """

    folder: Path
    country_code: str
    files: dict[str, Path]
    admin_level: int | None = None
    validity_start_year: int | None = None
    coverage_end_year: int | None = None
    notes: str = ""
    columns: dict[str, dict[str, str]] = field(default_factory=dict)
    name_rules: dict[str, list[str]] | None = None

    @property
    def normalizer(self) -> Callable[[str], str] | None:
        """The name normalizer the rules describe, or ``None`` if there are none."""
        return normalizer_from_rules(self.name_rules) if self.name_rules else None

    def lineage_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for :class:`stablebound.Lineage` (all but the code)."""
        return {
            "relationship_table_path": self.files["relationship_table"],
            "coarse_lineage_path": self.files.get("coarse_lineage"),
            "baseline_path": self.files.get("baseline"),
            "name_change_log_path": self.files.get("name_change_log"),
            "admin_level": self.admin_level,
            "validity_start_year": self.validity_start_year,
            "coverage_end_year": self.coverage_end_year,
            "notes": self.notes or None,
            "normalizer": self.normalizer,
            "columns": self.columns or None,
        }

    def to_dict(self) -> dict[str, Any]:
        """The description as JSON-ready data, with paths relative to its folder.

        ``read_country_description`` of a folder holding ``json.dumps(to_dict())``
        gives back an equal description.
        """
        out: dict[str, Any] = {"format": FORMAT, "country_code": self.country_code}
        for key in ("admin_level", "validity_start_year", "coverage_end_year"):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        if self.notes:
            out["notes"] = self.notes
        out["files"] = {
            t: self.files[t].relative_to(self.folder).as_posix()
            for t in TABLES
            if t in self.files
        }
        if self.columns:
            out["columns"] = {t: dict(m) for t, m in self.columns.items()}
        if self.name_rules:
            out["name_rules"] = {k: list(v) for k, v in self.name_rules.items()}
        return out


def read_country_description(folder: Path | str) -> CountryDescription:
    """Read and check the ``stablebound.json`` in ``folder``.

    ``folder`` may also be the path of the JSON file itself. Every check that
    can be made without reading the tables is made here — format, keys,
    types, that each named file exists, that each name rule compiles — so a
    broken description fails at once with a message naming the problem,
    instead of partway through an analysis.
    """
    p = Path(folder)
    path = p if p.is_file() else p / DESCRIPTION_FILE
    _expect(path.is_file(), path, f"no {DESCRIPTION_FILE} here")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CountryFolderError(f"{path}: not valid JSON ({exc})") from exc
    _expect(isinstance(raw, dict), path, "the top level must be a JSON object")
    data = _drop_comments(raw)

    unknown = sorted(set(data) - _TOP_LEVEL_KEYS)
    _expect(not unknown, path, f"unknown key(s) {unknown}; the keys are {sorted(_TOP_LEVEL_KEYS)}")
    _expect(
        data.get("format") == FORMAT,
        path,
        f"format is {data.get('format')!r}; this version of StableBound reads {FORMAT!r}",
    )
    code = data.get("country_code")
    _expect(isinstance(code, str) and bool(code.strip()), path, "country_code must be a string")
    assert isinstance(code, str)

    for key in ("admin_level", "validity_start_year", "coverage_end_year"):
        v = data.get(key)
        _expect(
            v is None or (isinstance(v, int) and not isinstance(v, bool)),
            path,
            f"{key} must be a whole number, got {v!r}",
        )
    if data.get("admin_level") is not None:
        _expect(data["admin_level"] >= 1, path, "admin_level must be 1 or more")
    notes = data.get("notes", "")
    _expect(isinstance(notes, str), path, "notes must be a string")

    files_raw = data.get("files")
    _expect(isinstance(files_raw, dict), path, "files must be an object naming the tables")
    assert isinstance(files_raw, dict)
    files_raw = _drop_comments(files_raw)
    bad_tables = sorted(set(files_raw) - set(TABLES))
    _expect(not bad_tables, path, f"files names unknown table(s) {bad_tables}; "
            f"the tables are {list(TABLES)}")
    _expect("relationship_table" in files_raw, path, "files.relationship_table is required")
    base = path.parent
    files: dict[str, Path] = {}
    for table, rel in files_raw.items():
        _expect(isinstance(rel, str), path, f"files.{table} must be a path string")
        f = (base / rel).resolve()
        _expect(f.is_file(), path, f"files.{table} points at {f}, which does not exist")
        files[table] = f
    if "coarse_lineage" in files:
        _expect(
            (data.get("admin_level") or 1) >= 2,
            path,
            "files.coarse_lineage is given but admin_level is not 2 or more; a coarse "
            "lineage describes the level above the working one",
        )

    columns_given = data.get("columns") or {}
    _expect(isinstance(columns_given, dict), path, "columns must be an object")
    columns_raw = _drop_comments(columns_given)
    columns: dict[str, dict[str, str]] = {}
    for table, mapping in columns_raw.items():
        _expect(table in TABLES, path, f"columns names unknown table {table!r}; "
                f"the tables are {list(TABLES)}")
        _expect(table in files, path, f"columns.{table} is given but files.{table} is not")
        _expect(isinstance(mapping, dict), path, f"columns.{table} must be an object")
        mapping = _drop_comments(mapping)
        _expect(
            all(isinstance(k, str) and isinstance(v, str) for k, v in mapping.items()),
            path,
            f"columns.{table} must map header strings to canonical-name strings",
        )
        columns[table] = dict(mapping)

    rules_raw = data.get("name_rules")
    name_rules: dict[str, list[str]] | None = None
    if rules_raw is not None:
        _expect(isinstance(rules_raw, dict), path, "name_rules must be an object")
        rules_raw = _drop_comments(rules_raw)
        for kind, values in rules_raw.items():
            _expect(
                isinstance(values, list) and all(isinstance(x, str) for x in values),
                path,
                f"name_rules.{kind} must be a list of strings",
            )
        try:
            normalizer_from_rules(rules_raw)  # compiles every pattern; raises if bad
        except CountryFolderError as exc:
            raise CountryFolderError(f"{path}: {exc}") from exc
        name_rules = {k: list(v) for k, v in rules_raw.items()}

    return CountryDescription(
        folder=base.resolve(),
        country_code=code.strip().upper(),
        files=files,
        admin_level=data.get("admin_level"),
        validity_start_year=data.get("validity_start_year"),
        coverage_end_year=data.get("coverage_end_year"),
        notes=notes,
        columns=columns,
        name_rules=name_rules,
    )


def write_country_description(description: CountryDescription, path: Path | str) -> Path:
    """Write ``description`` as JSON to ``path`` (a folder or a file path).

    The paths are written relative to the description's own folder, so the
    file belongs next to the tables it describes.
    """
    p = Path(path)
    out = p / DESCRIPTION_FILE if p.is_dir() else p
    out.write_text(json.dumps(description.to_dict(), indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    return out


__all__ = [
    "DESCRIPTION_FILE",
    "FORMAT",
    "CountryDescription",
    "CountryFolderError",
    "normalizer_from_rules",
    "read_country_description",
    "write_country_description",
]
