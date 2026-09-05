# Legacy-dialect relationship table fixture

`relationshiptable_XX.csv` is a **hand-authored** relationship table in the
11-column dialect FEWS NET distributes (`category` / `relationship_type` rows,
FNID-keyed). It describes the same country as the canonical synthetic fixture
`tests/fixtures/synthetic/legacy_admin1/`: a four-unit admin-1 country with a
split (2015), a rename carried as a same-id successor (2020) and a merge (2025)
that makes the unit count fall.

The two files are each other's oracle. `test_rt_convert.py` converts this file
and asserts the result equals the canonical fixture row for row;
`test_legacy_relationship.py` writes the canonical fixture in the legacy
dialect and asserts the vintage structure matches this file. Neither file is
generated from the other, and this one must stay hand-written: producing it
with `build_legacy_relationship_table` would make the round-trip tests compare
the exporter with itself.

Row blocks, in order:

- 18 `hierarchical` / `admin1_0` rows: one per unit per vintage (4 at 2010,
  5 at 2015, 5 at 2020, 4 at 2025). Names carry the `, Exampleland` suffix so
  the reader's comma-strip is exercised. FNIDs follow `XX{vintage}A1{ss}`;
  a continuing unit keeps its `ss` code across vintages (Charlton keeps
  Charlie's), a newly created unit gets a fresh one.
- 15 `temporal` rows between consecutive vintages: 2 `split`, 2 `merge`,
  11 `successor`, one of which (Charlie -> Charlton) has differing names and
  is what the converter turns into a `NameChange`. The dialect also allows an
  explicit `name change` row for this, and that is the form the package's own
  writer emits; this file deliberately uses the other form so both reader
  branches stay covered. Successor rows are in chronological order, which the
  converter's chain collapsing relies on.

Expected converter output: ids `XX.ADM1.00001`-`00006` and `00008` (`00007`
is allocated to the rename and then collapsed onto `00003`), an 18-entry FNID
map, baseline year 2010, events 2015-2025, coverage end 2025.
