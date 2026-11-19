# India, as shipped — a frozen test fixture

These five files are India's district relationship tables exactly as the
StableBound-India dataset publishes them in its Zenodo deposit
(`intermediates/`), with their own file names and UPPERCASE headers, plus the
`stablebound.json` that describes them for `Lineage.from_folder`.

**Frozen. Never edit these files.** They are test data, not the India master.
The master is maintained in the StableBound-India repository, under
`lineage/`. A correction to India's boundaries is made there and published
with that dataset; this copy is replaced only deliberately, as a reviewed
change to the tests, with the hashes below updated.

| file | sha256 |
|---|---|
| `lineage_adm2.xlsx` | `34032a6ebe7f55c3b1a7a4129cd75dc8bee5bc0da371492a77489fb451efe434` |
| `lineage_adm1.xlsx` | `0574b337501d8d258a9af4ea31d50e697405ca32fe6e2070d1b67f2d5db0b9f8` |
| `name_changes.xlsx` | `d1cd89d2637dcded64e4bced6c28d9a4db54501f1129c10ed78f65f41029e62b` |
| `baseline_1991.csv` | `6b08145b2bdf9aa63a6603b34ad0e577ed785384ea7e563425b9adcc01016061` |
| `stablebound.json` | `3fbbd0345c40aad8ef6e2b9f6739aaec1bf88c50ef3d78854e32bae7ea20a892` |

Where it came from (2026-11-19): a deposit built by StableBound-India's
`package_deposit.py` from its repository at commit `0e72fd9`. The four tables
are byte-identical to that commit's `lineage/lineage_files/` and
`lineage/admin_snapshots/admin_snapshot_1991.csv`; `stablebound.json` is the
deposit's, derived from the master's `lineage/stablebound.json` with the
deposit's file names.

Until 0.2.0 the package carried a converted copy of these tables (lowercase
headers) inside `src/stablebound/data/IN/` and loaded it as `Lineage("IN")`.
Loaded from this folder, they give the same lineage on every fact, table,
graph, validation finding, yearly snapshot (1991–2025), normalised name and
`export_fews` file; `tests/india_reference.py` holds the numbers recorded
from that copy before it was removed.
