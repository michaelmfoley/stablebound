# tools/india — regenerating the bundled India data

These scripts turn the canonical, hand-authored India sources into the files
the package ships and the shapefile the India pipeline uses. They are
developer tooling, not part of the installed package, and they need the
private data tree:

```sh
export STABLEBOUND_DATA_ROOT=/path/to/Crops      # the directory holding data/india/...
python tools/india/bundle_data.py --check          # verify src/stablebound/data/IN/* matches the sources
python tools/india/bundle_data.py                  # regenerate them
python tools/india/prepare_geolocet.py             # trim the Geolocet product; add the `state` column
python tools/india/prepare_shapefile.py            # match Geolocet districts to lineage ids -> _prepared/
python tools/india/prepare_inputs.py               # reshape the raw statistics for the regression harness
```

| script | produces |
|---|---|
| `bundle_data.py` | `src/stablebound/data/IN/{lineage.xlsx, lineage_adm1.xlsx, baseline.csv, name_change_log.xlsx}` |
| `prepare_geolocet.py` | the publishable Geolocet-derived district layer with a `state` column |
| `prepare_shapefile.py` | `_prepared/India_modern_with_ids.geojson` and `match_log.csv` (ids attached at the map's 2024 vintage) |
| `prepare_inputs.py` | `_prepared/{baseline,stats_long}.csv` |
| `india_overrides.py` | the hand-confirmed name aliases `prepare_shapefile.py` applies |
| `paths.py`, `sources.toml` | resolve and checksum-pin the external inputs (`python tools/india/paths.py --pin` refreshes the pins) |

`_prepared/` and `_out/` here are gitignored. The published pipeline that
consumes these outputs is the separate `stablebound-india` repository; the
shipped `India_modern_with_ids.geojson` lives there.

History: these scripts lived in `regression/india/` until v0.1.4, when the
regression harnesses were archived under `experimental/regression/` and the
release was scoped to India.
