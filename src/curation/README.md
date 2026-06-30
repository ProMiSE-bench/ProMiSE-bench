# Curation (`src/curation/`)

Build the ProMiSE benchmark from PDB mmCIFs: clustering, crystal filtering, pair extraction, redundancy removal.

Prerequisites: `bash install.sh` (`promise` env; step 5 uses `prodigy-cryst` env if available).

---

## Quick start

```bash
conda activate promise

promise_data run \
  --spec data/clusters.json \
  --mmcif-store /path/to/pdb_mmcif/mmcif_files
```

| Option | Description |
|--------|-------------|
| `--spec` | Cluster spec JSON (GroupSet) |
| `--mmcif-store` | Flat mmCIF directory (`*.cif`) |
| `--keep-intermediates` | Keep `asms-*`, `combinations/`, etc. under `data/` (default: temp dir, deleted at end) |
| `-C / --data-root` | Custom data root instead of `data/` |

Final output: `data/dataset-pipeline/`. List steps: `promise_data steps`.

### Partial runs

```bash
promise_data run --spec data/clusters.json --mmcif-store /path --start-from curate_sets
promise_data run --spec data/clusters.json --mmcif-store /path --stop-after cluster_by_tmscore
```

### Single step

```bash
python -m curation.pipeline.create_msa --help
python -m curation.pipeline.curate_sets --help
```

---

## Pipeline steps

| # | Step | Key outputs |
|---|------|-------------|
| 1 | `create_msa` | `data/msas/`, `data/coords/` |
| 2 | `pairwise_tm` | `data/scores/` |
| 3 | `cluster_by_tmscore` | `data/clusters/`, `filtered-pairs.csv` |
| 4 | `prepare_inputs` | `asms-raw/`, `cif-asms/` |
| 5 | `run_prodigy` | `pair-calls.csv` |
| 6 | `filter_xtal` | `asms-bio/` |
| 7 | `subsets` | `asms-subset/` |
| 8 | `process_metal` | `asms-metal/` |
| 9 | `curate_sets` | `combinations/` |
| 10 | `select_representative` | `combinations-filtered/` |
| 11 | `filter_seq_clusters` | `dataset-pipeline/` |
| 12 | `auxillary_filters` | final cleanup on `dataset-pipeline/` |

---

## mmCIF download

```bash
python src/curation/utils/download_mmcif.py --data-dir /path/to/mmcif --pdb-list ids.txt
python src/curation/utils/download_mmcif.py --data-dir /path/to/mmcif   # full mirror (~600 GB)
```

---

## Evaluation prep (after release CSVs)

The released dataset in `data/dataset/` is not a 1:1 rerun of the pipeline (manual edits after step 9). For eval inputs:

```bash
bash scripts/setup_examples_layout.sh
python -m curation.make_pairs   # → valid_pairs.json, seq_cluster_to_answer_map.json
```

See `src/eval/README.md` for `promise_eval run`.

---

## Notes

- Post–`curate_sets` manual curation was applied for the paper dataset; `representative_sequences_total.json` may differ from pipeline output.
- Modules live under `src/curation/pipeline/`; shared helpers in `src/curation/utils/`.
