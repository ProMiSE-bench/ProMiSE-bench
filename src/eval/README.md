# Evaluation (`src/eval/`)

ProMiSE-bench evaluation: structure ConfBench, distogram ConfBench, MSA bias, and training memorization bias. Paths default from `config/config.yaml` (`eval` section).

---

## Quick start

```bash
conda activate promise

# Rebuild examples/ symlinks (predictions, MSAs, reference CIFs)
bash scripts/setup_examples_layout.sh

# Default: Pipeline 1 (struct) + Pipeline 2 (distogram)
promise_eval run -m boltz2

# All four pipelines
promise_eval run -m boltz2 --with-msa --with-train --weights-json /path/to/weights.json

# List steps
promise_eval steps
```

Equivalent shell wrappers: `scripts/eval_pipeline_run.sh` (P1–P4; P3/P4 skipped by default).

---

## Pipelines and steps

| Pipeline | Steps | Main outputs |
|----------|-------|--------------|
| **1 — Struct ConfBench** | `make_pairs` → `distogram_prep` → `alignment` → `ref_metrics` → `struct_confbench` | `data_eval/confbench_scores_{model}.json` |
| **2 — Distogram ConfBench** | `ref_distogram_diff` → `distogram_loss` → `distogram_confbench` | `data_eval/confbench_scores_distogram_{model}.json` |
| **3 — MSA bias** | `msa_renumbered_pdbs` → `msa_esm_contacts` → `msa_bias` → `msa_summary` | `data_eval/per_pair_summary.csv` |
| **4 — Training bias** | `train_foldseek_hits` → `train_mmseqs_hits` → `train_intersection` → `train_bias` | `training_bias_per_pair_{model}.json` under `data_eval/train/training_bias/` |

`make_pairs` writes `data/dataset/valid_pairs.json` and `seq_cluster_to_answer_map.json` (MSA paths, reference CIFs, chains, distogram patterns).

Slice a run with `--start-from` / `--stop-after` (step names from `promise_eval steps`).

---

## Prerequisites

| Requirement | Used by |
|-------------|---------|
| `examples/` layout (`setup_examples_layout.sh`) | `make_pairs`, alignment, distogram |
| `fair-esm` in `promise` env | MSA ESM contacts (`--skip-esm` to skip) |
| `--weights-json` | Pipeline 4 `train_bias` |
| External FoldSeek/MMseqs hit TSVs (`foldseek_hits_dir` in config) | Pipeline 4 hit collection |

Smoke test (demo clusters `7OYW_1`, `2H3H_1`): `bash scripts/eval_all_pipelines_smoke.sh`.

---

## Manual step order

Use individual modules when you need finer control (Slurm sharding, partial reruns). Dependencies:

1. `python -m curation.make_pairs`
2. `python -m eval.distogram.extract_reference_cb --answer-map data/dataset/seq_cluster_to_answer_map.json`
3. `python -m eval.distogram.collect_distograms --json <map_with_cb_paths.json>`
4. `python -m eval.align.generate_alignment_tasks` → `split_alignment_jobs` / `struct_align_batch`
5. `python -m eval.struct.calc_reference_structural_metrics` → `calc_confbench_score_valid_pairs`
6. `python -m eval.distogram.calc_reference_distogram_diff` → `calc_distogram_loss` → `calc_distogram_confbench`
7. MSA: `eval.msa.cif_to_renumbered_pdb` → `esm_run` → `msa_bias` → `summarize_msa_bias`
8. Train: `eval.train.collect_memorization_hits_foldseek` → `collect_memorization_hits_mmseqs` → `create_intersection_hits` → `calculate_training_bias_per_pair_weighted`

Alignment and distogram prep can run in parallel after `make_pairs`. Distogram loss/ref-diff/confbench are sequential.

Optional: `eval.merge_all` merges struct + distogram ConfBench + MSA + training bias into `preference_scores.json`.

---

## Module index

| Area | Key modules |
|------|-------------|
| Orchestration | `eval/run.py` (`promise_eval`) |
| Alignment | `align/generate_alignment_tasks`, `split_alignment_jobs`, `struct_align_batch` |
| Struct | `struct/calc_reference_structural_metrics`, `calc_confbench_score_valid_pairs` |
| Distogram | `distogram/extract_reference_cb`, `collect_distograms`, `calc_*`, `check_*`, `generate_*_jobs` |
| MSA | `msa/cif_to_renumbered_pdb`, `esm_run`, `msa_bias`, `summarize_msa_bias` |
| Train | `train/collect_memorization_hits_foldseek`, `collect_memorization_hits_mmseqs`, `create_intersection_hits`, `calculate_training_bias_per_pair_weighted` |
| Merge | `merge_all.py`, `update_preference_scores.py` |
