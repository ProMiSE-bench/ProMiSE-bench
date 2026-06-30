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

---

## `bias_score.json`

Canonical aggregate at `data/bias_score.json` (flat `data/bias_score.csv` alongside). Nested dict: `model → set → cluster → pair`:

```
{model}/{set}/{cluster_id}/{conf1-conf2} → {metric: value}
```

Models: `alphafold3`, `boltz1`, `boltz2`, `chai`, `bioemu`. Sets: `apo-monomers`, `ligand-induced`, `protein-induced` (`apo-monomers` = intrinsic dynamics).

| Field | Source | Set type |
|-------|--------|----------|
| `confbench_mean` | Pipeline 1 | apo-monomers |
| `confbench_apo_pred`, `confbench_holo_pred` | Pipeline 1 | induced |
| `distogram_confbench`, `distogram_dynamic_confbench` | Pipeline 2 | apo-monomers |
| `distogram_confbench_apo/holo`, `distogram_dynamic_confbench_apo/holo` | Pipeline 2 | induced |
| `msa_pref_sum`, `msa_pref_avg`, … | Pipeline 3 | all |
| `bias_ratio_diff`, `bias_entry*_hits` | Pipeline 4 | all |
| `rmsd_conf1_conf2` | Reference metrics | all |
| `after_training_cutoff` | Cutoff filter | all |

Full rebuild:

```bash
python -m eval.merge_all \
  --valid-pairs-json data/dataset/valid_pairs.json \
  --confbench-json data_eval/confbench_scores_boltz2.json \
  --confbench-distogram-json data_eval/confbench_scores_distogram_boltz2.json \
  --msa-pref-csv data_eval/per_pair_summary.csv \
  --training-bias-dir data_eval/train/training_bias
```

Defaults write `data/bias_score.json` and `data/bias_score.csv` (`eval.files.bias_score` in config).

Patch one model’s struct + distogram ConfBench after a re-run (other fields unchanged):

```bash
python -m eval.update_merged_confbench \
  --bias-score-json data/bias_score.json \
  --struct-confbench-json data_eval/confbench_scores_boltz2.json \
  --distogram-confbench-json data_eval/confbench_scores_distogram_boltz2.json \
  --model boltz2
```

Training bias only: `python -m eval.train.update_merged_bias_ratio_diff`.

