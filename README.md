# ProMiSE-bench

ProMiSE: **Pro**tein **M**ult**i**-**S**tate **E**valuation Benchmark in Biological Contexts

A curated benchmark dataset of protein conformational changes derived from experimentally determined structures in the Protein Data Bank (PDB).

## Overview

ProMiSE-bench provides high-quality protein conformational change pairs for:
- Assessing protein structure prediction models (e.g., AlphaFold3, Boltz-1,2, Chai-1, BioEmu)
- Evaluating multi-state conformation sampling capabilities of prediction models with novel metrics



### Key Features

- **Biology-Aware Pairs**: High-resolution pairs capturing binder-induced conformational changes
- **Stringent QC Pipeline**: Removal of crystal artifacts and redundant assemblies to ensure physiological relavance
- **Advanced Evaluation**: Multi-state success metrics and rigorous leakage analysis beyond traditional RMSD

### Quick Install

```bash
git clone https://github.com/ProMiSE-bench/ProMiSE-bench.git
cd ProMiSE-bench
bash install.sh
```

This script requires [uv](https://docs.astral.sh/uv/getting-started/installation/) (used to install the editable `promise-data` package into the conda env). It creates two conda environments:
- `promise`: Main curation pipeline (Python 3.9+)
- `prodigy-cryst`: Crystal contact classifier (Python 3.8, used internally)

To install only the Python package and dependencies into a local virtualenv (for example when reproducing eval scripts), run `uv sync` from the repository root, or `uv pip install -e .` into an existing environment. The full curation pipeline still expects the conda-provided tools (FAMSA, Foldseek, MMseqs2, and so on) from `environment.yaml`.


## Usage

### Dataset

The ProMiSE dataset is available in [data/dataset](data/dataset), where each CSV file is assigned to one of three categories: intrinsic dynamics, ligand-induced, or protein-induced.

### Running the Full Pipeline

```bash
conda activate promise
cd ProMiSE-bench

promise_data run \
    --spec data/clusters.json \
    --mmcif-store /path/to/pdb_mmcif/mmcif_files
```
`data/clusters.json` is provided in the repo. However, mmcif files should be manually downloaded with `src/curation/utils/download_mmcif.py`. Refer to [src/curation/README.md](src/curation/README.md) for details.


**Download Key Pipeline Outputs**: (https://drive.google.com/drive/folders/1BALc--RHPy8QVZaFNtI3LLXFfGWL_z4V?usp=drive_link)

Pre-computed pipeline outputs are available via the Google Drive link above. These files allow you to start from Step 4 and skip the computationally expensive Steps 1–3 (Create key files, TM-score computation, and conformation clustering). Since some outputs are required for evaluation, we strongly recommend downloading them.

After downloading, extract `data.tar.gz` in the `data/` directory:

```bash
tar -xzvf data.tar.gz
```

Then start the pipeline from Step 4 using the `--start-from` option (see [src/curation/README.md](src/curation/README.md) for more details):

```bash
promise_data run \
    --spec data/clusters.json \
    --mmcif-store /path/to/pdb_mmcif/mmcif_files \
    --start-from prepare_inputs
```


Pre-computed outputs downloaded (`data.tar.gz`) should be unzipped under `data/` directory.

## Evaluation

### Bias scores

Per-pair metrics for all models live in `data/bias_score.json` (see [src/eval/README.md](src/eval/README.md)). ConfBench, MSA bias, and training bias are combined there for analysis and plotting.

## Contributing

Contributions are welcome! Please open an issue or pull request.

## Contact

For questions or issues, please:
- Open a [GitHub issue](https://github.com/ProMiSE-bench/ProMiSE-bench/issues)

