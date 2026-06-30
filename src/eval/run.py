"""CLI entry point for promise-bench evaluation pipelines.

Install the package (``pip install -e .``) then run::

    promise_eval run -m boltz2

Or invoke via ``python -m eval``::

    python -m eval run -m boltz2 --align-parts 4 --loss-parts 20
    python -m eval run --with-msa --with-train --weights-json /path/to/weights.json

List steps::

    promise_eval steps
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import click

_REPO_ROOT = Path(__file__).resolve().parents[2]

EVAL_STEPS: tuple[str, ...] = (
    # Pipeline 1 — struct ConfBench
    "make_pairs",
    "distogram_prep",
    "alignment",
    "ref_metrics",
    "struct_confbench",
    # Pipeline 2 — distogram ConfBench
    "ref_distogram_diff",
    "distogram_loss",
    "distogram_confbench",
    # Pipeline 3 — MSA bias
    "msa_renumbered_pdbs",
    "msa_esm_contacts",
    "msa_bias",
    "msa_summary",
    # Pipeline 4 — training bias
    "train_foldseek_hits",
    "train_mmseqs_hits",
    "train_intersection",
    "train_bias",
)

_P1_STEPS = frozenset(EVAL_STEPS[:5])
_P2_STEPS = frozenset(EVAL_STEPS[5:8])
_P3_STEPS = frozenset(EVAL_STEPS[8:12])
_P4_STEPS = frozenset(EVAL_STEPS[12:])


def _step_index(name: str) -> int:
    try:
        return EVAL_STEPS.index(name)
    except ValueError as e:
        raise click.ClickException(f"Unknown step {name!r}") from e


def _in_range(step: str, start: str | None, stop: str | None) -> bool:
    i = _step_index(step)
    if start is not None and i < _step_index(start):
        return False
    if stop is not None and i > _step_index(stop):
        return False
    return True


def _needs_pipeline1(start: str | None, stop: str | None) -> bool:
    return any(_in_range(s, start, stop) for s in _P1_STEPS)


def _needs_pipeline2(start: str | None, stop: str | None) -> bool:
    return any(_in_range(s, start, stop) for s in _P2_STEPS)


def _needs_pipeline3(start: str | None, stop: str | None) -> bool:
    return any(_in_range(s, start, stop) for s in _P3_STEPS)


def _needs_pipeline4(start: str | None, stop: str | None) -> bool:
    return any(_in_range(s, start, stop) for s in _P4_STEPS)


def _p1_env(
    base: dict[str, str],
    start: str | None,
    stop: str | None,
    *,
    run_distogram: bool,
) -> dict[str, str]:
    env = base.copy()
    env["SKIP_MAKE_PAIRS"] = "0" if _in_range("make_pairs", start, stop) else "1"
    if _in_range("distogram_prep", start, stop):
        env["RUN_DISTOGRAM"] = "1"
        env["SKIP_DISTOGRAM"] = "0"
    else:
        env["RUN_DISTOGRAM"] = "1" if run_distogram else "0"
        env["SKIP_DISTOGRAM"] = "1"
    env["SKIP_ALIGN"] = "0" if _in_range("alignment", start, stop) else "1"
    env["SKIP_REF_METRICS"] = "0" if _in_range("ref_metrics", start, stop) else "1"
    env["SKIP_SCORE"] = "0" if _in_range("struct_confbench", start, stop) else "1"
    return env


def _p2_env(base: dict[str, str], start: str | None, stop: str | None) -> dict[str, str]:
    env = base.copy()
    if _in_range("distogram_prep", start, stop):
        env["RUN_PREP"] = "0"
    else:
        env["RUN_PREP"] = "1" if _in_range("ref_distogram_diff", start, stop) else "0"
    env["SKIP_REF_DIFF"] = "0" if _in_range("ref_distogram_diff", start, stop) else "1"
    env["SKIP_LOSS"] = "0" if _in_range("distogram_loss", start, stop) else "1"
    env["SKIP_SCORE"] = "0" if _in_range("distogram_confbench", start, stop) else "1"
    return env


def _p3_env(base: dict[str, str], start: str | None, stop: str | None) -> dict[str, str]:
    env = base.copy()
    env["SKIP_RENUMBER"] = "0" if _in_range("msa_renumbered_pdbs", start, stop) else "1"
    env["SKIP_ESM"] = "0" if _in_range("msa_esm_contacts", start, stop) else "1"
    env["SKIP_BIAS"] = "0" if _in_range("msa_bias", start, stop) else "1"
    env["SKIP_SUMMARY"] = "0" if _in_range("msa_summary", start, stop) else "1"
    return env


def _p4_env(base: dict[str, str], start: str | None, stop: str | None) -> dict[str, str]:
    env = base.copy()
    env["SKIP_FOLDSEEK"] = "0" if _in_range("train_foldseek_hits", start, stop) else "1"
    env["SKIP_MMSEQS"] = "0" if _in_range("train_mmseqs_hits", start, stop) else "1"
    env["SKIP_INTERSECTION"] = "0" if _in_range("train_intersection", start, stop) else "1"
    env["SKIP_TRAIN_BIAS"] = "0" if _in_range("train_bias", start, stop) else "1"
    return env


def _run_full(
    *,
    model: str | None,
    examples_dir: str | None,
    align_parts: int,
    loss_parts: int,
    ref_diff_parts: int,
    run_distogram: bool,
    skip_pipeline1: bool,
    skip_pipeline2: bool,
    with_msa: bool,
    with_train: bool,
    skip_esm: bool,
    weights_json: str | None,
    train_models: str | None,
    start_from: str | None,
    stop_after: str | None,
    collect_force: bool,
    loss_no_skip: bool,
) -> None:
    base = os.environ.copy()
    if model:
        base["MODELS"] = model
    base["ALIGN_PARTS"] = str(align_parts)
    base["LOSS_PARTS"] = str(loss_parts)
    base["REF_DIFF_PARTS"] = str(ref_diff_parts)
    if examples_dir:
        base["EXAMPLES_DIR"] = examples_dir
    if collect_force:
        base["COLLECT_FORCE"] = "1"
    if loss_no_skip:
        base["LOSS_NO_SKIP"] = "1"
    if weights_json:
        base["WEIGHTS_JSON"] = weights_json
    if train_models:
        base["TRAIN_MODELS"] = train_models
    if skip_esm:
        base["SKIP_ESM"] = "1"

    if start_from or stop_after:
        start = start_from or EVAL_STEPS[0]
        stop = stop_after or EVAL_STEPS[-1]
        if _needs_pipeline1(start, stop):
            if not model:
                raise click.ClickException("--model is required for struct/distogram ConfBench steps.")
            subprocess.run(
                ["bash", str(_REPO_ROOT / "scripts/eval_pipeline1_run.sh")],
                env=_p1_env(base, start, stop, run_distogram=run_distogram),
                cwd=_REPO_ROOT,
                check=True,
            )
        if _needs_pipeline2(start, stop):
            if not model:
                raise click.ClickException("--model is required for struct/distogram ConfBench steps.")
            subprocess.run(
                ["bash", str(_REPO_ROOT / "scripts/eval_pipeline2_run.sh")],
                env=_p2_env(base, start, stop),
                cwd=_REPO_ROOT,
                check=True,
            )
        if _needs_pipeline3(start, stop):
            subprocess.run(
                ["bash", str(_REPO_ROOT / "scripts/eval_pipeline3_run.sh")],
                env=_p3_env(base, start, stop),
                cwd=_REPO_ROOT,
                check=True,
            )
        if _needs_pipeline4(start, stop):
            subprocess.run(
                ["bash", str(_REPO_ROOT / "scripts/eval_pipeline4_run.sh")],
                env=_p4_env(base, start, stop),
                cwd=_REPO_ROOT,
                check=True,
            )
        return

    skip_p1 = skip_pipeline1
    skip_p2 = skip_pipeline2
    skip_p3 = not with_msa
    skip_p4 = not with_train

    if not model and not skip_p1 and not skip_p2:
        raise click.ClickException(
            "--model is required unless both ConfBench pipelines are skipped."
        )

    if skip_p1 and skip_p2 and skip_p3 and skip_p4:
        raise click.ClickException("All pipelines skipped; nothing to run.")

    env = base.copy()
    env["RUN_DISTOGRAM"] = "1" if run_distogram else "0"
    env["SKIP_PIPELINE1"] = "1" if skip_p1 else "0"
    env["SKIP_PIPELINE2"] = "1" if skip_p2 else "0"
    env["SKIP_PIPELINE3"] = "1" if skip_p3 else "0"
    env["SKIP_PIPELINE4"] = "1" if skip_p4 else "0"
    subprocess.run(
        ["bash", str(_REPO_ROOT / "scripts/eval_pipeline_run.sh")],
        env=env,
        cwd=_REPO_ROOT,
        check=True,
    )


@click.group()
@click.version_option(package_name="promise-data")
def promise_eval():
    """promise-bench evaluation pipelines (ConfBench, MSA, training bias)."""


@promise_eval.command()
@click.option(
    "--model",
    "-m",
    "model",
    default=None,
    help="Prediction method for ConfBench (e.g. boltz2). Optional if only running MSA/train.",
)
@click.option(
    "--examples-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Prediction samples root (default: pipeline examples path in shell scripts).",
)
@click.option("--align-parts", type=int, default=4, show_default=True)
@click.option("--loss-parts", type=int, default=20, show_default=True)
@click.option("--ref-diff-parts", type=int, default=10, show_default=True)
@click.option(
    "--no-distogram-prep",
    is_flag=True,
    default=False,
    help="Skip extract_cb / collect / distogram_tasks in pipeline 1.",
)
@click.option("--skip-pipeline1", is_flag=True, default=False)
@click.option("--skip-pipeline2", is_flag=True, default=False)
@click.option(
    "--with-msa",
    is_flag=True,
    default=False,
    help="Also run pipeline 3 (MSA bias).",
)
@click.option(
    "--with-train",
    is_flag=True,
    default=False,
    help="Also run pipeline 4 (training memorization bias).",
)
@click.option(
    "--skip-esm",
    is_flag=True,
    default=False,
    help="Skip GPU ESM contact prediction in pipeline 3.",
)
@click.option(
    "--weights-json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Cluster weights JSON for pipeline 4 train_bias step.",
)
@click.option(
    "--train-models",
    default=None,
    help="Space-separated train model keys (default: af3 boltz_2 chai_1 bioemu).",
)
@click.option(
    "--start-from",
    type=click.Choice(EVAL_STEPS, case_sensitive=False),
    default=None,
    help="Resume from this step (inclusive).",
)
@click.option(
    "--stop-after",
    type=click.Choice(EVAL_STEPS, case_sensitive=False),
    default=None,
    help="Stop after this step (inclusive).",
)
@click.option(
    "--collect-force",
    is_flag=True,
    default=False,
    help="Pass --force to collect_distograms (refresh chain_seq_mapping.json).",
)
@click.option(
    "--loss-no-skip",
    is_flag=True,
    default=False,
    help="Force distogram loss recomputation (--no-skip).",
)
def run(
    model: str | None,
    examples_dir: Path | None,
    align_parts: int,
    loss_parts: int,
    ref_diff_parts: int,
    no_distogram_prep: bool,
    skip_pipeline1: bool,
    skip_pipeline2: bool,
    with_msa: bool,
    with_train: bool,
    skip_esm: bool,
    weights_json: Path | None,
    train_models: str | None,
    start_from: str | None,
    stop_after: str | None,
    collect_force: bool,
    loss_no_skip: bool,
):
    """Run evaluation pipelines (ConfBench, optionally MSA + train)."""
    _run_full(
        model=model,
        examples_dir=str(examples_dir) if examples_dir else None,
        align_parts=align_parts,
        loss_parts=loss_parts,
        ref_diff_parts=ref_diff_parts,
        run_distogram=not no_distogram_prep,
        skip_pipeline1=skip_pipeline1,
        skip_pipeline2=skip_pipeline2,
        with_msa=with_msa,
        with_train=with_train,
        skip_esm=skip_esm,
        weights_json=str(weights_json) if weights_json else None,
        train_models=train_models,
        start_from=start_from,
        stop_after=stop_after,
        collect_force=collect_force,
        loss_no_skip=loss_no_skip,
    )


@promise_eval.command("steps")
def list_steps():
    """List evaluation pipeline steps."""
    sections = (
        ("Pipeline 1 (struct ConfBench)", EVAL_STEPS[:5]),
        ("Pipeline 2 (distogram ConfBench)", EVAL_STEPS[5:8]),
        ("Pipeline 3 (MSA bias)", EVAL_STEPS[8:12]),
        ("Pipeline 4 (training bias)", EVAL_STEPS[12:]),
    )
    idx = 1
    for title, steps in sections:
        click.echo(title + ":")
        for step in steps:
            click.echo(f"  {idx:2d}. {step}")
            idx += 1
