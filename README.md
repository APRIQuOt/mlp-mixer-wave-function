# MLP Mixer paper code

This repository contains the TFIM and Rydberg ground-state implementations and
the experiments, analysis, figures, and tables used by in paper paper, [Multi-Layer Perceptron Mixer Wave Functions](https://www.overleaf.com/project/6aae96ae4e15eb466a285122/detached). The
models are the static MLP Mixer, RNN, and Transformer, including their patched
and hierarchical variants.

The campaign is frozen from all **540 saved paper runs** in
`data/paper_ready_v2`: 36 architecture/system designs, three Hamiltonian phase
points per design, and five seeds (123, 456, 789, 1234, 5678). Each run definition
contains its resolved architecture, training settings, actual learning-rate
schedule, float32 parameter dtype, reference energy density, and source hashes.
Training does not rerun a parameter-budget search.

## Contents

- `src/tnqs/`: the ground-state numerical implementation.
- `src/paper_workflow/`: frozen campaign execution, analysis, and figure code.
- `configs/campaign.json`, `configs/runs/`: the complete experiment definitions.
- `references/energies.json`: benchmark energy densities and available uncertainties.
- `data/paper_ready_v2/`: our results, including checkpoints and parquet histories.
- `assets/`: the paper's fixed architecture image and portable plotting style.
- `scripts/`: commands for data generation, analysis, figures, and exact references.
- `tests/`: numerical, configuration, and published-summary checks.

Generated checkpoints, histories, processed data, plots, and caches go under
`build/`, which is ignored by Git. The copied results archive under `data/` is
also ignored by Git. Notebooks, logs, bundled environments, and archived source
snapshots are not included. Legacy archive helpers are retained with the results;
the reproduction workflow uses the clean scripts and the two parquet files.

## Setup

Use Python 3.13 or later. Dependency versions are pinned to the local environment
used to verify extraction. From this repository:

```sh
uv sync --extra test
```

Alternatively, create a virtual environment and run `python -m pip install -e '.[test]'`.
The source-checkout scripts also work directly with an environment containing
the dependencies in `pyproject.toml`.

## Generate training data

List all frozen runs:

```sh
uv run python scripts/generate_data.py --list
```

Run a short CPU pilot before committing to a full campaign:

```sh
uv run python scripts/generate_data.py --task-index 0 --device cpu --steps 2 --samples 64
```

Pilots save into a separate directory and are rejected by production analysis.

Run one production experiment, or the complete campaign:

```sh
uv run python scripts/generate_data.py --task-index 0 --device cuda
uv run python scripts/generate_data.py --device auto
```

`--run-id` selects named experiments; `--task-index` selects a zero-based index
from 0 to 539 for a cluster job array. `--output` selects persistent storage.
Completed matching runs are skipped. Interrupted runs restart from the beginning;
mid-training resume is not implemented.

TFIM runs have 15,000 updates; Rydberg runs have 20,000 updates for 4×4, 8×8,
and 16×16, and 25,000 for 24×24. Each run uses its recorded constant sample
count, either 256 or 1,024. The two recovered cosine schedules are preserved
per run: 214 decay to zero, and 326 decay to 1% of the initial learning rate.
The 16×16 hierarchical Transformer/Mixer group contains both schedules.

## Analyze and plot

After the production campaign completes:

```sh
uv run python scripts/analyze.py
uv run python scripts/generate_plots.py
```

Analysis writes processed parquet files, per-design accuracy summaries in JSON
and CSV, a dispatcher operation-coverage report, and a generated FLOP table for
all 36 designs. Use `--allow-partial` to analyze completed production histories
while the campaign is still running.

To reproduce plots from the copied results without retraining:

```sh
uv run python scripts/analyze.py --input data/paper_ready_v2
uv run python scripts/generate_plots.py
```

That input directory only needs `tfim_data.parquet` and `rydberg_data.parquet`.
Models and costs are rebuilt from this repository's frozen JSON definitions.

The paper figures use PGF and PDF with LaTeX text. Install a TeX distribution
with `pdflatex`, `amsmath`, `amssymb`, and `siunitx`, and ensure `pdflatex` is on
`PATH`. Outputs are written to `build/paper/images/`, with PNG previews in
`build/paper/previews/` and metadata/tables in `build/paper/tables/`.
The architecture overview alone can be generated without TeX:

```sh
uv run python scripts/generate_plots.py --diagram-only
```

The accuracy coordinate is the absolute difference between the reference
energy density and the mean of the last 1,000 energy-density reports through
update 15,000. Each main comparison pools 15 run errors across three phase
points and five seeds, showing their mean and 25th–75th percentile interval.
Training panels use the lowest-numbered seed, a trailing 1,000-update mean,
and an unsmoothed energy inset.

FLOPs are the PyTorch dispatcher's **supported** forward matrix-product FLOPs
for a batch of eight divided by eight.

## Reference calculations

Generate float64 sparse exact-diagonalization checks for all six 4×4 benchmark
Hamiltonians, or select one:

```sh
uv run python scripts/generate_references.py
uv run python scripts/generate_references.py --hamiltonian tfim --phase 1
```

The larger TFIM ITensor and Rydberg QMC reference values are supplied inputs.
Their original external solver code is unavailable in the source project;
this repository preserves the values rather than claiming to regenerate those
calculations. The 24×24 Rydberg standard deviations are recorded separately.

`assets/architecture.pdf` is the fixed appendix drawing used by the manuscript.
Its original generator was unavailable. The main architecture overview has a
complete vector generator in `src/paper_workflow/diagram.py`.
