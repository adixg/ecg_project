# Repository Restructure Guide

This document is the migration contract for collaborators and coding agents.
Its purpose is to keep independently developed approaches easy to find, run,
compare, and maintain without creating filename conflicts.

## Canonical layout

Organize approach-specific code under one lowercase directory per model family:

```text
ecg_project/
├── README.md
├── REPO_RESTRUCTURE_GUIDE.md
├── artifacts/                 # Generated locally; ignored by Git
│   ├── classical_ml/
│   ├── cnn/
│   └── transformers/
├── src/
│   ├── classical_ml/          # Classical/non-neural ML approaches
│   │   ├── README.md
│   │   └── scratch.md
│   ├── cnn/                   # CNN and CNN-hybrid approaches
│   │   ├── README.md
│   │   └── scratch.md
│   └── transformers/          # Transformer approaches
│       ├── README.md
│       └── scratch.md
└── ...
```

Use these exact directory names:

- `src/classical_ml/` for feature-based and other classical machine-learning
  work. Move relevant files such as classical models and feature extraction
  code here.
- `src/cnn/` for CNNs and CNN hybrids, including CNN-BiLSTM models, training,
  and approach-specific evaluation code.
- `src/transformers/` for Transformer models. This directory already exists;
  keep the plural name rather than creating a second `src/transformer/` tree.

All package and Python filenames must use lowercase `snake_case`. Generic names
such as `model.py`, `train.py`, `metrics.py`, and `data.py` are acceptable inside
an approach directory because the directory provides the namespace. Do not put
new approach-specific modules directly under `src/`.

Shared, approach-independent utilities may remain directly under `src/` or be
moved into a clearly named shared package when there is a real second consumer.
Do not duplicate shared code merely to complete this restructure.

## What belongs in `artifacts/`

Put generated experiment outputs under the root-level `artifacts/` directory,
namespaced by approach and then by run. For example:

```text
artifacts/cnn/<run_name>/
├── config.json
├── epoch_metrics.json
├── validation_metrics.json
├── test_metrics.json
├── thresholds.json
├── best_model.pt
└── predictions.csv
```

All epoch-by-epoch metric JSON files belong here, not beside source files and
not in a tracked reports directory. Checkpoints, predictions, plots, caches,
and other generated run outputs also belong here unless there is a documented
reason to store them elsewhere. `artifacts/` is intentionally listed in
`.gitignore`; collaborators can regenerate these files or obtain them from the
person who ran the experiment.

Do not move source datasets or split definitions into `artifacts/`. Dataset
files remain under `data/`, and reproducible split files remain under `splits/`.

## Required README in each approach directory

Every approach directory must contain a concise but complete `README.md` that
lets another collaborator or agent run the approach without inspecting every
source file. Include all of the following:

1. A short approach and model-architecture overview.
2. A table listing every file in the directory and its purpose.
3. Environment setup and any dependencies beyond the root requirements.
4. Expected dataset location, input shape, labels, preprocessing, and split
   assumptions.
5. Exact commands, run from the repository root, for training, evaluation,
   prediction, and a short smoke test when available.
6. The main entry point and a table of every CLI flag it passes or consumes,
   including type, default value, and meaning.
7. The output directory structure and an explanation of the important output
   files.
8. Checkpoint-selection, early-stopping, threshold-selection, and metric
   conventions.
9. Any known limitations or reproducibility caveats.

Prefer module invocations such as:

```bash
python -m src.cnn.run --help
python -m src.classical_ml.run --help
python -m src.transformers.run --help
```

Commands documented in an approach README must work from the repository root.
If an approach cannot use this pattern, document its exact working directory
and explain why.

The root `README.md` remains the project entry point. It should summarize the
dataset and shared setup, list the available model families, and link to each
approach README rather than duplicating their full instructions.

## Required `scratch.md` in each approach directory

Each approach directory must also contain a tracked `scratch.md`. This is a
small, human-readable experiment summary, not an epoch log or a replacement for
raw artifacts. Update it with the best verified runs and enough provenance to
interpret them.

For each reported run, record at least:

- run name or local artifact path;
- model or variant name;
- dataset version and training fraction;
- seed and relevant reproducibility settings;
- device and approximate training time;
- checkpoint-selection metric and selected best epoch;
- validation and test macro AUROC, macro AUPRC, and macro F1;
- validation and test classwise AUROC, AUPRC, and F1;
- decision thresholds and how they were selected, when applicable;
- major preprocessing, class weighting, and hyperparameter details;
- any missing evaluation, caveat, or known comparability issue.

Use compact tables where possible. Report validation and test results
separately, identify whether F1 uses a fixed or tuned threshold, and never
choose or describe a checkpoint as "best" based on test-set performance.
Epoch-wise histories stay in `artifacts/`; `scratch.md` contains only selected
high-level results.

Suggested structure:

```markdown
# <Approach> experiment summary

## <Run or model name>

- Artifact: `artifacts/<approach>/<run_name>/`
- Seed: 42
- Selected epoch: 8 (highest validation macro AUROC)
- Thresholds: selected on validation data and frozen for test
- Training time: ...

| Split | Macro AUROC | Macro AUPRC | Macro F1 |
|---|---:|---:|---:|
| Validation | ... | ... | ... |
| Test | ... | ... | ... |

| Split | Class | AUROC | AUPRC | F1 | Threshold |
|---|---|---:|---:|---:|---:|
| Validation | NORM | ... | ... | ... | ... |
| Test | NORM | ... | ... | ... | ... |
```

## Migration procedure for agents

When asked to restructure one approach, use this sequence:

1. Read this guide, the root `README.md`, `.gitignore`, and the approach's
   existing source and documentation before editing.
2. Create the canonical approach directory and add `__init__.py` when it is a
   Python package.
3. Move only files belonging to that approach. Preserve unrelated work and do
   not reorganize another collaborator's approach as a side effect.
4. Update imports, module entry points, paths, tests, and documentation for the
   new location. Prefer paths resolved from the repository root rather than the
   caller's current directory.
5. Move generated epoch metric JSONs and other run outputs into the matching
   ignored `artifacts/<approach>/<run_name>/` hierarchy. Do not commit them.
6. Create or update the approach `README.md` and `scratch.md` using the
   requirements above.
7. Update the root `README.md` with a short link to the approach documentation.
8. Run the documented smoke test and any relevant tests. At minimum, confirm
   imports and `python -m src.<approach>.run --help` (or the documented
   equivalent) work from the repository root.
9. Inspect `git status` and verify that no dataset, checkpoint, cache, epoch
   history, credential, or unrelated collaborator file was accidentally added.

## Completion checklist

An approach migration is complete only when:

- its code is contained in the correct canonical directory;
- package imports and documented commands work from the repository root;
- its directory has both `README.md` and `scratch.md`;
- the README describes every file, entry point, CLI flag, input, and output;
- the scratch file records selected validation/test aggregate and classwise
  metrics, seed, selected epoch, thresholds, and training metadata;
- epoch-wise JSONs and other generated outputs are under ignored `artifacts/`;
- root documentation links to the approach documentation; and
- relevant tests or smoke checks pass.

