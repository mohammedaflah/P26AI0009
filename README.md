cat > README.md << 'EOF'
# ML Assignment 2: Bagging, Boosting and Bias-Variance

Name: Mohammed Aflah | Roll number: p26ai0009 | Seed: 260009

## Setup
Python 3.11.5

    pip install -r requirements.txt

## Reproduce
Run from the repository root. The first run downloads Olivetti Faces.

    python code/main.py 

All results are deterministic given the seed.

## Layout
- `configs/resolved.json`: all settings (seed, grids, bagging sizes, resamples)
- `data/`: saved split (`split.npz`), resamples (`resamples.npz`), metadata
- `code/`: scripts; `common.py` sharedd helpers
- `results/`: logs, JSON results, figures; `selected.json` carries the
  info of choices from Q1 and Q2 into Q2d and Q3
- `report/`: LaTeX version of report

## Notes
- first run Q1 and Q2, then Q3, because Q3 reads `results/selected.json`.
EOF