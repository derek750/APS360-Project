# APS360 Progress Report — Logo Decade Classification

Reproduce the Logopedia scrape, Palermo-style color baseline, and shallow CNN training from [notebooks/progress_report.ipynb](../notebooks/progress_report.ipynb).

## Outputs

| Path | Description |
|------|-------------|
| `data/splits.json` | Company-level train/val/test assignments |
| `data/manifest.json` | Image metadata (company, decade, path, split) |
| `data/stats.json` | Dataset counts for the LaTeX report |
| `data/results.json` | Baseline and CNN metrics |
| `figures/*.pdf` | Report figures |

## Local run

```bash
python3 -m pip install requests Pillow scikit-learn torch torchvision matplotlib tqdm
python3 notebooks/run_pipeline.py
```

Images are cached under `data/images/` (gitignored). The notebook uses the same logic and is intended for Google Colab submission.
