# FlowSentinel — Drift-Aware Intrusion Detection

Research-oriented tabular IDS pipeline for the improved CICIDS-2017 and CSE-CICIDS-2018 datasets. It keeps data preparation, model evaluation, explainability hooks, and drift detection separate so experiments remain reproducible.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
flowsentinel --input examples/flows.csv --label label --output outputs
pytest
```

## Included

- Numeric schema validation and infinite-value handling
- Chronological holdout to reduce leakage
- Random Forest baseline with balanced class weights
- PSI drift report between reference and current windows
- Metrics and serialized model output

## Planned research extensions

Autoencoder latent features, XGBoost, TreeSHAP/LIME explanations, ADWIN online drift alerts, and controlled retraining belong behind the existing interfaces. They are documented as extensions rather than falsely presented as completed benchmark results.

The CICIDS files are not committed because of their size and license. Download them from their authorized source and place them under `data/raw/`.
