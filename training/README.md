# Training pipeline

Notebooks of the [Predictive Maintenance Framework](../README.md) that turn one machine's raw telemetry into a deployable anomaly-detection model, without any failure labels.

| # | Notebook | Step |
|---|---|---|
| 1 | `1_getMacData.ipynb` | Extract current, vibration and temperature data from MongoDB by node MAC address |
| 2 | `2_Pre_Data_Processing.ipynb` | Keep ON-state readings, clean, and join the sensor streams on time |
| 3 | `3_isolated_forest.ipynb` | Isolation Forest (10 % contamination) → pseudo-labels |
| 4 | `4_model.ipynb` | Train the MLP (256 ReLU → sigmoid) with early stopping |
| 5 | `5_evaluate_model.ipynb` | Evaluate, save the scaler, compute maintenance indicators |
| 6 | `6_deploy_model.ipynb` | Zip and upload the model to the service's `/update_model` endpoint |

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Set `MONGO_URL` and `mac_address` in notebook 1 and the service URL in notebook 6. To redraw the README figures from the executed notebooks, run `python scripts/plot_results.py`; it writes to `../assets/`.
