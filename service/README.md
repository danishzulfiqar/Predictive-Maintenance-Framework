# Service

FastAPI microservice of the [Predictive Maintenance Framework](../README.md). It keeps a registry of per-machine models, analyses recent telemetry for a machine, attributes every anomaly to its dominant sensor with SHAP, and forecasts the next maintenance date.

```bash
docker build -t pdm-api .
docker run -p 8000:8000 -e MONGO_URL="mongodb://USER:PASSWORD@HOST:PORT/admin" pdm-api
# API docs: http://localhost:8000/docs
```

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/{mac}/{minutes}/predict_machine_for_minutes` | Analyse the last *N* minutes of a machine's telemetry |
| `GET` | `/{mac}/{minutes}/get_interval_data` | Return the merged telemetry window |
| `POST` | `/predict_chenab` | Analyse telemetry posted in the request body |
| `POST` | `/update_model` | Upload `<MAC>.zip` with `model.h5` + `scaler.pkl` |
| `GET` | `/list_models` | List machines with a deployed model |

Models are trained and published by the notebooks in [`../training`](../training).
