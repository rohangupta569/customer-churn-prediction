# Contributing

Small fixes and experiments are welcome. Please open an issue before making a large change so the approach can be discussed first.

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,app]'
pytest
```

Keep data preparation inside the model pipeline so cross-validation remains leakage-safe. New behaviour should include a test where practical. Do not commit private customer data or trained artifacts based on private data.
