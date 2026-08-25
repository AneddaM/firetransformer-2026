# Exact patch for scripts/run_lono.py

Change:

```python
choices=["ft64", "ft128", "bilstm_bce", "bilstm_fl"],
```

to:

```python
choices=["ft32", "ft64", "ft128", "bilstm_bce", "bilstm_fl"],
```

No other change is required in `run_lono.py`: the script reads the selected model
configuration from `cfg["models"][model_name]`, so FT-32 will automatically use the
same leakage-controlled LONO pipeline once it exists in `configs/default.yaml`.
