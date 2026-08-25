# Exact patch for tests/smoke_test.py

After the existing FT-64 parameter assertion, add:

```python
ft32 = build_model(
    cfg["models"]["ft32"],
    n_features=20,
    gas_feature_index=3,
)

assert sum(
    p.numel()
    for p in ft32.parameters()
) == 20002
```

This verifies that the restored FT-32 configuration matches the paper.
