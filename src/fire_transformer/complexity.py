def transformer_macs(sequence_len, n_features, d_model, d_ff, n_layers):
    """Approximate MACs/inference for the paper architecture.

    Counts linear projections and dense attention matrix products; excludes LayerNorm,
    activation, softmax, residual additions and positional encoding.
    """
    s = sequence_len
    input_proj = (s - 1) * n_features * d_model
    per_layer = 4 * s * d_model * d_model + 2 * s * d_model * d_ff + 2 * s * s * d_model
    attention_pool = (s - 1) * d_model
    classifier = 2 * d_model * d_model + d_model
    return int(input_proj + n_layers * per_layer + attention_pool + classifier)


def parameter_bytes(model, bytes_per_parameter=4):
    return sum(p.numel() for p in model.parameters()) * bytes_per_parameter
