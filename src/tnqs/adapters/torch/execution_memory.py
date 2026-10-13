"""Conservative per-call workspace bounds, not process peak-memory guarantees."""


def local_evaluation_row_limit(local, max_rows=65536):
    if max_rows <= 0:
        raise ValueError("max_rows must be positive.")
    tokens, channels = local.num_tokens, local.channel_dim
    hidden, vocabulary = local.token_hidden_dim, local.vocab_size
    item_bytes = next(local.parameters()).element_size()
    # Include categorical logits and log-softmax, not only the token MLP. In
    # particular, a local 4x4 categorical patch has 65,536 outcomes per token.
    backbone = channels * max(tokens, hidden)
    elements = 2 * backbone + 6 * tokens * channels + 3 * tokens * vocabulary
    return max(1, min(max_rows, (1 << 28) // (item_bytes * backbone),
                      (1 << 30) // (item_bytes * elements)))
