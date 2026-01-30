def align_shapes(queries, keys, values):
    """
    Assertions and possible reshapes
    The compression algorithms in compressed attention assume input tensors with shape (b, n, d).

    This function checks the shapes of the input tensors and reshapes them if necessary.

    Args:
        queries: (B, (H), T, E) The tensor containing the queries
        keys: (B, (H), S, E) The tensor containing the keys
        values: (B, (H), S, D) The tensor containing the values

    Returns:
        queries: (B', T, E) The reshaped queries tensor
        keys: (B', S, E) The reshaped keys tensor
        values: (B', S, D) The reshaped values tensor
        queries_shape: tuple containing the original shape of queries
    """

    queries_shape = queries.shape

    if len(keys.shape) > 3:
        """
        Reshape inputs to (n_batch*n_heads, n_seq, dim)
        """
        queries = queries.reshape(-1, queries.shape[-2], queries.shape[-1])
        keys = keys.reshape(-1, keys.shape[-2], keys.shape[-1])
        values = values.reshape(-1, values.shape[-2], values.shape[-1])

    return queries, keys, values, queries_shape