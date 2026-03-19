import torch
import torch.nn as nn
import torch.functional as F
from math import sqrt
import math

from wildcat.compresskv import compress_kv
from wildcat.weighted_attention import weighted_attention

class WildCat(nn.Module):
    """Implementation of WildCat module."""

    def __init__(
        self,
        scale: float | None = None,
        r: int = 128,
        mode: str = "eager",
        bins: int = 1,
        dim_bins: int = 1,
        **kwargs: dict,
    ):
        """Initialize the WildCat module.

        Args:
            scale (float): scale for dot-product attention. 
              If `None`, scale is chosen as 1/sqrt(keys.shape[-1]) in forward.
            r (int): number of key-value pairs to select, a nonnegative integer
            mode (str): if "eager", uses pytorch operations, only.
            bins (int): number of bins into which the sequence should be divided; 
              compression is performed independently on each bin.
            dim_bins (int): number of bins into which the key features should be divided.
            kwargs: placeholder for other arguments
        """
        super().__init__()

        self.scale = scale
        self.r = r
        self.mode = mode
        self.bins = bins
        self.dim_bins = dim_bins

    @torch.compile(mode="max-autotune")###mode="reduce-overhead",fullgraph=True)
    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        scale: float | None = None,
    ) -> torch.Tensor:
        
        """Forward pass of the WildCat module."""

        # Make input tensors have three dimensions (batch_size*num_heads, sequence_length, model_dimension)
        queries, keys, values, queries_shape = align_shapes(queries, keys, values)

        B, N, E = keys.shape
        B, M, E = queries.shape
        B, N, D = values.shape

        # Number of chunks of sequence
        C = self.bins
        # Folds along feature dimension
        F = self.dim_bins

        # Scale parameter of self-attention softmax
        scale = scale or self.scale or 1 / sqrt(E)

        # The attention output takes values in a convex polytope bounded by max_val and min_val
        max_val = values.amax(dim = -2, keepdim=True)
        min_val = values.amin(dim = -2, keepdim=True)

        # Recenter keys.
        kbar = keys.mean(dim = -2, keepdim=True)
        keys = keys - kbar

        # We chunk the input sequence and apply the compression algorithm for all chunks in parallel
        if C > 1:
            assert N % C == 0, "Sequence length of keys and values must be divisible by number of bins"
            # Divide key-value pairs into bins
            bin_r = self.r // C
            # Unfold bin dimension into batch dimension
            keys = keys.reshape(B*C, N//C, E)
            values = values.reshape(B*C, N//C, D)
        else:
            bin_r = self.r

        core_keys, core_values, core_one = compress_kv(keys, values, scale, bin_r)

        core_keys = core_keys.reshape(B, self.r, E)
        core_values = core_values.reshape(B, self.r, D)
        core_one = core_one.reshape(B, self.r)
        # # Reconstruction of attention output.
        # # TODO: Test other implementation, e.g. via flash-attention
        out = weighted_attention(queries = queries,
                                 core_keys = core_keys,
                                 core_values = core_values,
                                 core_one = core_one,
                                 scale = scale,
                                 min_val = min_val,
                                 max_val = max_val
                                 )
        
        out = out.view(*queries_shape[:-1], D)

        return out
    

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



