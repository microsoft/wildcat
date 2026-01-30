import torch
import torch.nn as nn
import torch.functional as F
from math import sqrt
import math

from cmpd_attn.transformer_utils import align_shapes
from cmpd_attn.math_utils import lambert_w_circ_exp

from cmpd_attn.tr_update_kernel import update_kernel_triton
    

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
        
        """Forward pass of the wildCat module."""

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

        # Shift values to minimize the max norm of the recentered values
        # Note that vbar can be integrated exactly into the attention output later
        vbar = (max_val+min_val)/2 #values.mean(dim = -2, keepdim=True)
        values = values - vbar

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

        # We compress chunks of the dimension independently and recombine later.
        # The effective coreset size by the two chunk methods is C*(r//C)^F
        if F > 1:
            assert E % F == 0, "Model dimension must be divisible by dim_bins"
            assert F <= 8, "Coreset sizes grow exponentially with dim_bins, please set dim_bins <= 8"
            # Divide key-value pairs into bins along feature dimension
            keys = keys.reshape(B*C, N//C, E//F, F).permute(0, 3, 1, 2).reshape(B*C*F, N//C, E//F)
            #queries = queries.reshape(B, M, E//F, F).permute(0, 3, 1, 2).reshape(B*F, M, E//F)
            

        # Preprocessing of the keys and queries
        # Determine rescaling of keys to be used during compression

        # Shape (B*C*F, N//C)
        sqd_knorm = keys.square().sum(dim=-1)

        # Shape (B, F)
        # TODO: Consider reshaped queries for F > 1. Changes which dimension is reduced over.
        if F > 1:
            queries = queries.reshape(B, M, E//F, F)
            q_scale = queries.square().sum(dim = -2).sqrt().amax(dim = -2)
            q_scale = q_scale.reshape(B, 1, F, 1).expand(B, C, F, 1).reshape(B*C*F, 1)
            queries = queries.reshape(B, M, E)
        else:
            q_scale = queries.square().sum(dim = -1).sqrt().amax(dim = -1)
            q_scale = q_scale.reshape(B, 1).expand(B, C).reshape(B*C, 1)
        
        # Shape (B*C*F, 1)
        k_scale = sqd_knorm.sqrt().amax(dim = -1, keepdim=True)

        # Shape (B*C*F, 1)
        tau = find_kernel_temperature(
            scale = scale,
            q_scale=q_scale,
            k_scale=k_scale,
            n = N,
            phi = None
        )

        key_multiplier = sqrt(scale) / tau
        keys = keys * key_multiplier.unsqueeze(-1)
        sqd_knorm = sqd_knorm * (key_multiplier**2)

        # Compression of keys and values
        # Outputs kernel_inv and kernel_core computed from Gaussian kernel
        coreset, kernel_inv, kernel_core = rp_nystrom(
            keys=keys,
            sqd_knorm=sqd_knorm,
            r=bin_r,
            mode=self.mode,
        )

        # Select compressed keys:
        # Shape (B*C*F, r//C, E//F)
        core_keys = keys.gather(-2, coreset.unsqueeze(-1).expand(*coreset.shape, E//F))
        # Undo rescaling of keys
        core_keys /= key_multiplier.unsqueeze(-1)
        core_sqd_knorms = sqd_knorm.gather(-1, coreset)

        # Compute Nystrom weights for Gaussian kernel
        W = torch.einsum("...rs, ...sl -> ...rl", kernel_inv, kernel_core)

        if F > 1:
            # Rescaling does not need to be undone for norms, as they become part of weights determined by Nystrom
            codes = torch.arange(bin_r**F, device=keys.device)
            powers = (bin_r ** torch.arange(F, device=keys.device))
            grid_ids = (codes[:, None] // powers[None, :]) % bin_r
            grid_ids = grid_ids.T # F, r_bin^F

            # Take all combinations of coresets along feature dimensions
            core_keys = core_keys.reshape(B*C, F, bin_r, E//F)
            core_keys = core_keys.gather(-2, grid_ids[None, ..., None].expand(B*C, F, bin_r**F, E//F))
            core_keys = core_keys.permute(0, 2, 1, 3).reshape(B, C*bin_r**F, E)

            core_sqd_knorms = core_sqd_knorms.reshape(B*C, F, bin_r)
            core_sqd_knorms = core_sqd_knorms.gather(-1, grid_ids[None, ...].expand(B*C, F, bin_r**F)).sum(dim=-2)

        else:
            # Fold bin dimension back into sequence dimension
            core_keys = core_keys.reshape(B, self.r, E)

        # Compute compressed values:
        if F > 1:
            W = W.reshape(B*C, F, bin_r, N//C)
            W = W.gather(-2, grid_ids[None, ..., None].expand(B*C, F, bin_r**F, N//C))
            W = W.prod(-3)

            # Multiply by scaling terms
            # sqd_knorm has shape (B*C*F, N//C)
            # core_sqd_knorms has shape (B, C, (r//C)^F)
            sqd_knorm = sqd_knorm.reshape(B*C, F, N//C).sum(dim=-2)

            # (B, C, (r//C)^F, N//C)
            scaling = -core_sqd_knorms.unsqueeze(-1) + sqd_knorm.unsqueeze(-2)
            scaling = scaling - scaling.amax((-1,-2), keepdim=True)
            W = W * torch.exp(scaling / 2.)

            # (B, C*(r//C)^F, D)
            compressed_values = torch.einsum("...rn, ...nd -> ...rd", W, values).reshape(B, C*bin_r**F, D)

            # (B, C*(r//C)^F)
            compressed_one = W.sum(dim=-1).reshape(B, C*bin_r**F)

        else:
            # Shapes: 
            #   kernel_core (B*C, r//C, N//C)
            #   kernel_inv (B*C, r//C, r//C)
            #   values (B*C, N//C, D)
            # Reduce over full sequence length first to reduce flop count
            scaling = -core_sqd_knorms.unsqueeze(-1) + sqd_knorm.unsqueeze(-2) 
            scaling = scaling - scaling.amax((-1,-2), keepdim=True)
            W = W * torch.exp(scaling / 2.)
            # compressed_values = torch.einsum("...rn, ...nd -> ...rd", kernel_core, values)
            # compressed_one = kernel_core.sum(dim=-1)

            # Apply kernel inverse to get Nystrom weighting
            compressed_values = torch.einsum("...rl, ...ld -> ...rd", W, values)
            compressed_one = W.sum(dim=-1)

            compressed_values = compressed_values.reshape(B, self.r, D)
            compressed_one = compressed_one.reshape(B, self.r)

        # Reconstruction of attention output.
        # TODO: Test other implementation, e.g. via flash-attention

        # Incorporate temperature scaling for queries
        QK = scale*torch.einsum("...te, ...re -> ...tr", queries, core_keys)
        QK -= QK.amax(-1, keepdim=True)
        QK = QK.exp()

        QK1 = torch.einsum("...tr, ...r -> ...t", QK, compressed_one).unsqueeze(-1)

        # Multiply by Nystrom-weighted values
        # TODO: Determine reasonable cut-off threshold
        eps = 1e-20
        out = torch.where(QK1 > eps, torch.einsum("...tr, ...rd -> ...td", QK, compressed_values) / QK1, 0.)

        # Add in impact of value centers
        out = out + vbar 

        # Exact attention output should always lie in the range of the original
        # values, so enforce this constraint
        out = out.clamp(min = min_val, max = max_val)
        out = out.view(*queries_shape[:-1], D)

        return out


def rp_nystrom(
    keys: torch.Tensor,
    sqd_knorm: torch.Tensor,
    r: int,
    mode: str = "eager",
    accelerate = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Implements the randomly pivoted Cholesky algorithm optimized for torch.compile.

    Args:   
        keys (Tensor): torch.Tensor of shape (..., n, E) where n is the number of keys
        sqd_knorm (Tensor): squared norms of keys, shape (..., n)
        r (int): rank of the Nystrom approximation

    Returns:
        coreset (LongTensor): indices of the chosen landmark points; shape (..., r)
        weights (Tensor): Nystrom weights of shape (..., r, n)
    """
    keys_dtype, device = keys.dtype, keys.device
    dtype = torch.float32 if keys_dtype in [torch.bfloat16, torch.float16] else keys_dtype

    keys = keys.to(dtype)
    sqd_knorm = sqd_knorm.to(dtype)
    hsqd_knorm = sqd_knorm / 2.

    n = keys.shape[-2]
    batch_shape = keys.shape[:-2]

    # Pre-allocate all tensors
    kernel_core = torch.zeros((*batch_shape, r, n), dtype=dtype, device=device)
    kernel_core_dim = kernel_core.shape[0]
    kernel_inv = torch.zeros((*batch_shape, r, r), dtype=dtype, device=device)
    res_diagonal = torch.ones((*batch_shape, n), dtype=dtype, device=device)

    coreset_list = [None] * r 

    uniform = torch.empty((*batch_shape, n), dtype=dtype, device=device)
    g = torch.full((*batch_shape, r), -1., dtype=dtype, device=device)

    # Main loop:
    if mode == "eager":

        for i in range(r):
            # Sample with Gumbel-max trick (more compile-friendly)
            uniform.uniform_()
            scores = torch.log(res_diagonal) + sqd_knorm - torch.log(-torch.log(uniform))
            ids = torch.argmax(scores, dim=-1, keepdim=True)
            
            # Update coreset
            coreset_list[i] = ids

            if i > 0:
                # Gather kernel values for previously selected points
                a = torch.gather(kernel_core[:, :i, :], -1, ids[..., None].expand(kernel_core_dim, i, 1)).squeeze(2)
                
                # Compute Cholesky factor of kernel inverse
                # bmm faster than einsum
                g[..., :i] = torch.bmm(kernel_inv[..., :i, :i], a.unsqueeze(-1)).squeeze(-1)
                g[..., :i+1] *= torch.rsqrt(res_diagonal.gather(-1, ids))
                
            # Update kernel inverse in-place
            kernel_inv[..., :i+1, :i+1] += g[..., :i+1].unsqueeze(-1) * g[...,:i+1].unsqueeze(-2)
            
            # Compute kernel row corresponding to selected point
            kernel_row = gsn_kernel(keys, ids, hsqd_knorm).clamp(max = 1.)
            kernel_core[..., i, :] = kernel_row.squeeze(-2)

            if i < r-1:
                # Update residual diagonal
                y = torch.einsum(
                    "...si, ...s -> ...i", kernel_core[..., :i+1, :], g[..., :i+1])
                
                res_diagonal -= y.square()
                # Set diagonal entries for selected points to zero
                res_diagonal.scatter_(-1, ids, 0.0)
                # Enforce nonnegativity
                res_diagonal.clamp_(min=0.0)

        # Concatenate indices
        coreset = torch.cat(coreset_list, dim=-1)
        
    elif mode == "triton":
        # Compute initial Gumbel scores
        uniform.uniform_()
        scores = torch.log(res_diagonal) + sqd_knorm - torch.log(-torch.log(uniform))
        for i in range(r):
            # Select the index with the highest score (this is the Gumbel-max trick for sampling)
            ids = scores.argmax(dim=-1, keepdim=True)

            # Update coreset
            # Storing indices in a list and concatenating is faster than using scatter_
            coreset_list[i] = ids
            
            # Update kernel inverse
            if i > 0:
                # Gather kernel values for previously selected points
                a = torch.gather(kernel_core[:, :i, :], -1, ids[..., None].expand(kernel_core_dim, i, 1)).squeeze(2)
                
                # Compute Cholesky factor of kernel inverse
                # bmm faster than einsum
                g[..., :i] = torch.bmm(kernel_inv[..., :i, :i], a.unsqueeze(-1)).squeeze(-1)
                g[..., :i+1] *= torch.rsqrt(res_diagonal.gather(-1, ids))
                
            # Update kernel inverse in-place
            kernel_inv[..., :i+1, :i+1] += g[..., :i+1].unsqueeze(-1) * g[...,:i+1].unsqueeze(-2)
            
            update_kernel_triton(
                iteration=i,
                x=keys,
                x_hsqn=hsqd_knorm,
                ids=ids.squeeze(-1),
                kernel_core=kernel_core,
                g=g,
                res_diagonal=res_diagonal,
                uniform=uniform,
                scores=scores,
            )

        # Concatenate indices
        coreset = torch.cat(coreset_list, dim=-1)
    
    return coreset, kernel_inv.to(keys_dtype), kernel_core.to(keys_dtype)


def gsn_kernel(
        keys: torch.Tensor,
        ids: torch.LongTensor,
        halfsqdkeynorms: torch.Tensor,
    ) -> torch.Tensor:
        """Returns tensor of Gaussian kernel matrices
        kernel_mat
            = exp(keys[...,ids,:] @ keys[...,:,:].T 
                - halfsqdkeynorms[...,ids] - halfsqdkeynorms.T)

        Note: Assumes key has already been scaled appropriately by
        sqrt(softmax_temp)

        Args:
            key: tensor of shape [..., n, E]
            ids: tensor of shape [..., r]
            halfsqdkeynorms: tensor of shape [..., n]]]

        Returns tensor of shape [..., r, n]
        """
        E = keys.shape[-1]
        key_term = torch.einsum(
            '...re, ...ne -> ...rn', keys.gather(-2, ids.unsqueeze(-1).expand(*ids.shape, E)), keys)
        ###TODO: check if inplace exp_ is faster
        return torch.exp(key_term - halfsqdkeynorms.gather(-1, ids).unsqueeze(-1)
                        - halfsqdkeynorms.unsqueeze(-2))

# Two times the constant rho_0 = sqrt(1+exp(2W_0(2/e^2)+2))
# up to machine precision
TWO_RHO_0 = 6.383202050647408
def find_kernel_temperature(
        scale,
        q_scale,
        k_scale,
        n: int,
        phi: float | None = None,
):
    """Finds the relative scale between keys and queries that optimises the trade-off
    between low-rank approximability of the attention kernel incurred error factors.

    Args:   q_scale (Tensor): shape (batch_dims, 1) max_i ||q_i||_2
            k_scale (Tensor): shape (batch_dims, 1) max_i ||k_i||_2
            n (int): number of key-value pairs
            phi (float): adjustable hyperparameter, default 1.0
    """

    if phi is not None:
        n = n*phi**2

    b = math.log(n)/(scale*q_scale*k_scale) + 2.
    upper = b/(2*lambert_w_circ_exp((b/TWO_RHO_0).log()))
    tau = torch.sqrt(k_scale/q_scale * upper)

    return tau