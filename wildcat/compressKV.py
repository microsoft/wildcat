import torch
import torch.nn as nn
from math import sqrt
import math
import random

from wildcat.math_utils import lambert_w_circ_exp
#from cmpd_attn.wildcat import rp_nystrom, find_kernel_temperature

def compress_kv(
        keys,
        values,
        scale,
        r
):
    keys_shape = keys.shape
    values_shape = values.shape

    keys = keys.reshape(-1, keys.shape[-2], keys.shape[-1])
    values = values.reshape(-1, values.shape[-2], values.shape[-1])

    E = keys.shape[-1]
    n = keys.shape[-2]

    sqd_knorm = keys.square().sum(dim=-1)

    k_scale = sqd_knorm.sqrt().amax(dim = -1, keepdim=True)

    # Shape (B*C*F, 1)
    tau = find_kernel_temperature(
        scale = scale,
        q_scale=k_scale,
        k_scale=k_scale,
        n = n,
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
        r=r,
        mode="eager",
    )

    # Select compressed keys:
    # Shape (B*C*F, r//C, E//F)
    core_keys = keys.gather(-2, coreset.unsqueeze(-1).expand(*coreset.shape, E))
    core_sqd_knorms = sqd_knorm.gather(-1, coreset)
    # Undo rescaling of keys
    # Undoing of rescaling only has to be applied to core_keys, not the norms, since the weights are computed for rescaled keys.
    core_keys /= key_multiplier.unsqueeze(-1)

    # Compute Nystrom weights for Gaussian kernel
    W = torch.einsum("...rs, ...sl -> ...rl", kernel_inv, kernel_core)

    # Adjust to weights for the exponential kernel
    scaling = -core_sqd_knorms.unsqueeze(-1) + sqd_knorm.unsqueeze(-2)
    ###scaling = scaling - scaling.amax((-1,-2), keepdim=True) # this line is only valid if window and sinks get scaled too?
    W = W * torch.exp(scaling / 2.)

    KV = torch.einsum("...rn, ...nd -> ...rd", W, values)
    K1 = W.sum(dim=-1)

    # core_keys = core_keys.reshape(*keys_shape[:-2], r, E)
    # #core_values = core_values.reshape(*values_shape[:-2], r, E)
    # KV = KV.reshape(*values_shape[:-2], r, E)
    # K1 = K1.reshape(*keys_shape[:-2], r, 1)

    return core_keys, KV, K1
    

def compress(
        module: torch.nn.Module | None,
        hidden_states: torch.Tensor | None,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        r: int | float,
        kwargs: dict
):

    scale = module.scaling

    return compress_kv(keys, values, scale, r)

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


    