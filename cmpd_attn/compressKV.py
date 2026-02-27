import torch
import torch.nn as nn
from math import sqrt
import math
import random

from cmpd_attn.wildcat import rp_nystrom, find_kernel_temperature

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

    core_keys = core_keys.reshape(*keys_shape[:-2], r, E)
    #core_values = core_values.reshape(*values_shape[:-2], r, E)
    KV = KV.reshape(*values_shape[:-2], r, E)
    K1 = K1.reshape(*keys_shape[:-2], r, 1)

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


    