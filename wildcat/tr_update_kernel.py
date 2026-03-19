import torch
import triton
import triton.language as tl
import random

configs = [
    triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_E': 32, "num_warps":4, "num_stages":2}),
    triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_E': 32, "num_warps":8, "num_stages":1}),
    triton.Config({'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_E': 64, "num_warps":8, "num_stages":1}),
    triton.Config({'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_E': 128, "num_warps":8, "num_stages":2}),
]

configs_gauss = [
    triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_E': 32}, num_warps=4, num_stages=1),
    triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_E': 128}, num_warps=4, num_stages=1),
    triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_E': 32}, num_warps=4, num_stages=1),
    triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_E': 64}, num_warps=8, num_stages=1),
    triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_E': 64}, num_warps=8, num_stages=2),
]

res_diagonal_configs = [
    triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_R': 32}, num_warps=4, num_stages=1),
    triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_R': 64}, num_warps=4, num_stages=1),
    triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_R': 128}, num_warps=4, num_stages=1),
    triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_R': 32}, num_warps=8, num_stages=1),
    triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_R': 64}, num_warps=8, num_stages=2),
]

def next_power_of_2(x):  
    return 1 if x == 0 else 2**(x - 1).bit_length()

def update_kernel_triton(
        iteration,
        x,
        x_hsqn,
        ids,
        kernel_core,
        g,
        res_diagonal,
        uniform,
        scores
        #maxima,
        #ids_out
        ):
    
    """
    Computes sampled rows of the kernel, updates the residual diagonal,
    and samples pivot elements for the next iteration.

    Args:
        iteration: int, current iteration (0 <= iteration < R)
        x: tensor of shape (BATCH_SIZE, N, E), input data set
        x_hsqn: tensor of shape (BATCH_SIZE, N), half squared norms of x
        ids: tensor of shape (BATCH_SIZE,), indices of current pivots
        kernel_core: tensor of shape (BATCH_SIZE, R, N), kernel matrix of sampled rows
        g: tensor of shape (BATCH_SIZE, iteration + 1), row of Cholesky factor of kernel inverse
        res_diagonal: tensor of shape (BATCH_SIZE, N), current residual diagonal used for pivoting

    Returns:
        ids_out: tensor of shape (BATCH_SIZE,), indices of next pivots
        Updates kernel_core and res_diagonal in place.
    """
    
    BATCH_SIZE, N, E = x.shape
    # R = kernel_core.shape[1]
    R = iteration + 1

    # Tiling of program.
    # TODO: tune these parameters
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_E = 32
    BLOCK_SIZE_R = min(32, next_power_of_2(R))
    #print(f"Using BLOCK_SIZE_R: {BLOCK_SIZE_R}", flush=True)


    # Each program instance processes a batch element and a block of N elements.
    # The grid specifies which parts of the output are computed by individual program instances
    grid = lambda META: (
        BATCH_SIZE,
        triton.cdiv(N, META["BLOCK_SIZE_N"])
    )

    # Strides to compute kernel row:
    x_b, x_n, x_e = x.stride(0), x.stride(1), x.stride(2)
    x_hsqn_b, x_hsqn_n = x_hsqn.stride(0), x_hsqn.stride(1)
    kc_b, kc_r, kc_n = kernel_core.stride(0), kernel_core.stride(1), kernel_core.stride(2)

    # TODO: Fuse kernel row computation and residual diagonal update
    # Gaussian kernel row computation
    # gsn_row_kernel[grid](
    #     iteration,
    #     x,
    #     x_hsqn,
    #     ids,
    #     kernel_core,
    #     BATCH_SIZE, N, E,
    #     x_b, x_n, x_e,
    #     x_hsqn_b, x_hsqn_n,
    #     kc_b, kc_r, kc_n,
    #     # BLOCK_SIZE_N = BLOCK_SIZE_N,
    #     # BLOCK_SIZE_E = BLOCK_SIZE_E
    # )

    # TODO: Generate noise on-the-fly in the kernel. For this, find ways to modify seed
    #seed = 3141
    #seed = 1234
    #uniform = torch.zeros((BATCH_SIZE, N), dtype=x.dtype, device=x.device)
    uniform.uniform_()

    # Argmax of gumbel score is computed per block
    # Maxima are reduced outside of kernel in pytorch

    # Over allocation of memory to make BLOCK_SIZE_N tunable
    # lower_BLOCK_SIZE_N = 16
    # maxima = torch.zeros((BATCH_SIZE, -(N//-lower_BLOCK_SIZE_N)), device=x.device, dtype=x.dtype).log()
    # ids_out = torch.zeros((BATCH_SIZE, -(N//-lower_BLOCK_SIZE_N)), device=ids.device, dtype=ids.dtype)

    # Strides to compute residual diagonal update
    g_b, g_r = g.stride(0), g.stride(1)
    rd_b, rd_n = res_diagonal.stride(0), res_diagonal.stride(1)
    u_b, u_n = uniform.stride(0), uniform.stride(1)
    s_b, s_n = scores.stride(0), scores.stride(1)
    #m_b, m_n = maxima.stride(0), maxima.stride(1)
    #i_b, i_n = ids_out.stride(0), ids_out.stride(1)

    fused_update_kernel[grid](
        iteration,
        x,
        x_hsqn,
        ids,
        kernel_core,
        g,
        res_diagonal,
        uniform,
        scores,
        # maxima,
        # ids_out,
        BATCH_SIZE, N, E, R,
        x_b, x_n, x_e,
        x_hsqn_b, x_hsqn_n,
        kc_b, kc_r, kc_n,
        g_b, g_r,
        rd_b, rd_n,
        u_b, u_n,
        s_b, s_n,
        # m_b, m_n,
        # i_b, i_n,
        BLOCK_SIZE_N = BLOCK_SIZE_N,
        BLOCK_SIZE_E = BLOCK_SIZE_E,
        BLOCK_SIZE_R = BLOCK_SIZE_R,
        num_warps = 8,
        num_stages = 1
    )

    #    seed = seed,

    # res_diagonal_kernel[grid](
    #     g,
    #     kernel_core,
    #     ids,
    #     res_diagonal,
    #     uniform,
    #     x_hsqn,
    #     maxima,
    #     ids_out,
    #     BATCH_SIZE, N, R,
    #     g_b, g_r,
    #     kc_b, kc_r, kc_n,
    #     rd_b, rd_n,
    #     u_b, u_n,
    #     x_hsqn_b, x_hsqn_n,
    #     m_b, m_n,
    #     i_b, i_n,
    # #    seed = seed,
    #     # BLOCK_SIZE_N = BLOCK_SIZE_N,
    #     # BLOCK_SIZE_R = BLOCK_SIZE_R
    # )

    # Potentially improve this reduction using atomic operations in the kernel
    # This way, the entire algorithm could be fused into a single kernel

    # best_config = res_diagonal_kernel.get_best_config()
    #print(fused_update_kernel.best_config.kwargs)
    # best_BLOCK_SIZE_N = res_diagonal_kernel.best_config.kwargs['BLOCK_SIZE_N']
    # n_block = -(N//-best_BLOCK_SIZE_N)

    # maxima = maxima[:, :n_block]
    # ids_out = ids_out[:, :n_block]

    # max_ids = maxima.argmax(dim=-1)
    # ids_out = ids_out.gather(1, max_ids[:, None]).squeeze(1)
    #print(f"Selected BLOCK_SIZE_N: {best_BLOCK_SIZE_N}")
    #return ids_out

@triton.autotune(
    configs=configs_gauss,
    key=['N', 'E'],
)
@triton.jit
def gsn_row_kernel(
        iteration,
        x_ptr,
        x_hsqn_ptr,
        ids_ptr,
        out_ptr, 
        BATCH_SIZE, N, E,
        x_b, x_n, x_e,
        x_hsqn_b, x_hsqn_n,
        o_b, o_r, o_n,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_E: tl.constexpr
    ):

    """
    Computes sampled rows of the Gaussian kernel matrix.
    Expects output of shape (BATCH_SIZE, R, N)
    Output corresponds to R serially computed kernel rows

    Args:
        iteration: int, current iteration (0 <= iteration < R)
        input tensors and their descriptions

    Updates:
        out_ptr: tensor of shape (BATCH_SIZE, R, N), kernel matrix of sampled rows
        row at index 'iteration' is updated in place.
    """


    pid_b = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    if pid_b >= BATCH_SIZE:
        return
    
    # offsets along N dimension
    n_offs = pid_n*BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    e_offs = tl.arange(0, BLOCK_SIZE_E)
    # Guard to only load tensors up to maximum sequence length
    n_mask = n_offs < N

    # read query id for this batch
    ids = tl.load(ids_ptr + pid_b)

    acc = tl.zeros((BLOCK_SIZE_N, ), dtype=tl.float32)

    for e_iter in range(tl.cdiv(E, BLOCK_SIZE_E)):
        e_mask = e_offs < E

        # Pointer to the query
        x_query_ptr = x_ptr + pid_b * x_b + (ids*x_n) + (e_offs*x_e)
        x_block_ptr = x_ptr + pid_b * x_b + (n_offs[:, None] * x_n) + (e_offs[None, :] * x_e)

        x_block = tl.load(x_block_ptr, mask=n_mask[:, None] & e_mask[None, :], other=0.0)
        x_query = tl.load(x_query_ptr)

        #compute dot product with the query ids
        #dot = tl.sum((x_block - x_query[None, :])*(x_block - x_query[None, :]), axis=-1)
        dot = tl.sum(x_block * x_query[None, :], axis=-1)

        acc += dot
        e_offs += BLOCK_SIZE_E

    x_hsqn_query = tl.load(x_hsqn_ptr + pid_b * x_hsqn_b + ids * x_hsqn_n)
    x_hsqn_block = tl.load(x_hsqn_ptr + pid_b * x_hsqn_b + n_offs * x_hsqn_n, mask=n_mask, other=0.0)

    #out = tl.exp(-0.5*acc)
    # TODO: Use exp2 to reduce overhead by rescaling queries and keys bei log(2)
    out = tl.exp(acc - x_hsqn_query - x_hsqn_block)

    out_ptrs = out_ptr + pid_b * o_b + iteration*o_r + n_offs * o_n
    tl.store(out_ptrs, out, mask=n_mask)

@triton.autotune(
    configs=res_diagonal_configs,
    key=['N', 'R'],
)
@triton.jit
def res_diagonal_kernel(
        g_ptr,
        kernel_core_ptr,
        ids_ptr,
        out_ptr,
        noise_ptr,
        x_hsqn_ptr,
        maxima_ptr,
        ids_out_ptr,
        BATCH_SIZE, N, R,
        g_b, g_r,
        kc_b, kc_r, kc_n,
        o_b, o_n,
        noise_b, noise_n,
        x_hsqn_b, x_hsqn_n,
        m_b, m_n,
        i_b, i_n,
    #    seed: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_R: tl.constexpr
    ):

    pid_b = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    if pid_b >= BATCH_SIZE:
        return
    
    # offsets along N dimension
    n_offs = pid_n*BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    r_offs = tl.arange(0, BLOCK_SIZE_R)
    # Guard to only load tensors up to maximum sequence length
    n_mask = n_offs < N

    ids = tl.load(ids_ptr + pid_b)
    ids_mask = (ids != n_offs)

    out_ptrs = out_ptr + pid_b * o_b + n_offs * o_n
    
    acc = tl.zeros((BLOCK_SIZE_N, ), dtype=tl.float32)

    for r_iter in range(tl.cdiv(R, BLOCK_SIZE_R)):
        r_mask = r_offs < R

        g_ptrs = g_ptr + pid_b * g_b + r_offs*g_r
        kernel_core_ptrs = kernel_core_ptr + pid_b * kc_b + (n_offs[:, None] * kc_n) + (r_offs[None, :] * kc_r)

        g_vec = tl.load(g_ptrs)
        kernel_core_mat = tl.load(kernel_core_ptrs, mask=n_mask[:, None] & r_mask[None, :], other=0.0)
        
        dot = tl.sum(kernel_core_mat * g_vec[None, :], axis=-1)

        acc += dot
        r_offs += BLOCK_SIZE_R

    acc *= acc

    res_diagonal = tl.load(out_ptrs, mask=n_mask, other=0.0)
    res_diagonal = tl.where((res_diagonal > 0.)&ids_mask, res_diagonal - acc, 0.)
    res_diagonal = tl.clamp(res_diagonal, 0.0, 1.)

    #uniform_noise = tl.rand(seed=seed, offset = n_offs)
    uniform_noise = tl.load(noise_ptr + pid_b * noise_b + n_offs * noise_n, mask=n_mask, other=0.0)
    sqd_norm = 2*tl.load(x_hsqn_ptr + pid_b * x_hsqn_b + n_offs * x_hsqn_n, mask=n_mask, other=0.0)
    gumbel_score = tl.where((res_diagonal > 0.)&ids_mask, tl.log(res_diagonal) + sqd_norm -tl.log(-tl.log(uniform_noise)), -float("inf"))
    
    max_score, max_ids = tl.max(gumbel_score, axis=-1, return_indices = True)

    tl.store(maxima_ptr + pid_b*m_b + pid_n*m_n, max_score)
    tl.store(ids_out_ptr + pid_b*i_b + pid_n*i_n, pid_n*BLOCK_SIZE_N + max_ids)
    tl.store(out_ptrs, res_diagonal, mask=n_mask)

    

# @triton.autotune(
#     configs=configs,
#     key=['N', 'E', 'WARPS', 'STAGES'],
# )
@triton.jit
def fused_update_kernel(
        iteration,
        x_ptr,
        x_hsqn_ptr,
        ids_ptr,
        kernel_core_ptr,
        g_ptr,
        out_ptr,
        noise_ptr,
        score_ptr,
        # maxima_ptr,
        # ids_out_ptr,
        BATCH_SIZE, N, E, R,
        x_b, x_n, x_e,
        x_hsqn_b, x_hsqn_n,
        kc_b, kc_r, kc_n,
        g_b, g_r,
        o_b, o_n,
        noise_b, noise_n,
        s_b, s_n,
        # m_b, m_n,
        # i_b, i_n,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_E: tl.constexpr,
        BLOCK_SIZE_R: tl.constexpr,
        num_warps: tl.constexpr,
        num_stages: tl.constexpr,
):
    #---- Compute Gaussian kernel row ----
    
    pid_b = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    if pid_b >= BATCH_SIZE:
        return
    
    # offsets along N dimension
    n_offs = pid_n*BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    e_offs = tl.arange(0, BLOCK_SIZE_E)
    r_offs = tl.arange(0, BLOCK_SIZE_R)
    # Guard to only load tensors up to maximum sequence length
    n_mask = n_offs < N

    # read query id for this batch
    ids = tl.load(ids_ptr + pid_b)

    acc = tl.zeros((BLOCK_SIZE_N, ), dtype=tl.float32)

    for e_iter in range(tl.cdiv(E, BLOCK_SIZE_E)):
        e_mask = e_offs < E

        # Pointer to the query
        x_query_ptr = x_ptr + pid_b * x_b + (ids*x_n) + (e_offs*x_e)
        x_block_ptr = x_ptr + pid_b * x_b + (n_offs[:, None] * x_n) + (e_offs[None, :] * x_e)

        x_block = tl.load(x_block_ptr, mask=n_mask[:, None] & e_mask[None, :], other=0.0)
        x_query = tl.load(x_query_ptr)

        #compute dot product with the query ids
        #dot = tl.sum((x_block - x_query[None, :])*(x_block - x_query[None, :]), axis=-1)
        dot = tl.sum(x_block * x_query[None, :], axis=-1)

        acc += dot
        e_offs += BLOCK_SIZE_E

    x_hsqn_query = tl.load(x_hsqn_ptr + pid_b * x_hsqn_b + ids * x_hsqn_n)
    x_hsqn_block = tl.load(x_hsqn_ptr + pid_b * x_hsqn_b + n_offs * x_hsqn_n, mask=n_mask, other=0.0)

    #out = tl.exp(-0.5*acc)
    # TODO: Use exp2 to reduce overhead by rescaling queries and keys bei log(2)
    out = tl.exp(acc - x_hsqn_query - x_hsqn_block)

    # Write result in kernel core at row 'iteration'
    kernel_row_ptrs = kernel_core_ptr + pid_b * kc_b + iteration*kc_r + n_offs * kc_n
    tl.store(kernel_row_ptrs, out, mask=n_mask)

    # ---- Update residual diagonal and sample pivots ----

    ids_mask = (ids != n_offs)

    # Pointer for residual diagonal
    out_ptrs = out_ptr + pid_b * o_b + n_offs * o_n
    
    #Reset accumulation tensor
    #acc = tl.zeros((BLOCK_SIZE_N, ), dtype=tl.float32)
    acc *= 0.

    for r_iter in range(tl.cdiv(R, BLOCK_SIZE_R)):
        r_mask = r_offs < R

        g_ptrs = g_ptr + pid_b * g_b + r_offs*g_r
        kernel_core_ptrs = kernel_core_ptr + pid_b * kc_b + (n_offs[:, None] * kc_n) + (r_offs[None, :] * kc_r)

        g_vec = tl.load(g_ptrs)
        kernel_core_mat = tl.load(kernel_core_ptrs, mask=n_mask[:, None] & r_mask[None, :], other=0.0)
        
        dot = tl.sum(kernel_core_mat * g_vec[None, :], axis=-1)

        acc += dot
        r_offs += BLOCK_SIZE_R

    acc *= acc

    res_diagonal = tl.load(out_ptrs, mask=n_mask, other=0.0)
    res_diagonal = tl.where((res_diagonal > 0.)&ids_mask, res_diagonal - acc, 0.)
    res_diagonal = tl.clamp(res_diagonal, 0.0, 1.)

    #uniform_noise = tl.rand(seed=seed, offset = n_offs)
    uniform_noise = tl.load(noise_ptr + pid_b * noise_b + n_offs * noise_n, mask=n_mask, other=0.0)
    sqd_norm = 2*tl.load(x_hsqn_ptr + pid_b * x_hsqn_b + n_offs * x_hsqn_n, mask=n_mask, other=0.0)
    gumbel_score = tl.where((res_diagonal > 0.)&ids_mask, tl.log(res_diagonal) + sqd_norm -tl.log(-tl.log(uniform_noise)), -float("inf"))
    
    tl.store(score_ptr + pid_b * s_b + n_offs * s_n, gumbel_score, mask=n_mask)
    #max_score, max_ids = tl.max(gumbel_score, axis=-1, return_indices = True)

    #tl.store(maxima_ptr + pid_b*m_b + pid_n*m_n, max_score)
    #tl.store(ids_out_ptr + pid_b*i_b + pid_n*i_n, pid_n*BLOCK_SIZE_N + max_ids)
    tl.store(out_ptrs, res_diagonal, mask=n_mask)


if __name__ == "__main__":
    # Example usage
    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

    def update_gumbel_scores_ref(iteration, x, x_hsqn, ids, kernel_core, g, res_diagonal):
        i = iteration
        BATCH_SIZE, N, E = x.shape

        x_query = torch.gather(x, 1, ids[:, None, None].expand(BATCH_SIZE, 1, E)).squeeze(1)
        
        kernel_core[:, iteration, :] = torch.exp(-0.5*(x - x_query[:, None, :]).pow(2).sum(-1))
        y = torch.einsum("...s, ...si -> ...i", g[..., :i+1], kernel_core[..., :i+1, :])

        acc = y.pow(2)
        res_diagonal = torch.where(res_diagonal > 0, res_diagonal - acc, torch.zeros_like(res_diagonal))
        res_diagonal.scatter_(-1, ids[:, None], 0.0)

        # y = torch.einsum(
        #     "...si, ...s -> ...i", kernel_core[..., :i+1, :], kernel_inv_update[..., :i+1])
        # nys_diagonal += y.pow(2)
        # #/C.unsqueeze(-1)
        
        # # Update residual diagonal
        # mask = (res_diagonal > 0)
        # res_diagonal = torch.where(mask, (1.0 - nys_diagonal), torch.zeros_like(res_diagonal))
        # res_diagonal.scatter_(-1, ids, 0.0)

        return kernel_core, res_diagonal


    BATCH_SIZE, R, N, E = 3, 128, 1047, 64
    iteration = 47

    x = torch.randn((BATCH_SIZE, N, E), dtype=torch.float32, device=device)
    x_hsqn = x.pow(2).sum(dim=-1) / 2.  # half-squared norms
    ids = torch.randint(0, N, (BATCH_SIZE,), dtype=torch.int64, device=device)

    kernel_core = torch.zeros((BATCH_SIZE, R, N), device=device, dtype=x.dtype)
    g = torch.randn((BATCH_SIZE, R), device=device, dtype=x.dtype)
    res_diagonal = torch.randn((BATCH_SIZE, N), device=device, dtype=x.dtype)

    kernel_core_ref = kernel_core.clone()
    res_diagonal_ref = res_diagonal.clone()

    update_kernel_triton(
        iteration,
        x,
        x_hsqn,
        ids,
        kernel_core,
        g,
        res_diagonal)
    
    kernel_core_row_triton = kernel_core[:, iteration, :]
    res_diagonal_triton = res_diagonal

    kernel_core_ref, res_diagonal_ref = update_gumbel_scores_ref(
        iteration,
        x,
        x_hsqn,
        ids,
        kernel_core_ref,
        g,
        res_diagonal_ref)


    #comparison = torch.einsum("bne, be -> bn", x, x_query)
    print(kernel_core_row_triton)
    print(kernel_core_ref[:, iteration, :])
    print("----")
    print(res_diagonal_triton)
    print(res_diagonal_triton.gather(1, ids[:, None]).squeeze(1))  # Should be unchanged
    print(res_diagonal_ref)
    print("----")
    print((res_diagonal_triton - res_diagonal_ref).abs().max())  # Should be small