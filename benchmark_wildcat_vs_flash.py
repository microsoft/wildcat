"""
Benchmark: WildCat vs FlashAttention (non-causal forward pass only).

Usage:
    python benchmark_wildcat_vs_flash.py [--r R] [--num_bins NUM_BINS]
                                          [--batch_size B] [--head_size H] [--dim D]
                                          [--warmup W] [--rep REP]

The --error flag runs the max-entry error sweep instead of the timing benchmark.
"""

import argparse
import torch
import triton


from flash_attn import flash_attn_func as flash_attn_cuda

from wildcat import WildCat


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_tensors(batch_size, seq_len, head_size, dim, *, shared=False):
    """Return (q, k, v) in (batch, heads, seq, dim) layout."""
    shape = (batch_size, head_size, seq_len, dim)
    q = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    return q, k, v


def flash_fwd(q, k, v):
    # flash_attn expects (batch, seq, heads, dim).
    # Pass args positionally (dropout_p, softmax_scale, causal) to avoid
    # TypeError from C++ autograd Function implementations that reject kwargs.
    return flash_attn_cuda(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2)
    ).transpose(1, 2)


def wildcat_fwd(attn, q, k, v):
    # WildCat accepts (batch, heads, seq, dim) or (batch, seq, heads, dim);
    # align_shapes handles the merge of batch+heads axes.
    return attn(q, k, v)


# ---------------------------------------------------------------------------
# Warm-up (triggers Triton / torch.compile JIT for a given seq_len)
# ---------------------------------------------------------------------------

def warmup(batch_size, seq_len, head_size, dim, attn, n_warmup=3):
    """Run both methods without timing to force compilation."""
    q, k, v = get_tensors(batch_size, seq_len, head_size, dim)
    for _ in range(n_warmup):
        with torch.no_grad():
            flash_fwd(q, k, v)
            wildcat_fwd(attn, q, k, v)
    torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Timed benchmarks
# ---------------------------------------------------------------------------

def bench_flash(batch_size, seq_len, head_size, dim, warmup_iters=5, rep=100):
    q, k, v = get_tensors(batch_size, seq_len, head_size, dim)
    fn = lambda: flash_fwd(q, k, v)
    return triton.testing.do_bench(fn, warmup=warmup_iters, rep=rep,
                                   quantiles=[0.2, 0.5, 0.8])


def bench_wildcat(attn, batch_size, seq_len, head_size, dim, warmup_iters=5, rep=100):
    q, k, v = get_tensors(batch_size, seq_len, head_size, dim)
    fn = lambda: wildcat_fwd(attn, q, k, v)
    return triton.testing.do_bench(fn, warmup=warmup_iters, rep=rep,
                                   quantiles=[0.2, 0.5, 0.8])


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------

def max_entry_error(attn, batch_size, seq_len, head_size, dim):
    """
    Compute the maximum absolute entry-wise error between WildCat and
    FlashAttention outputs for a single random draw of (q, k, v).

    Returns:
        float: max |wildcat_out - flash_out|
    """
    q, k, v = get_tensors(batch_size, seq_len, head_size, dim)
    with torch.no_grad():
        ref = flash_fwd(q, k, v).float()
        approx = wildcat_fwd(attn, q, k, v).float()
    return (approx - ref).abs().max().item()


def run_error_sweep(seq_lens, batch_size, head_size, dim, r, num_bins, n_warmup=3):
    print(f"\n{'seq_len':<12}  {'max_abs_error'}")
    print("-" * 30)
    for seq_len in seq_lens:
        attn = WildCat(r=r, num_bins=num_bins).to(device="cuda", dtype=torch.bfloat16)
        warmup(batch_size, seq_len, head_size, dim, attn, n_warmup=n_warmup)
        err = max_entry_error(attn, batch_size, seq_len, head_size, dim)
        print(f"{seq_len:<12}  {err:.6f}")


def run_param_sweep(seq_lens, batch_size, head_size, dim, n_warmup=3,
                    output_file="param_sweep_results.txt"):
    """Sweep over all valid (seq_len, num_bins, r) triples where seq_len and r
    are both divisible by num_bins, and print the max absolute error for each.

    At least 25 (r, num_bins) combinations are explored per seq_len.
    Results are written to output_file in addition to stdout.
    """
    r_values        = [64, 128, 256, 512, 1024]
    num_bins_values = [1, 2, 4, 8, 16, 32, 64]

    header = f"{'num_bins':<10}  {'r':<8}  {'seq_len':<10}  {'max_abs_error'}"
    separator = "-" * len(header)

    def emit(line):
        print(line)
        f.write(line + "\n")

    with open(output_file, "w") as f:
        emit(f"\n{header}")
        emit(separator)

        for seq_len in seq_lens:
            for num_bins in num_bins_values:
                if seq_len % num_bins != 0:
                    continue
                for r in r_values:
                    if r % num_bins != 0 or r >= seq_len:
                        continue
                    attn = WildCat(r=r, num_bins=num_bins).to(device="cuda", dtype=torch.bfloat16)
                    warmup(batch_size, seq_len, head_size, dim, attn, n_warmup=n_warmup)
                    err = max_entry_error(attn, batch_size, seq_len, head_size, dim)
                    emit(f"{num_bins:<10}  {r:<8}  {seq_len:<10}  {err:.6f}")

    print(f"\nResults saved to {output_file}")


def collect_timed_results(args):
    """Run the timing benchmark for all seq_lens and collect results in a dict."""
    seq_lens = [2**i for i in range(10, args.log_2_max_len)]
    batch_size = args.batch_size
    head_size  = args.head_size
    dim        = args.dim
    num_bins   = [2**i for i in range(4, args.log_2_max_bins)]
    r = [2**i*bins for i in range(2, 4) for bins in num_bins]

    for seq_len in seq_lens:
        # Build a fresh WildCat for this seq_len
        attn = WildCat(r=args.r, num_bins=args.num_bins).to(device="cuda", dtype=torch.bfloat16)

        # Non-timed warm-up — forces Triton JIT compilation for this seq_len
        print(f"  [warmup] seq_len={seq_len} ...", end="\r", flush=True)
        warmup(batch_size, seq_len, head_size, dim, attn, n_warmup=args.n_warmup)

        # Timed runs
        flash_ms   = bench_flash(batch_size, seq_len, head_size, dim,
                                  warmup_iters=args.warmup, rep=args.rep)
        wildcat_ms = bench_wildcat(attn, batch_size, seq_len, head_size, dim,
                                    warmup_iters=args.warmup, rep=args.rep)

        print(f"{seq_len:<10}  {'flash':<12}  {flash_ms[0]:>10.4f}  {flash_ms[1]:>12.4f}  {flash_ms[2]:>10.4f}")
        print(f"{seq_len:<10}  {'wildcat':<12}  {wildcat_ms[0]:>10.4f}  {wildcat_ms[1]:>12.4f}  {wildcat_ms[2]:>10.4f}")
    return results

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def get_arguments():
    parser = argparse.ArgumentParser(
        description="Benchmark WildCat vs FlashAttention (non-causal fwd)."
    )
    parser.add_argument("--r", type=int, default=256,
                        help="Coreset size r for WildCat (default: use subsample_ratio=0.25)")
    parser.add_argument("--num_bins", type=int, default=64,
                        help="Number of bins for WildCat (default: 1)")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--head_size", type=int, default=32)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=5,
                        help="Warm-up iterations for triton.testing.do_bench")
    parser.add_argument("--rep", type=int, default=100,
                        help="Timed repetitions for triton.testing.do_bench")
    parser.add_argument("--n_warmup", type=int, default=3,
                        help="Untimed warm-up runs before benchmarking each seq_len")
    parser.add_argument("--error", action="store_true",
                        help="Run max-entry error sweep instead of timing benchmark")
    parser.add_argument("--param_sweep", action="store_true",
                        help="Sweep over (r, num_bins) pairs and report max-entry error")
    parser.add_argument("--param_sweep_out", type=str, default="param_sweep_results.txt",
                        help="Output file for --param_sweep results (default: param_sweep_results.txt)")
    return parser.parse_args()


def main():
    args = get_arguments()
    for k, v in vars(args).items():
        print(f"  {k:<14}: {v}")
    print()

    seq_lens = [2**i for i in range(10, 15)]

    batch_size = args.batch_size
    head_size  = args.head_size
    dim        = args.dim

    if args.error:
        run_error_sweep(
            seq_lens, batch_size, head_size, dim,
            r=args.r, num_bins=args.num_bins,
            n_warmup=args.n_warmup,
        )
        return

    if args.param_sweep:
        run_param_sweep(seq_lens, batch_size, head_size, dim, n_warmup=args.n_warmup,
                        output_file=args.param_sweep_out)
        return

    # ---- Timing benchmark ----
    col = f"{'seq_len':<10}  {'method':<12}  {'p20 (ms)':>10}  {'median (ms)':>12}  {'p80 (ms)':>10}"
    print(col)
    print("-" * len(col))

    for seq_len in seq_lens:
        # Build a fresh WildCat for this seq_len
        attn = WildCat(r=args.r, num_bins=args.num_bins).to(device="cuda", dtype=torch.bfloat16)

        # Non-timed warm-up — forces Triton JIT compilation for this seq_len
        print(f"  [warmup] seq_len={seq_len} ...", end="\r", flush=True)
        warmup(batch_size, seq_len, head_size, dim, attn, n_warmup=args.n_warmup)

        # Timed runs
        flash_ms   = bench_flash(batch_size, seq_len, head_size, dim,
                                  warmup_iters=args.warmup, rep=args.rep)
        wildcat_ms = bench_wildcat(attn, batch_size, seq_len, head_size, dim,
                                    warmup_iters=args.warmup, rep=args.rep)

        print(f"{seq_len:<10}  {'flash':<12}  {flash_ms[0]:>10.4f}  {flash_ms[1]:>12.4f}  {flash_ms[2]:>10.4f}")
        print(f"{seq_len:<10}  {'wildcat':<12}  {wildcat_ms[0]:>10.4f}  {wildcat_ms[1]:>12.4f}  {wildcat_ms[2]:>10.4f}")


if __name__ == "__main__":
    main()
