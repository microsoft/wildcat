<p align="center">
  <img src="WildCat.png" alt="WildCat" width="300"/>
</p>

<h1 align="center">WildCat</h1>

<p align="center">
  <strong>Near-Linear Attention in Theory and in Practice</strong><br/>
  Tobias Schröder &amp; Lester Mackey
</p>

<p align="center">
  <a href="https://github.com/microsoft/wildcat"><img src="https://img.shields.io/badge/GitHub-microsoft%2Fwildcat-blue?logo=github" alt="GitHub"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green" alt="License: MIT"/></a>
  <img src="https://img.shields.io/badge/Python-%3C3.13-blue" alt="Python <3.13"/>
</p>

---

## Overview

WildCat is a drop-in PyTorch module for **sub-quadratic attention**. Given queries, keys, and values, WildCat compresses the key-value sequence into a small weighted coreset from which the full attention mechanism can be reconstructed, thereby achieving near-exact attention outputs at a fraction of the cost.

---
## Usage

Using the wildcat package could not be simpler: Our WildCat module can be used directly as drop-in replacement for standard attention implementations at inference time.

```python
import torch
from wildcat import WildCat

# Initialise module
attn = WildCat(
    r=128,           # coreset size (number of KV pairs to keep)
    num_bins=1,      # divide sequence into bins; compress each independently
    subsample_ratio=0.25,  # fallback compression ratio when r is not set
)

B, H, N, D = 2, 8, 4096, 64
queries = torch.randn(B, H, N, D)
keys    = torch.randn(B, H, N, D)
values  = torch.randn(B, H, N, D)

# Drop-in replacement for standard attention
output = attn(queries, keys, values)  # shape: (B, H, N, D)
```

The `compress_kv` function can also be used standalone to compress a KV cache:

```python
from wildcat import compress_kv

cmpd_keys, cmpd_values, weights = compress_kv(keys, values, r=128)
```

---

## Installation

WildCat requires Python < 3.13 and PyTorch. Install directly from GitHub:

```bash
pip install git+https://github.com/microsoft/wildcat.git
```

Or clone and install locally:

```bash
git clone https://github.com/microsoft/wildcat.git
cd wildcat
pip install -e .
```

**Dependencies:** `torch`, `numpy`

---

## Experiments

All experiment scripts live under `examples/`. Navigate into each subfolder before running commands.

### BigGAN Image Generation

**Prerequisites:** No dataset download required — ImageNet validation statistics are provided.

**Setup:**
```bash
# Follow the T2T-ViT dependency instructions first (see below), then:
pip install boto3 requests scipy
```

**Run:**
```bash
cd examples/biggan

# Reproduce all FID and IS scores
bash generate.sh

# Test WildCat only
python demo_generate_images.py --fid --attention wildcat

# Benchmark runtimes
bash runtime.sh

# Generate LaTeX results table
python table.py
```

---

### T2T-ViT ImageNet Classification

**Prerequisites:**
1. Download the ILSVRC2012 validation set from [image-net.org](https://www.image-net.org/download.php) (~6.3 GB).
2. Extract it:
   ```bash
   bash examples/t2t/extract_ILSVRC.sh
   ```
3. Download the pretrained model:
   ```bash
   mkdir -p examples/t2t/checkpoints
   wget -P examples/t2t/checkpoints \
     https://github.com/yitu-opensource/T2T-ViT/releases/download/main/82.6_T2T_ViTt_24.pth.tar
   ```

**Setup (with Scatterbrain):**
```bash
yes | conda create -n t2t python=3.12 cuda-nvcc cuda-cudart cuda-toolkit pip -c nvidia
conda activate t2t && export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130
git clone https://github.com/idiap/fast-transformers.git
sed -i 's/return \["-arch=compute_60"\]/return ["-arch=compute_80"]/' fast-transformers/setup.py
pip install --no-build-isolation fast-transformers/
pip install "timm==0.3.4" pyyaml einops lightning lightning-bolts numpy matplotlib pandas tabulate "setuptools<82"
yes | cp examples/t2t/helpers.py $CONDA_PREFIX/lib/python3.12/site-packages/timm/models/layers/helpers.py
pip install git+https://github.com/microsoft/thinformer.git
pip install git+https://github.com/microsoft/wildcat.git
```

> **Hopper GPUs (H100):** replace `compute_80` with `compute_90` in the `sed` command above.

**Run:**
```bash
cd examples/t2t

# Test WildCat only
python accuracy.py --method1 wildcat --method2 wildcat

# Reproduce all accuracy numbers
bash accuracy.sh

# Benchmark runtimes
bash runtime.sh

# Generate LaTeX results table
python table.py
```

---

### KV Cache Compression (LLMs)

**Setup:**
```bash
yes | conda create -n kvpress python=3.12
conda activate kvpress
pip install -r examples/kvcache/requirements.txt
pip install -e .
pip install kvpress==0.3.0 levenshtein
```

**Run:**
```bash
cd examples/kvcache

# Reproduce all quality numbers
bash benchmark.sh

# Test compress_kv with a single config
python evaluate.py --config_file evaluate_config.yaml --press_name compress_kv_12

# Generate LaTeX results table
python table.py
```

---

## How It Works

WildCat approximates the softmax attention output

$$\text{Attn}(Q, K, V) = \text{softmax}\!\left(\tfrac{QK^\top}{\sqrt{d}}\right) V$$

by replacing the full key-value sequence with a weighted coreset of size $r \ll N$. The figure below illustrates the low-rank approximation realised of WildCat.

<p align="center">
  <img src="VisualisationKDE.png" alt="WildCat" width="100%"/>
</p>

---


## License

This project is licensed under the [MIT License](LICENSE).
