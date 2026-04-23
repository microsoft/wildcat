# KV Cache Compression

This example folder recreates the KV cache compression experiments of [WildCat: Near-Linear Attention in Theory and Practice](https://arxiv.org/abs/2602.10056).

All scripts should be run from this directory.

## Dependencies

To prepare a conda environment with all dependencies:

```bash
yes | conda create -n kvpress python=3.12
conda activate kvpress
pip install -r requirements.txt
pip install -e ../../../wildcat
pip install kvpress==0.3.0
pip install levenshtein
```

## Resultsat

To test `compress_kv` in isolation, please run:

```bash
python evaluate.py --config_file evaluate_config.yaml --press_name compress_kv_12
```

To obtain the evaluate all KV cache compression methods, please run:

```bash
bash benchmark.sh
```

To generate a LaTeX results table, please run:

```bash
python table.py
```
