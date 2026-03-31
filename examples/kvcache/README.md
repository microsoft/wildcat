# KV Cache Compression

This example folder recreates the KV cache compression experiments.

To ensure correct file paths, navigate to wildcat/examples/kvcache

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

To obtain the quality numbers for all methods, please run:

```bash
bash benchmark.sh
```

To test compresskv, run:

```bash
python evaluate.py --config_file evaluate_config.yaml --press_name compress_kv_12
```

To generate a LaTeX results table, please run:

```bash
python table.py
```