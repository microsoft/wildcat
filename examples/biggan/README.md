# Image Generation with BigGAN Model

This example folder recreates the BigGAN image generation experiment.

## Prerequisites

We provide the Imagenet Validation statistics at [imagenet_val_inception_moments.npz](./imagenet_val_inception_moments.npz), so there is no need to download Imagenet for the BigGAN experiment. Similarly, model checkpoints will be automatically downloaded.

## Dependencies

To prepare a conda environment with all dependencies installed, first follow the [t2t dependency instructions](../t2t/README.md#dependencies). Then execute the following command: 
```bash
pip install boto3 requests scipy
```

## Results

Please follow the steps below to recreate the BigGAN experiment:

Navigate to wildcat/examples/biggan

To compute FID and IS scores, please run:

```bash
bash generate.sh
```

To test wildcat, only, please run:

```bash
python demo_generate_images.py --fid --attention wildcat
```

> \[!TIP\]
> The FID and IS scores are outputed to the console and to `fid_score_results.txt`.

To compute runtimes, please run:

```bash
bash runtime.sh
```

To generate a LaTeX results table, please run:

```bash
python table.py
```