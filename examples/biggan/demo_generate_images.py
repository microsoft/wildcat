import argparse
import os
import torch
# import torchvision
import sys
import time
import random
import numpy as np
from tqdm import tqdm
if False:
    from pytorch_pretrained_biggan import (BigGAN, one_hot_from_names, 
                                        truncated_noise_sample, one_hot_from_int,
                                        save_as_images)

    from pytorch_pretrained_biggan.model_fast import FastBigGAN
    from pytorch_pretrained_biggan.model_performer import PerformerBigGAN
    from pytorch_pretrained_biggan.model_reformer import ReformerBigGAN
    from pytorch_pretrained_biggan.model_sblocal import SBlocalBigGAN
else:
    from biggan_models.model import BigGAN
    from biggan_models.utils import (
        truncated_noise_sample, 
        one_hot_from_int, 
        save_as_images
    )
    from biggan_models.model_fast import FastBigGAN
    from biggan_models.model_kdeformer import KDEformerBigGAN
    from biggan_models.model_performer import PerformerBigGAN
    from biggan_models.model_reformer import ReformerBigGAN
    from biggan_models.model_sblocal import SBlocalBigGAN
    from biggan_models.model_thinformer import ThinformerBigGAN
    from biggan_models.model_catformer import CATformerBigGAN

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",type=str, default='biggan-deep-512')
    parser.add_argument("--num_classes",type=int, default=1000)
    parser.add_argument("--data_per_class",type=int, default=1)
    parser.add_argument("--seed",type=int, default=1)
    parser.add_argument("--num_splits", "-ns",type=int, default=10)
    parser.add_argument("--batch_size",type=int, default=32)
    parser.add_argument("--attention",type=str, default='exact', choices=['exact', 'kde', 'kdeformer', 'performer', 'reformer', 'sblocal', 'thinformer', 'compressformer'])
    parser.add_argument("--truncation",type=float, default=0.4)
    parser.add_argument("--no_store",action='store_true')
    parser.add_argument("--fid",action='store_true')
    parser.add_argument("--debug",action='store_true')
    parser.add_argument("--postfix", type=str, default='')
    parser.add_argument("--r", "-r", type=int, default=96, help="WildCat rank parameter")
    parser.add_argument("--mode", type=str, default="eager", help="WildCat mode")
    parser.add_argument("--bins", "-b", type=int, default=8, help="WildCat number of bins")
    parser.add_argument("--dim_bins", "-db", type=int, default=1, help="WildCat number of dimension bins")
    
    return parser.parse_args()

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def main():
    args = get_args()
    seed_everything(args.seed)

    for aa, bb in args.__dict__.items():
        print(f"{aa}: {bb}")

    # data = torchvision.datasets.ImageNet("/home/ih244/workspace/data/imagenet/", split='val')

    model_name = args.model_name
    num_classes = args.num_classes
    data_per_class = args.data_per_class
    batch_size = args.batch_size
    attention = args.attention
    truncation = args.truncation

    # Load pre-trained model tokenizer (vocabulary)
    if attention == 'exact':
        model = BigGAN.from_pretrained(model_name)
    elif attention == 'kde':
        model = FastBigGAN.from_pretrained(model_name)
    elif attention == 'kdeformer':
        model = KDEformerBigGAN.from_pretrained(model_name)
    elif attention == 'performer':
        model = PerformerBigGAN.from_pretrained(model_name)
    elif attention == 'reformer':
        model = ReformerBigGAN.from_pretrained(model_name)
    elif attention == 'sblocal':
        model = SBlocalBigGAN.from_pretrained(model_name)
    elif attention == 'thinformer':
        model = ThinformerBigGAN.from_pretrained(model_name)
    elif attention == 'compressformer':
        model = CATformerBigGAN.from_pretrained(model_name, r=args.r,
                                                    mode=args.mode, bins=args.bins, dim_bins=args.dim_bins)
    else:
        raise NotImplementedError("Invalid attention option")

    print(model.__class__)

    # Prepare a input
    labels = np.repeat(np.arange(num_classes), data_per_class).tolist()
    class_vector = one_hot_from_int(labels, batch_size=len(labels))
    noise_vector = truncated_noise_sample(truncation=truncation, batch_size=len(labels), seed=args.seed)

    # All in tensors
    noise_vector = torch.from_numpy(noise_vector)
    class_vector = torch.from_numpy(class_vector)

    if torch.cuda.is_available():
        # If you have a GPU, put everything on cuda
        noise_vector = noise_vector.to('cuda')
        class_vector = class_vector.to('cuda')
        model = model.to('cuda')

    tic = time.time()
    model.eval()
    output_all = []
    num_batches = len(labels) // batch_size + 1
    for idx in tqdm(range(num_batches)):
        batch_idx = list(range(idx * batch_size, min(len(labels), (idx+1) * batch_size)))
        if len(batch_idx) == 0:
            continue
        # res_all.append(batch_idx)

        n_vec = noise_vector[batch_idx]
        c_vec = class_vector[batch_idx]

        # Generate an image
        # print("n_vec.sum: ", n_vec.sum())
        # print("c_vec.sum: ", c_vec.sum())
        # print("truncation: ", truncation)
        output = model(n_vec, c_vec, truncation)
        output = output.to('cpu')
        # print("output.sum: ", output.sum())

        output_all.append(output)

    time_generation = time.time() - tic

    output_all = torch.cat(output_all)
    print(f"output_all.shape: {output_all.shape}")
    print(f"generation time : {time_generation:.4f} sec")
    del model, noise_vector, class_vector

    if args.fid:
        print("computing FID & Inception scores ...")
        from demo_inception_score import get_logits
        import inception_utils
        
        pool, logits = get_logits(output_all)
        is_mean_fake, is_std_fake = inception_utils.calculate_inception_score(logits.cpu().numpy(), num_splits=args.num_splits)
        print(f"Inception score : {is_mean_fake:.5f} (std : {is_std_fake:.5f})", flush=True)

        mu, sigma = np.mean(pool.cpu().numpy(), axis=0), np.cov(pool.cpu().numpy(), rowvar=False)
        data_mu = np.load('imagenet_val_inception_moments.npz')['mu']
        data_sigma = np.load('imagenet_val_inception_moments.npz')['sigma']

        fid_value = inception_utils.numpy_calculate_frechet_distance(mu, sigma, data_mu, data_sigma)
        print(f"FID  : {fid_value}", flush=True)

        print("Saving results to file", flush=True)
        if attention == 'compressformer':
            attention = f"compressformer_r{args.r}_b{args.bins}"
        res_str = f"model: {args.model_name}, data_per_class: {data_per_class}, num_splits: {args.num_splits}, seed: {args.seed}, attention: {attention:<30}, fid: {fid_value}, is_mean_fake: {is_mean_fake}, is_std_fake: {is_std_fake}\n"
        with open("./fid_score_results.txt", "a") as f:
            f.write(res_str)

        if not args.no_store:
            print("Saving images to disk", flush=True)
            output_path = f"./generations/{model_name.replace('-','_')}/{attention}-n{len(labels)}{args.postfix}-ns{args.num_splits}-s{args.seed}"
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            tic = time.time()
            num_images_to_save = 2
            print(f"saving {num_images_to_save} images....")
            save_as_images(output_all[:num_images_to_save], output_path + "/img")
            print(f"done. ({time.time() - tic:.4f} sec)")

def fid_test():

    imgs = torch.randint(0, 255, (100, 3, 512, 512), dtype=torch.uint8)
    from torchmetrics.image.inception import InceptionScore
    aa = InceptionScore(feature=2048)
    xx = aa.inception(imgs)
    # # inception.update(imgs)
    # print(xx.shape)

    # from torchmetrics.image.fid import FrechetInceptionDistance
    # bb = FrechetInceptionDistance(feature=2048)
    # yy = bb.inception(imgs)
    # print(yy.shape)

    # from pytorch_fid.fid_score import get_activations, calculate_frechet_distance
    from pytorch_fid.inception import InceptionV3
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    inception_model = InceptionV3([block_idx])
    pred = get_activations((imgs / 255.).float(), inception_model)


    # from torchmetrics.image.fid import NoTrainInceptionV3
    # cc = NoTrainInceptionV3(name="inception-v3-compat", features_list=[str(2048)])
    # zz = cc(imgs)
    # is_mean, is_std = inception.compute()

    import pdb; pdb.set_trace();



if __name__ == "__main__":
    # fid_test()
    main()
 