import torch
import argparse

from soul_gan.distribution import GANTarget
from soul_gan.sample import soul
from soul_gan.feature import FeatureFactory
#from soul_gan.models import ModelFactory
from general_utils import DotConfig, random_seed
from pathlib import Path
import datetime
import numpy as np


def load_gan(config):
    gen_class = ModelFactory.create_model(config.gen.name)
    gen = gen_class(config.gen.params)

    dis_class = ModelFactory.create_model(config.dis.name)
    dis = dis_class(config.dis.params)

    gen.eval()
    dis.eval()
    return gen, dis


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('gan_config', type=str)
    # parser.add_argument('--n_steps', type=int, default=100)
    # parser.add_argument('--n_sampling_steps', type=int, default=3)
    # parser.add_argument('--seed', type=int)
    # parser.add_argument('--device', type=int)

    args = parser.parse_args()
    return args
    

def main(config, gan_config, device):
    gen, dis = load_gan(gan_config)
    gen = gen.to(device)
    dis = dis.to(device)

    if 'sample' in config.__dict__:
        log_dir = Path(config.sample.log_dir,  datetime.datetime.now().strftime("%Y_%m_%d_%H_%M"))
        log_dir.mkdir(exist_ok=True, parents=True)

        feature_class = FeatureFactory.create_feature(config.feature.name)
        feature = feature_class(config.feature.params)

        z_dim = gen.z_dim
        proposal = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(z_dim), torch.eye(z_dim))
        ref_dist = GANTarget(gen, dis, proposal)
       
        total_sample = []
        for i in range(0, config.sample.total_n, config.sample.batch_size):
            z = torch.randn(config.sample.batch_size, z_dim)
            zs = soul(z, gen, ref_dist, feature, **config.sample.params)
            zs = torch.stack(zs, 0)
            total_sample.append(zs)

        total_sample = torch.cat(total_sample, 1)  # (number_of_steps / every) x total_n x latent_dim

        latents_dir = Path(log_dir, 'latents')
        latents_dir.mkdir(exist_ok=True)
        for slice_id, slice in enumerate(total_sample):
            np.save(Path(latents_dir, f'{slice_id * config.sample.save_every}.npy', slice.numpy()))

    if 'compute_fid' in config.__dict__:
        pass

    if 'compute_is' in config.__dict__:
        pass


if __name__ == '__main__':
    args = parse_arguments()
    config = DotConfig(args.config)
    gan_config = DotConfig(args.gan_config)
    # if args.n_steps is not None:
    #     config.n_steps = args.n_steps
    # if args.n_sampling_steps is not None:
    #     config.n_sampling_steps = args.n_sampling_steps
    # if args.seed is not None:
    #     config.seed = args.seed
    # if args.device is not None:
    #     config.device = args.device

    if config.seed is not None:
        random_seed(config.seed)

    device = torch.device(config.device if torch.cuda.is_available() else 'cpu') 
    main(config, gan_config, device)
