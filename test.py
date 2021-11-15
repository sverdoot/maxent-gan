import torch
import argparse

from soul_gan.distribution import GANTarget
from soul_gan.sample import soul
from soul_gan.feature import FeatureFactory
from general_utils import DotConfig, random_seed


# def load_gan(config):
#     gen.eval()
#     dis.eval()
#     return gen, dis


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('configs', type=str)
    parser.add_argument('gan_config', type=str)
    # parser.add_argument('--n_steps', type=int, default=100)
    # parser.add_argument('--n_sampling_steps', type=int, default=3)
    # parser.add_argument('--seed', type=int)
    # parser.add_argument('--device', type=int)

    args = parser.parse_args()
    return args
    

def main(config, device):
    gen, dis = load_gan(config)
    gen = gen.to(device)
    dis = dis.to(device)

    if 'sample' in config.__dict__:
        feature_class = FeatureFactory.create_feature(config.feature.name)
        feature = feature_class(config.feature.params)

        z_dim = gen.z_dim
        proposal = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(z_dim), torch.eye(z_dim))
        z = torch.randn(config.batch_size, z_dim)
        ref_dist = GANTarget(gen, dis, proposal)
        zs = soul(z, gen, ref_dist, feature, config.sample.params)

    # save, compute fid


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
