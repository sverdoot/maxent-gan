from pathlib import Path

import numpy as np
import studiogan
import studiogan.configs
import torch

from soul_gan.models.base import (
    BaseDiscriminator,
    BaseGenerator,
    ModelRegistry,
)

configs = Path(studiogan.configs.__path__[0])


@ModelRegistry.register()
class StudioGen(BaseGenerator):
    def __init__(self, mean, std, config, label=None):
        super().__init__(mean, std)
        cfg = studiogan.config.Configurations(Path(configs, "CIFAR10", config))
        self.n_classes = cfg.DATA.num_classes

        module = __import__(
            "models.{backbone}".format(backbone=cfg.MODEL.backbone),
            fromlist=["something"],
        )
        self.gen = module.Generator(
            z_dim=cfg.MODEL.z_dim,
            g_shared_dim=cfg.MODEL.g_shared_dim,
            img_size=cfg.DATA.img_size,
            g_conv_dim=cfg.MODEL.g_conv_dim,
            apply_attn=cfg.MODEL.apply_attn,
            attn_g_loc=cfg.MODEL.attn_g_loc,
            g_cond_mtd=cfg.MODEL.g_cond_mtd,
            num_classes=cfg.DATA.num_classes,
            g_init=cfg.MODEL.g_init,
            g_depth=cfg.MODEL.g_depth,
            mixed_precision=False,  # cfg.RUN.mixed_precision,
            MODULES=cfg.MODULES,
        )
        self.z_dim = self.gen.z_dim
        self.label = label

    def load_state_dict(self, state_dict, strict: bool = True):
        return self.gen.load_state_dict(
            state_dict["state_dict"], strict=strict
        )

    def forward(self, x):
        label = self.label
        return self.gen.forward(x, label)


@ModelRegistry.register()
class StudioDis(BaseDiscriminator):
    def __init__(self, mean, std, output_layer, config, label=None):
        super().__init__(mean, std, output_layer)
        self.config_name = config[: -len(".yaml")]
        cfg = studiogan.config.Configurations(Path(configs, "CIFAR10", config))
        self.n_classes = cfg.DATA.num_classes

        module = __import__(
            "models.{backbone}".format(backbone=cfg.MODEL.backbone),
            fromlist=["something"],
        )
        self.dis = module.Discriminator(
            img_size=cfg.DATA.img_size,
            d_conv_dim=cfg.MODEL.d_conv_dim,
            apply_d_sn=cfg.MODEL.apply_d_sn,
            apply_attn=cfg.MODEL.apply_attn,
            attn_d_loc=cfg.MODEL.attn_d_loc,
            d_cond_mtd=cfg.MODEL.d_cond_mtd,
            aux_cls_type=cfg.MODEL.aux_cls_type,
            d_embed_dim=cfg.MODEL.d_embed_dim,
            num_classes=cfg.DATA.num_classes,
            normalize_d_embed=cfg.MODEL.normalize_d_embed,
            d_init=cfg.MODEL.d_init,
            d_depth=cfg.MODEL.d_depth,
            mixed_precision=False,  # cfg.RUN.mixed_precision,
            MODULES=cfg.MODULES,
        )
        self.label = label

    def load_state_dict(self, state_dict, strict: bool = True):
        if self.config_name == "DCGAN":
            self.dis.conv = self.dis.conv1
            del self.dis.conv1
            self.dis.bn = self.dis.bn1
            del self.dis.bn1

        out = self.dis.load_state_dict(state_dict["state_dict"], strict=strict)

        if self.config_name == "DCGAN":
            self.dis.conv1 = self.dis.conv
            del self.dis.conv
            self.dis.bn1 = self.dis.bn
            del self.dis.bn

        return out

    def forward(self, x):
        label = self.label
        return self.dis.forward(x, label)["adv_output"]
