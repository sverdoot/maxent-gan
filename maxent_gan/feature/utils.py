from pathlib import Path
from typing import Union

import numpy as np

from maxent_gan.utils.callbacks import CallbackRegistry

from .eval_feature import evaluate
from .feature import BaseFeature, Feature, FeatureRegistry


def create_feature(
    config, gan, dataloader, dataset_stuff, save_dir, device
) -> Union[BaseFeature, Feature]:
    feature_callbacks = []
    callbacks = config.callbacks.feature_callbacks
    if callbacks:
        for _, callback in callbacks.items():
            params = callback.params.dict
            # HACK
            if "gan" in params:
                params["gan"] = gan
            if "save_dir" in params:
                params["save_dir"] = save_dir
            if "np_dataset" in params:
                np_dataset = np.concatenate(
                    [gan.inverse_transform(batch).numpy() for batch in dataloader], 0
                )
                params["np_dataset"] = np_dataset
            if "modes" in params:
                params["modes"] = dataset_stuff["modes"]
            feature_callbacks.append(CallbackRegistry.create(callback.name, **params))

    feature_kwargs = config.sample_params.feature.params.dict
    # HACK
    if "gan" in config.sample_params.feature.params:
        feature_kwargs["gan"] = gan
    if "dataloader" in config.sample_params.feature.params:
        feature_kwargs["dataloader"] = dataloader

    feature = FeatureRegistry.create(
        config.sample_params.feature.name,
        callbacks=feature_callbacks,
        inverse_transform=gan.inverse_transform,
        **feature_kwargs,
    )

    if config.sample_params.feature.params.ref_stats_path:
        feature.eval = True
        stats = evaluate(
            feature,
            dataset_stuff["dataset"],
            config.batch_size,
            device,
            Path(config.sample_params.feature.params.ref_stats_path),
        )
        print(stats)
        feature = FeatureRegistry.create(
            config.sample_params.feature.name,
            callbacks=feature_callbacks,
            inverse_transform=gan.inverse_transform,
            **feature_kwargs,
        )
    feature.eval = False
    return feature
