from abc import ABC, abstractmethod
from functools import wraps
from typing import Callable

import numpy as np
import torch
import torchvision


# from main.nnclass import CNN
# from main.util.save_util import save_image_general, save_weight_vgg_feature


class AvgHolder(object):
    cnt : int = 0
    
    def __init__(self, init_val):
        self.val = init_val
    
    def upd(self, new_val):
        self.cnt += 1
        alpha = 1. / self.cnt
        if isinstance(self.val, list):
            for i in range(len(self.val)):
                self.val[i] = self.val[i] * (1. - alpha) + new_val[i]
        else:
            self.val = self.val * (1. - alpha) + new_val

    def reset(self):
        if isinstance(self.val, list):
            self.val = [0] * len(self.val)
        else:
            self.val = 0
        
    def data(self):
        return self.val


class Feature(ABC):
    def __init__(self):
        self.avg_weight = AvgHolder(0)
        self.avg_feature = AvgHolder(0)

    def log_prob(self, out):
        lik_f = 0
        out_lik = out.copy()
        for l in range(len(out_lik)):
            lik_f += torch.dot(self.weight[l], out_lik[l])
        lik_f *= -1
        return lik_f

    def weight_up(self, out, step):
        for i in range(len(self.weight)):
            self.weight[i] += + step * (out[i] - self.ref_feature[i])

    @staticmethod
    def average_feature(feature_method):
        @wraps
        def with_avg(self, *args, **kwargs):
            out = feature_method(self, *args, **kwargs)
            self.avg_feature.upd(out)
            return out
        return with_avg


class FeatureFactory:
    registry = {}

    @classmethod
    def register(cls, name: str) -> Callable:

        def inner_wrapper(wrapped_class: Feature) -> Callable:
            #if name in cls.registry:
                # logger.warning('Executor %s already exists. Will replace it', name)
            cls.registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def create_feature(cls, name: str, **kwargs) -> Feature:
        exec_class = cls.registry[name]
        executor = exec_class(**kwargs)
        return executor


@FeatureFactory.register('inception_score')
class InceptionScoreFeature(Feature):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = torchvision.models.inception_v3(pretrained=True)
        self.ref_feature = kwargs.get('ref_feature', 11.5)
    
    @Feature.average_feature
    def __call__(self, x):
        pis = torch.softmax(self.model(x), -1)
        avg_pis = pis.mean(0)
        score = None
        return score

    
class CNNFeature(Feature):
    """
    Construct CNN feature methods.

    Attributes
    ----------
    name : str
        Name of the class, here 'CNNFeature'.
    layer : tuple
        A tuple corresponding to the layers to be considered in the log-likelihood.
    nb_layer : int
        The number of layers, corresponding to len(self.layer).
    model_cnn : str
        Name of the neural network architecture to be used.
    network : instance of a class
        The instance of the class is given by CNN, see nnclass.py for more details.
    beta : float
        The features are divided by this value.
        Be careful, the algorithm is sensitive with respect to this value.
    forward : a function
        A function taking a tensor as an input and returning the tensor of the output at given layers.
        See self.forward in nnclass for more details.
    x0 : torch.tensor
        4d torch tensor, the examplar image.
        First dimension is one.
        Second and third dimension correspond to the width and the height of the image.
        Fourth dimension is set to 3.
    feature_init : list
        A list of length self.nb_layer. Each element of feature_init is a tensor.
    weight : list
        Same structure as feature_init and contains weight to compute log-likelihood.
    regularization : float
        Inverse of the variance in the a priori white noise (or regularization parameter).

    Methods
    -------
    feature_fun(x)
        Compute features of x.
    log_likelihood(out, x)
        Compute log-likelihood associated with x and features out.
    weight_up(weight, out, step, weight_old, n_it)
        Update weight.
    weight_avg(weight, step, weight_mv_avg, sum_step)
        Update moving average on weights.
    save_image(params, x, x_grad, n_it, folder_name, im_number=None)
        Save images.
    save_weight(params, feat, w, w_avg, n_it, n_epoch, folder_name)
        Save weight statistics.
    """

    def __init__(self, cnn, params):
        super().__init__()
        self.layer = params['layer']
        self.nb_layer = len(params['layer'])
        self.model_cnn = params['model_cnn']
        params_CNN = dict()
        params_CNN['layers'] = params['layer']
        params_CNN['model_str'] = params['model_cnn']
        self.beta = params['beta']
        self.cnn = cnn(params_CNN)
        self.x0 = params['x0']
        self.ref_feature = self(params['x_real'])
        self.weight = params['weight']

    @Feature.average_feature
    def __call__(self, x):
        feat = self.cnn(x)
        x0 = self.x0
        out = []
        size_x = x.shape[-2] * x.shape[-1]
        size_x0 = x0.shape[-2] * x0.shape[-1]
        ratio_size = size_x0 / size_x
        for feat_l in feat:
            ratio_layer = feat_l.shape[-2] * feat_l.shape[-1]
            ratio_layer = size_x / ratio_layer
            ratio = ratio_layer * ratio_size / self.beta
            out.append(feat_l[0].sum((1, 2)) * ratio)
        return out

    # def save_image(self, params, x, x_grad, n_it, folder_name, im_number=None):

    #     save_image_general(params, x, x_grad, n_it,
    #                        folder_name, im_number)

    # def save_weight(self, params, feat, w, w_avg, n_it, n_epoch, fd_name):
    #     save_weight_vgg_feature(self, params, feat, w,
    #                             w_avg, n_it, n_epoch, fd_name)
