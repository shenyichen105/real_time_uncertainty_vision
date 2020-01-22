import logging
import functools
import torch

from ptsemseg.loss.loss import (
    cross_entropy2d,
    bootstrapped_cross_entropy2d,
    multi_scale_cross_entropy2d,
    nll_laplace_2d,
    nll_gaussian_2d,
    logit_normal_loss,
)


logger = logging.getLogger("ptsemseg")

key2loss = {
    "cross_entropy": cross_entropy2d,
    "bootstrapped_cross_entropy": bootstrapped_cross_entropy2d,
    "multi_scale_cross_entropy": multi_scale_cross_entropy2d,
    "nll_guassian_loss": nll_gaussian_2d,
    "nll_laplace_loss": nll_laplace_2d,
    "logit_normal_loss": logit_normal_loss
}


def get_loss_function(cfg):
    if cfg["training"]["loss"] is None:
        logger.info("Using default cross entropy loss")
        return cross_entropy2d

    else:
        loss_dict = cfg["training"]["loss"]
        #print(loss_dict)
        loss_name = loss_dict["name"]
        loss_params = {k: v for k, v in loss_dict.items() if k != "name"}
        if "weight" in loss_params:
            loss_params["weight"] = torch.tensor(loss_params["weight"]).cuda()

        if loss_name not in key2loss:
            raise NotImplementedError("Loss {} not implemented".format(loss_name))

        logger.info("Using {} with {} params".format(loss_name, loss_params))
        return functools.partial(key2loss[loss_name], **loss_params)

def get_soft_loss_function(cfg):
    if cfg["training"]["soft_loss"] is None:
        logger.info("Using default nll guassian loss")
        return nll_gaussian_2d

    else:
        loss_dict = cfg["training"]["soft_loss"]
        #print(loss_dict)
        loss_name = loss_dict["name"]
        loss_params = {k: v for k, v in loss_dict.items() if k != "name"}
        if loss_name not in key2loss:
            raise NotImplementedError("Loss {} not implemented".format(loss_name))

        logger.info("Using {} with {} params".format(loss_name, loss_params))
        return functools.partial(key2loss[loss_name], **loss_params)
    