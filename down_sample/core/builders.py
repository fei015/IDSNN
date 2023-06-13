#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Model and loss construction functions."""

from down_sample.core.config import cfg
from down_sample.core.net import SoftCrossEntropyLoss
from down_sample.core.kd_losses import SoftTarget, Logits, NST, FSP, Hint, CC, PKTCosSim, RKD
from down_sample.models.anynet import AnyNet
from down_sample.models.effnet import EffNet
from down_sample.models.regnet import RegNet
from down_sample.models.resnet import ResNet

# Supported models
_models = {
    "anynet": AnyNet,
    "effnet": EffNet,
    "resnet": ResNet,
    "regnet": RegNet
}

# Supported loss functions
_loss_funs = {"cross_entropy": SoftCrossEntropyLoss}


def get_model():
    """Gets the model class specified in the config."""
    err_str = "Model type '{}' not supported"
    assert cfg.MODEL.TYPE in _models.keys(), err_str.format(cfg.MODEL.TYPE)
    return _models[cfg.MODEL.TYPE]


def get_teacher_model():
    """Gets the model class specified in the config."""
    err_str = "Model type '{}' not supported"
    assert cfg.TEACHER.MODEL.TYPE in _models.keys(), err_str.format(
        cfg.TEACHER.MODEL.TYPE)
    return _models[cfg.TEACHER.MODEL.TYPE]


def get_loss_fun():
    """Gets the loss function class specified in the config."""
    err_str = "Loss function type '{}' not supported"
    assert cfg.MODEL.LOSS_FUN in _loss_funs.keys(), err_str.format(
        cfg.TRAIN.LOSS)
    return _loss_funs[cfg.MODEL.LOSS_FUN]


def get_kd_loss_fun():
    """Gets the loss function class specified in the config."""
    # err_str = "Loss function type '{}' not supported"
    # assert cfg.TEACHER.LOSS_FUN in _loss_funs.keys(), err_str.format(
    #     cfg.TEACHER.LOSS_FUN)
    loss_func = cfg.TEACHER.LOSS_FUN
    if loss_func == 'soft_target':
        criterionKD = SoftTarget(T=cfg.TEACHER.SOFT_TEMP)
    if loss_func == 'logits':
        criterionKD = Logits()
    if loss_func == 'nst':
        criterionKD = NST()
    if loss_func == 'fsp':
        criterionKD = FSP()
    if loss_func == 'fitnet':
        criterionKD = Hint()
    if loss_func == 'cc':
        criterionKD = CC()
    if loss_func == 'pkt':
        criterionKD = PKTCosSim()
    if loss_func == 'rkd':
        criterionKD = RKD()
    return criterionKD


def build_model():
    """Builds the model."""
    return get_model()()


def build_teacher_model():
    """Builds the model."""
    return get_teacher_model()(is_teacher=True)


def build_loss_fun():
    """Build the loss function."""
    return get_loss_fun()()


def build_kd_loss_fun():
    """Build the loss function."""
    return get_kd_loss_fun()


def register_model(name, ctor):
    """Registers a model dynamically."""
    _models[name] = ctor


def register_loss_fun(name, ctor):
    """Registers a loss function dynamically."""
    _loss_funs[name] = ctor
