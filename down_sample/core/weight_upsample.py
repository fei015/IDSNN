#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Functions that upsample weights of small network to a bigger network."""
import torch
from torch.nn.functional import interpolate
from down_sample.core.config import cfg

import ipdb

def upsample_conv(in_weights, out_size):
    # out_weights = interpolate(in_weights.permute(2, 3, 0, 1),
    #                           (out_size[0], out_size[1]),
    #                           mode='nearest').permute(2, 3, 0, 1)
    # return out_weights
    in_size = in_weights.size()
    in_outchannel = in_size[0]
    out_outchannel = out_size[0]
    min_channel = min(in_outchannel, out_outchannel)
    in_groupwidth = in_size[1]
    out_groupwidth = out_size[1]

    out_weights = torch.empty(out_size)

    # expand in_channel dim
    sl = in_groupwidth  # small length
    bl = out_groupwidth  # big length
    count = bl // sl
    left = bl % sl
    k = 0
    for i in range(count):
        out_weights[0:min_channel,
                    k:k + sl, :, :] = in_weights[0:min_channel, :, :, :]
        k += sl
    out_weights[0:min_channel, k:k + left, :, :] = in_weights[0:min_channel,
                                                              0:left, :, :]

    # expand out_channel dim
    sl = in_outchannel  # small length
    bl = out_outchannel  # big length
    count = bl // sl
    left = bl % sl
    k = 0
    for i in range(count):
        out_weights[k:k + sl, :, :, :] = out_weights[0:sl, :, :, :]
        k += sl
    out_weights[k:k + left, :, :, :] = out_weights[0:left, :, :, :]

    if cfg.TRAIN.NORMED_UPSAMPLE:
        # in_outchannel = in_size[0]
        # out_outchannel = out_size[0]
        in_var = 2.0 / (in_size[0] * in_size[2] * in_size[3])                 # input weights are in distribution of U(0, theta)
        out_var = 2.0 / (out_size[0] * out_size[2] * out_size[3])
        k = torch.sqrt(torch.tensor(out_var / in_var))
        out_weights = out_weights * k

    return out_weights


def merge_bns(in_bns, out_channel):
    out_bn = in_bns[0]

    def merge_1d(in_tensor_lists, out_channel):
        in_tensors = torch.stack(in_tensor_lists, 0).reshape(-1)
        out_tensor = in_tensors[:out_channel]
        return out_tensor

    out_bn.update({
        key: merge_1d([bn[key] for bn in in_bns], out_channel)
        for key in out_bn.keys() if not key.endswith('num_batches_tracked')
    })
    return out_bn


def merge_convs(in_weight_lists, out_size):
    mode = cfg.TRAIN.MERGE_MODE
    if mode == 1:
        weights = torch.stack(in_weight_lists, 0)
        flat_weights = weights.reshape(-1, out_size[2], out_size[3])
        if out_size[0] * out_size[1] <= flat_weights.shape[0]:
            picked_flat_weights = flat_weights[:out_size[0] * out_size[1]]
            picked_weights = picked_flat_weights.reshape(out_size)
        else:
            left = flat_weights.shape[0] % out_size[1]
            if left != 0:
                picked_id = flat_weights.shape[0] - left
                flat_weights = flat_weights[:picked_id, :, :]
            picked_weights = flat_weights.reshape(-1, out_size[1], out_size[2], out_size[3])
    elif mode == 2:
        weights = torch.cat(in_weight_lists, 1)
        out_weights0 = weights[:, :out_size[1], :, :]
        in_channel_per = out_weights0.shape[1]
        out_channel_per = out_weights0.shape[0]
        iters_out = out_size[0] // out_channel_per + 1
        iters_in = weights.shape[1] // in_channel_per
        iters = min(iters_in, iters_out)
        for i in range(1, iters):
            out_weight_i = weights[:, in_channel_per * i: in_channel_per * (i+1), :, :]
            out_weights0 = torch.cat([out_weights0, out_weight_i], 0)
        if out_weights0.shape[0] > out_size[0]:
            out_weights0 = out_weights0[:out_size[0]]
        picked_weights = out_weights0
    elif mode == 3:
        weights = torch.cat(in_weight_lists, 0)
        out_weights0 = weights[:out_size[0], :, :, :]
        in_channel_per = out_weights0.shape[1]
        out_channel_per = out_weights0.shape[0]
        iters_out = weights.shape[0] // out_channel_per
        iters_in = out_size[1] // in_channel_per + 1
        iters = min(iters_in, iters_out)
        for i in range(1, iters):
            out_weight_i = weights[out_channel_per * i: out_channel_per * (i+1), :, :, :]
            out_weights0 = torch.cat([out_weights0, out_weight_i], 1)
        if out_weights0.shape[1] > out_size[1]:
            out_weights0 = out_weights0[:, :out_size[1], :, :]
        picked_weights = out_weights0
    return picked_weights


def upsample_wconv(in_weights, Win, Wout, out_size):
    in_size = in_weights.size()
    in_outchannel = in_size[0]
    out_outchannel = out_size[0]
    min_channel = min(in_outchannel, out_outchannel)
    in_groupwidth = in_size[1]
    out_groupwidth = out_size[1]

    out_weights = torch.empty(out_size)
    out_Win = torch.empty([out_size[1], 1]).float()
    out_Wout = torch.empty([1, out_size[0], 1, 1]).float()
    # expand in_channel dim
    sl = in_groupwidth  # small length
    bl = out_groupwidth  # big length
    count = bl // sl
    left = bl % sl
    k = 0
    for i in range(count):
        out_weights[0:min_channel,
                    k:k + sl, :, :] = in_weights[0:min_channel, :, :, :]
        out_Win[k:k + sl, :] = Win[:, :]
        k += sl
    out_weights[0:min_channel, k:k + left, :, :] = in_weights[0:min_channel,
                                                              0:left, :, :]
    out_Win[k:k+left, :] = Win[0:left, :]
    # expand out_channel dim
    sl = in_outchannel  # small length
    bl = out_outchannel  # big length
    count = bl // sl
    left = bl % sl
    k = 0
    for i in range(count):
        out_weights[k:k + sl, :, :, :] = out_weights[0:sl, :, :, :]
        out_Wout[:, k:k + sl, :, :] = out_Wout[:, 0:sl, :, :]
        k += sl
    out_weights[k:k + left, :, :, :] = out_weights[0:left, :, :, :]
    out_Wout[:, k:k + left, :, :] = out_Wout[:, 0:left, :, :]

    if cfg.TRAIN.NORMED_UPSAMPLE:
        # in_outchannel = in_size[0]
        # out_outchannel = out_size[0]
        in_var = 2.0 / (in_size[0] * in_size[2] * in_size[3])                 # input weights are in distribution of U(0, theta)
        out_var = 2.0 / (out_size[0] * out_size[2] * out_size[3])
        k = torch.sqrt(torch.tensor(out_var / in_var))
        out_weights = out_weights * k

    return out_weights

def expand_1d(in_tensor, out_channel):
    out_tensor = torch.empty(out_channel)
    sl = in_tensor.size()[0]  # small length
    bl = out_channel  # big length
    count = bl // sl
    left = bl % sl
    k = 0
    for i in range(count):
        out_tensor[k:k + sl] = in_tensor[0:sl]
        k += sl
    out_tensor[k:k + left] = in_tensor[0:left]
    return out_tensor


def upsample_fc(in_weights, in_bias, out_channel, out_class):

    # TODO:
    # in_weights = in_weights.view(1, 1,
    #                              in_weights.size()[0],
    #                              in_weights.size()[1])
    # in_bias = in_bias.view(1, 1, in_bias.size()[0])

    # # import ipdb;ipdb.set_trace();print()
    # out_weights = interpolate(in_weights,
    #                           size=(out_class, out_channel),
    #                           mode='nearest').squeeze()

    # out_bias = interpolate(in_bias, size=(out_class, ),
    #                        mode='nearest').squeeze()
    # return out_weights, out_bias

    out_weights = torch.empty(out_class, out_channel)
    sl = in_weights.shape[1]  # small length
    bl = out_channel  # big length
    count = bl // sl
    left = bl % sl
    k = 0
    for i in range(count):
        out_weights[:,
                    k:k + sl] = in_weights[:, :]
        k += sl
    out_weights[:, k:k + left] = in_weights[:, 0:left]

    return out_weights, in_bias


def upsample_bn(in_bn, out_channel):
    out_bn = in_bn

    def expand_1d(in_tensor, out_channel):
        # ipdb.set_trace()
        # TODO: 
        # in_tensor = in_tensor.view(1, 1, in_tensor.size()[0])
        # out_tensor = interpolate(in_tensor, size=(out_channel, ), mode='nearest').squeeze()
        # return out_tensor

        out_tensor = torch.empty(out_channel)
        sl = in_tensor.size()[0]  # small length
        bl = out_channel  # big length
        count = bl // sl
        left = bl % sl
        k = 0
        for i in range(count):
            out_tensor[k:k + sl] = in_tensor[0:sl]
            k += sl
        out_tensor[k:k + left] = in_tensor[0:left]
        return out_tensor
    out_bn.update({
        key: expand_1d(item, out_channel)
        for key, item in in_bn.items()
        if not key.endswith('num_batches_tracked')
    })
    return out_bn
