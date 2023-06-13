#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Functions that handle saving and loading of checkpoints."""

import os

import down_sample.core.distributed as dist
import torch
from down_sample.core.config import cfg
from down_sample.core.io import pathmgr
from down_sample.core.net import unwrap_model
from down_sample.core.weight_upsample import upsample_conv, upsample_fc, upsample_bn, upsample_wconv, merge_convs, merge_bns

# Common prefix for checkpoint file names
_NAME_PREFIX = "model_epoch_"

# Checkpoints directory name
_DIR_NAME = "checkpoints"


def get_checkpoint_dir():
    """Retrieves the location for storing checkpoints."""
    return os.path.join(cfg.OUT_DIR, _DIR_NAME)


def get_checkpoint(epoch):
    """Retrieves the path to a checkpoint file."""
    name = "{}{:04d}.pyth".format(_NAME_PREFIX, epoch)
    return os.path.join(get_checkpoint_dir(), name)


def get_checkpoint_best():
    """Retrieves the path to the best checkpoint file."""
    return os.path.join(cfg.OUT_DIR, "model.pyth")


def get_last_checkpoint():
    """Retrieves the most recent checkpoint (highest epoch number)."""
    checkpoint_dir = get_checkpoint_dir()
    checkpoints = [f for f in pathmgr.ls(checkpoint_dir) if _NAME_PREFIX in f]
    last_checkpoint_name = sorted(checkpoints)[-1]
    return os.path.join(checkpoint_dir, last_checkpoint_name)


def has_checkpoint():
    """Determines if there are checkpoints available."""
    checkpoint_dir = get_checkpoint_dir()
    if not pathmgr.exists(checkpoint_dir):
        return False
    return any(_NAME_PREFIX in f for f in pathmgr.ls(checkpoint_dir))


def save_checkpoint(model, optimizer, epoch, best):
    """Saves a checkpoint."""
    # Save checkpoints only from the master process
    if not dist.is_master_proc():
        return
    # Ensure that the checkpoint dir exists
    pathmgr.mkdirs(get_checkpoint_dir())
    # Record the state
    checkpoint = {
        "epoch": epoch,
        "model_state": unwrap_model(model).state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "cfg": cfg.dump(),
    }
    # Write the checkpoint
    checkpoint_file = get_checkpoint(epoch + 1)
    with pathmgr.open(checkpoint_file, "wb") as f:
        torch.save(checkpoint, f)
    # If best copy checkpoint to the best checkpoint
    if best:
        pathmgr.copy(checkpoint_file, get_checkpoint_best(), overwrite=True)
    return checkpoint_file


def load_checkpoint(checkpoint_file, model, optimizer=None):
    """Loads the checkpoint from the given file."""
    err_str = "Checkpoint '{}' not found"
    assert pathmgr.exists(checkpoint_file), err_str.format(checkpoint_file)
    with pathmgr.open(checkpoint_file, "rb") as f:
        checkpoint = torch.load(f, map_location="cpu")
    unwrap_model(model).load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(
        checkpoint["optimizer_state"]) if optimizer else ()
    return checkpoint["epoch"]


def delete_checkpoints(checkpoint_dir=None, keep="all"):
    """Deletes unneeded checkpoints, keep can be "all", "last", or "none"."""
    assert keep in ["all", "last",
                    "none"], "Invalid keep setting: {}".format(keep)
    checkpoint_dir = checkpoint_dir if checkpoint_dir else get_checkpoint_dir()
    if keep == "all" or not pathmgr.exists(checkpoint_dir):
        return 0
    checkpoints = [f for f in pathmgr.ls(checkpoint_dir) if _NAME_PREFIX in f]
    checkpoints = sorted(checkpoints)[:-1] if keep == "last" else checkpoints
    for checkpoint in checkpoints:
        pathmgr.rm(os.path.join(checkpoint_dir, checkpoint))
    return len(checkpoints)


def upsample_load_checkpoint(checkpoint_file, model, optimizer=None):
    """Load and upsample the checkpoint from the given file to a possibly
    bigger model. Currently only support regnet"""
    err_str = "Checkpoint '{}' not found"
    weighted = cfg.TRAIN.UPSAMPLE_WCONV
    # if cfg.REGNET.BLOCK_TYPE == 'weighted_res_bottleneck_block':
    #     weighted = True

    assert pathmgr.exists(checkpoint_file), err_str.format(checkpoint_file)
    with pathmgr.open(checkpoint_file, "rb") as f:
        checkpoint = torch.load(f, map_location="cpu")
    state_dict = checkpoint["model_state"]

    target_state_dict = unwrap_model(model).state_dict().copy()
    stems = ['stem']
    stages = ['s1', 's2', 's3', 's4']
    heads = []  # ['head.fc']
    basic_bn_keys = [
        'weight', 'bias', 'running_var', 'running_mean', 'num_batches_tracked'
    ]
    upsampled_keys = list()
    changed_target_keys = list()

    def upsample_bn_byprefix(prefix, target_prefix=None):
        if target_prefix is None:
            target_prefix = prefix
        bn_keys = [prefix + key for key in basic_bn_keys]
        target_bn_keys = [target_prefix + key for key in basic_bn_keys]
        bn = {key: state_dict[key] for key in bn_keys}
        out_channel = target_state_dict[target_prefix + 'weight'].size()[0]
        bn = upsample_bn(bn, out_channel)
        target_bn = {
            target_key: bn[key]
            for key, target_key in zip(bn_keys, target_bn_keys)
        }
        target_state_dict.update(target_bn)
        # record upsampled keys and target keys
        upsampled_keys.append(prefix)
        changed_target_keys.append(target_prefix)

    def upsample_conv_byprefix(prefix, target_prefix=None):
        if not cfg.TRAIN.LOAD_CONV:
            return
        if target_prefix is None:
            target_prefix = prefix
        target_state_dict[target_prefix + 'weight'] = upsample_conv(
            state_dict[prefix + 'weight'],
            target_state_dict[target_prefix + 'weight'].size())
        # record upsampled keys and target keys
        upsampled_keys.append(prefix)
        changed_target_keys.append(target_prefix)

    def upsample_weighted_conv_byprefix(prefix, target_prefix=None):
        if not cfg.TRAIN.LOAD_CONV:
            return
        if target_prefix is None:
            target_prefix = prefix
        target_state_dict[target_prefix + 'weight'] = upsample_wconv(
            state_dict[prefix + 'weight'], state_dict[prefix + 'Win'],
            state_dict[prefix + 'Wout'],
            target_state_dict[target_prefix + 'weight'].size())
        # record upsampled keys and target keys
        upsampled_keys.append(prefix)
        changed_target_keys.append(target_prefix)

    def upsample_fc_byprefix(prefix, target_prefix=None):
        if target_prefix is None:
            target_prefix = prefix
        weight, bias = upsample_fc(
            state_dict[prefix + 'weight'], state_dict[prefix + 'bias'],
            target_state_dict[target_prefix + 'weight'].size()[1],
            target_state_dict[target_prefix + 'weight'].size()[0])
        target_state_dict[target_prefix + 'weight'] = weight
        target_state_dict[target_prefix + 'bias'] = bias
        # record upsampled keys and target keys
        upsampled_keys.append(prefix)
        changed_target_keys.append(target_prefix)

    def read_stageinfo(keys, stages):
        depth = [0 for _ in range(len(stages))]
        for i in range(len(depth)):
            stage_keys = [
                int(key.split('.')[1][1:]) for key in keys
                if key.startswith(stages[i])
            ]
            depth[i] = max(stage_keys)
        return depth

    depth = read_stageinfo(state_dict.keys(), stages)
    depth_target = read_stageinfo(target_state_dict.keys(), stages)

    # upsample and load stem
    for stem in stems:
        upsample_conv_byprefix(stem + '.conv.')
        if cfg.TRAIN.BN_MODE >= 2:
            upsample_bn_byprefix(stem + '.bn.')

    # upsample and load body
    for s, d, dt in zip(stages, depth, depth_target):
        # upsample the blocks
        for j in range(1, dt + 1):
            # upsample the last stage if no such stages
            block_num = j if j < d else d
            block_prefix = s + '.b' + str(block_num)
            target_block_prefix = s + '.b' + str(j)
            proj_conv = block_prefix + '.proj.'
            # upsample projection conv layer
            if (proj_conv + 'weight' in target_state_dict.keys()
                    and proj_conv + 'weight' in state_dict.keys()):
                if weighted:
                    upsample_weighted_conv_byprefix(proj_conv)
                else:
                    upsample_conv_byprefix(proj_conv)
                proj_bn_prefix = block_prefix + '.bn.'
                if cfg.TRAIN.BN_MODE >= 3:
                    upsample_bn_byprefix(proj_bn_prefix)
            block_conv_prefixs = [
                block_prefix + '.' + key for key in ['f.a.', 'f.b.', 'f.c.']
            ]
            target_block_conv_prefixs = [
                target_block_prefix + '.' + key
                for key in ['f.a.', 'f.b.', 'f.c.']
            ]
            # upsample the convs
            for prefix, target_prefix in zip(block_conv_prefixs,
                                             target_block_conv_prefixs):
                if weighted:
                    upsample_weighted_conv_byprefix(prefix, target_prefix)
                else:
                    upsample_conv_byprefix(prefix, target_prefix)
            # upsample the bns
            block_bn_prefixs = [
                block_prefix + '.' + key
                for key in ['f.a_bn.', 'f.b_bn.', 'f.c_bn.']
            ]
            target_block_bn_prefixs = [
                target_block_prefix + '.' + key
                for key in ['f.a_bn.', 'f.b_bn.', 'f.c_bn.']
            ]
            for prefix, target_prefix in zip(block_bn_prefixs,
                                             target_block_bn_prefixs):
                if cfg.TRAIN.BN_MODE >= 4:
                    upsample_bn_byprefix(prefix, target_prefix)

    # upsample and load head
    for head in heads:
        upsample_fc_byprefix(head + '.')
    upsample_projection = {
        key: item
        for key, item in zip(changed_target_keys, upsampled_keys)
    }
    print("weights upsampling projection:")
    print(upsample_projection)
    unwrap_model(model).load_state_dict(target_state_dict)
    optimizer.load_state_dict(
        checkpoint["optimizer_state"]) if optimizer else ()
    return 0


def upsample_load_from_list(checkpoint_list, model, optimizer=None):
    """Load and upsample the checkpoint from the given file to a possibly
    bigger model. Currently only support regnet"""
    err_str = "Checkpoint '{}' not found"
    weighted = cfg.TRAIN.UPSAMPLE_WCONV
    # if cfg.REGNET.BLOCK_TYPE == 'weighted_res_bottleneck_block':
    #     weighted = True
    err_str = "Checkpoint '{}' not found"
    state_dicts = []
    for checkpoint_file in checkpoint_list:
        assert pathmgr.exists(checkpoint_file), err_str.format(checkpoint_file)
        with pathmgr.open(checkpoint_file, "rb") as f:
            checkpoint = torch.load(f, map_location="cpu")
        state_dict = checkpoint["model_state"]
        state_dicts.append(state_dict)
    state_dict = state_dicts[0]
    target_state_dict = unwrap_model(model).state_dict().copy()
    stems = ['stem']
    stages = ['s1', 's2', 's3', 's4']
    heads = []  # ['head.fc']
    basic_bn_keys = [
        'weight', 'bias', 'running_var', 'running_mean', 'num_batches_tracked'
    ]

    upsampled_keys = list()
    changed_target_keys = list()

    def upsample_bn_byprefix(prefix, target_prefix=None):
        if target_prefix is None:
            target_prefix = prefix
        bn_keys = [prefix + key for key in basic_bn_keys]
        target_bn_keys = [target_prefix + key for key in basic_bn_keys]
        # bn = {key: state_dict[key] for key in bn_keys}
        out_channel = target_state_dict[target_prefix + 'weight'].size()[0]
        bns = [{key: st[key] for key in bn_keys} for st in state_dicts]
        bn = merge_bns(bns, out_channel)
        bn = upsample_bn(bn, out_channel)
        target_bn = {
            target_key: bn[key]
            for key, target_key in zip(bn_keys, target_bn_keys)
        }
        target_state_dict.update(target_bn)
        # record upsampled keys and target keys
        upsampled_keys.append(prefix)
        changed_target_keys.append(target_prefix)

    def upsample_conv_byprefix(prefix, target_prefix=None):
        if not cfg.TRAIN.LOAD_CONV:
            return
        if target_prefix is None:
            target_prefix = prefix
        input_weights = [st[prefix + 'weight'] for st in state_dicts]
        merged_weight = merge_convs(input_weights, target_state_dict[target_prefix + 'weight'].size())
        target_state_dict[target_prefix + 'weight'] = upsample_conv(
            merged_weight,
            target_state_dict[target_prefix + 'weight'].size())
        # record upsampled keys and target keys
        upsampled_keys.append(prefix)
        changed_target_keys.append(target_prefix)

    def upsample_weighted_conv_byprefix(prefix, target_prefix=None):
        if not cfg.TRAIN.LOAD_CONV:
            return
        if target_prefix is None:
            target_prefix = prefix
        target_state_dict[target_prefix + 'weight'] = upsample_wconv(
            state_dict[prefix + 'weight'], state_dict[prefix + 'Win'],
            state_dict[prefix + 'Wout'],
            target_state_dict[target_prefix + 'weight'].size())
        # record upsampled keys and target keys
        upsampled_keys.append(prefix)
        changed_target_keys.append(target_prefix)

    def upsample_fc_byprefix(prefix, target_prefix=None):
        if target_prefix is None:
            target_prefix = prefix
        weight, bias = upsample_fc(
            state_dict[prefix + 'weight'], state_dict[prefix + 'bias'],
            target_state_dict[target_prefix + 'weight'].size()[1],
            target_state_dict[target_prefix + 'weight'].size()[0])
        target_state_dict[target_prefix + 'weight'] = weight
        target_state_dict[target_prefix + 'bias'] = bias
        # record upsampled keys and target keys
        upsampled_keys.append(prefix)
        changed_target_keys.append(target_prefix)

    def read_stageinfo(keys, stages):
        depth = [0 for _ in range(len(stages))]
        for i in range(len(depth)):
            stage_keys = [
                int(key.split('.')[1][1:]) for key in keys
                if key.startswith(stages[i])
            ]
            depth[i] = max(stage_keys)
        return depth

    depth = read_stageinfo(state_dict.keys(), stages)
    depth_target = read_stageinfo(target_state_dict.keys(), stages)

    # upsample and load stem
    for stem in stems:
        upsample_conv_byprefix(stem + '.conv.')
        if cfg.TRAIN.BN_MODE >= 2:
            upsample_bn_byprefix(stem + '.bn.')

    # upsample and load body
    for s, d, dt in zip(stages, depth, depth_target):
        # upsample the blocks
        for j in range(1, dt + 1):
            # upsample the last stage if no such stages
            block_num = j if j < d else d
            block_prefix = s + '.b' + str(block_num)
            target_block_prefix = s + '.b' + str(j)
            proj_conv = block_prefix + '.proj.'
            # upsample projection conv layer
            if (proj_conv + 'weight' in target_state_dict.keys()
                    and proj_conv + 'weight' in state_dict.keys()):
                if weighted:
                    upsample_weighted_conv_byprefix(proj_conv)
                else:
                    upsample_conv_byprefix(proj_conv)
                proj_bn_prefix = block_prefix + '.bn.'
                if cfg.TRAIN.BN_MODE >= 3:
                    upsample_bn_byprefix(proj_bn_prefix)
            block_conv_prefixs = [
                block_prefix + '.' + key for key in ['f.a.', 'f.b.', 'f.c.']
            ]
            target_block_conv_prefixs = [
                target_block_prefix + '.' + key
                for key in ['f.a.', 'f.b.', 'f.c.']
            ]
            # upsample the convs
            for prefix, target_prefix in zip(block_conv_prefixs,
                                             target_block_conv_prefixs):
                if weighted:
                    upsample_weighted_conv_byprefix(prefix, target_prefix)
                else:
                    upsample_conv_byprefix(prefix, target_prefix)
            # upsample the bns
            block_bn_prefixs = [
                block_prefix + '.' + key
                for key in ['f.a_bn.', 'f.b_bn.', 'f.c_bn.']
            ]
            target_block_bn_prefixs = [
                target_block_prefix + '.' + key
                for key in ['f.a_bn.', 'f.b_bn.', 'f.c_bn.']
            ]
            for prefix, target_prefix in zip(block_bn_prefixs,
                                             target_block_bn_prefixs):
                if cfg.TRAIN.BN_MODE >= 4:
                    upsample_bn_byprefix(prefix, target_prefix)

    # upsample and load head
    for head in heads:
        upsample_fc_byprefix(head + '.')
    upsample_projection = {
        key: item
        for key, item in zip(changed_target_keys, upsampled_keys)
    }
    print("weights upsampling projection:")
    print(upsample_projection)
    unwrap_model(model).load_state_dict(target_state_dict)
    optimizer.load_state_dict(
        checkpoint["optimizer_state"]) if optimizer else ()
    return 0


def downsample_load_resnet(checkpoint_file, model, optimizer=None):
    """Load and upsample the checkpoint from the given file to a possibly
    bigger model. Currently only support regnet"""
    err_str = "Checkpoint '{}' not found"
    # if cfg.REGNET.BLOCK_TYPE == 'weighted_res_bottleneck_block':
    #     weighted = True

    assert pathmgr.exists(checkpoint_file), err_str.format(checkpoint_file)
    with pathmgr.open(checkpoint_file, "rb") as f:
        checkpoint = torch.load(f, map_location="cpu")
    state_dict = checkpoint["model_state"]

    target_state_dict = unwrap_model(model).state_dict().copy()

    stems = ['stem']
    stages = ['s1', 's2', 's3', 's4']
    heads = []  # ['head.fc']
    basic_bn_keys = [
        'weight', 'bias', 'running_var', 'running_mean', 'num_batches_tracked'
    ]
    upsampled_keys = list()
    changed_target_keys = list()

    def upsample_bn_byprefix(prefix, target_prefix=None):
        if target_prefix is None:
            target_prefix = prefix
        bn_keys = [prefix + key for key in basic_bn_keys]
        target_bn_keys = [target_prefix + key for key in basic_bn_keys]
        bn = {key: state_dict[key] for key in bn_keys}
        out_channel = target_state_dict[target_prefix + 'weight'].size()[0]
        bn = upsample_bn(bn, out_channel)
        target_bn = {
            target_key: bn[key]
            for key, target_key in zip(bn_keys, target_bn_keys)
        }
        target_state_dict.update(target_bn)
        # record upsampled keys and target keys
        upsampled_keys.append(prefix)
        changed_target_keys.append(target_prefix)

    def upsample_conv_byprefix(prefix, target_prefix=None):
        if not cfg.TRAIN.LOAD_CONV:
            return
        if target_prefix is None:
            target_prefix = prefix
        target_state_dict[target_prefix + 'weight'] = upsample_conv(
            state_dict[prefix + 'weight'],
            target_state_dict[target_prefix + 'weight'].size())
        # record upsampled keys and target keys
        upsampled_keys.append(prefix)
        changed_target_keys.append(target_prefix)

    def upsample_fc_byprefix(prefix, target_prefix=None):
        if target_prefix is None:
            target_prefix = prefix
        weight, bias = upsample_fc(
            state_dict[prefix + 'weight'], state_dict[prefix + 'bias'],
            target_state_dict[target_prefix + 'weight'].size()[1],
            target_state_dict[target_prefix + 'weight'].size()[0])
        target_state_dict[target_prefix + 'weight'] = weight
        target_state_dict[target_prefix + 'bias'] = bias
        # record upsampled keys and target keys
        upsampled_keys.append(prefix)
        changed_target_keys.append(target_prefix)

    def read_stageinfo(keys, stages):
        depth = [0 for _ in range(len(stages))]
        for i in range(len(depth)):
            stage_keys = [
                int(key.split('.')[1][1:]) for key in keys
                if key.startswith(stages[i])
            ]
            depth[i] = max(stage_keys)
        return depth

    depth = read_stageinfo(state_dict.keys(), stages)
    depth_target = read_stageinfo(target_state_dict.keys(), stages)

    # upsample and load stem
    for stem in stems:
        upsample_conv_byprefix(stem + '.conv.')
        if cfg.TRAIN.BN_MODE >= 2:
            upsample_bn_byprefix(stem + '.bn.')

    # upsample and load body
    for s, d, dt in zip(stages, depth, depth_target):
        # upsample the blocks
        for j in range(1, dt + 1):
            # upsample the last stage if no such stages
            block_num = j if j < d else d
            block_prefix = s + '.b' + str(block_num)
            target_block_prefix = s + '.b' + str(j)
            proj_conv = block_prefix + '.proj.'
            # upsample projection conv layer
            if (proj_conv + 'weight' in target_state_dict.keys()
                    and proj_conv + 'weight' in state_dict.keys()):
                upsample_conv_byprefix(proj_conv)
                proj_bn_prefix = block_prefix + '.bn.'
                if cfg.TRAIN.BN_MODE >= 3:
                    upsample_bn_byprefix(proj_bn_prefix)
            block_conv_prefixs = [
                block_prefix + '.' + key for key in ['f.a.', 'f.b.']
            ]
            target_block_conv_prefixs = [
                target_block_prefix + '.' + key
                for key in ['f.a.', 'f.b.']
            ]
            # upsample the convs
            for prefix, target_prefix in zip(block_conv_prefixs,
                                             target_block_conv_prefixs):
                upsample_conv_byprefix(prefix, target_prefix)
            # upsample the bns
            block_bn_prefixs = [
                block_prefix + '.' + key
                for key in ['f.a_bn.', 'f.b_bn.']
            ]
            target_block_bn_prefixs = [
                target_block_prefix + '.' + key
                for key in ['f.a_bn.', 'f.b_bn.']
            ]
            for prefix, target_prefix in zip(block_bn_prefixs,
                                             target_block_bn_prefixs):
                if cfg.TRAIN.BN_MODE >= 4:
                    upsample_bn_byprefix(prefix, target_prefix)

    # upsample and load head
    for head in heads:
        upsample_fc_byprefix(head + '.')
    ds_projection = {
        key: item
        for key, item in zip(changed_target_keys, upsampled_keys)
    }
    print("weights downsampling projection:")
    print(ds_projection)
    unwrap_model(model).load_state_dict(target_state_dict)
    optimizer.load_state_dict(
        checkpoint["optimizer_state"]) if optimizer else ()
    return 0


def downsample_load_checkpoint(checkpoint_file, model, optimizer=None):
    """Load and upsample the checkpoint from the given file to a possibly
    bigger model. Currently only support regnet"""
    err_str = "Checkpoint '{}' not found"
    weighted = cfg.TRAIN.UPSAMPLE_WCONV
    # if cfg.REGNET.BLOCK_TYPE == 'weighted_res_bottleneck_block':
    #     weighted = True

    assert pathmgr.exists(checkpoint_file), err_str.format(checkpoint_file)
    with pathmgr.open(checkpoint_file, "rb") as f:
        checkpoint = torch.load(f, map_location="cpu")
    state_dict = checkpoint["model_state"]

    target_state_dict = unwrap_model(model).state_dict().copy()
    stems = ['stem']
    # stems = []
    stages = ['s1', 's2', 's3', 's4']
    heads = []  # ['head.fc']
    # heads = ['head.fc']
    basic_bn_keys = [
        'weight', 'bias', 'running_var', 'running_mean', 'num_batches_tracked'
    ]
    upsampled_keys = list()
    changed_target_keys = list()

    def upsample_bn_byprefix(prefix, target_prefix=None):
        if target_prefix is None:
            target_prefix = prefix
        bn_keys = [prefix + key for key in basic_bn_keys]
        target_bn_keys = [target_prefix + key for key in basic_bn_keys]
        bn = {key: state_dict[key] for key in bn_keys}
        out_channel = target_state_dict[target_prefix + 'weight'].size()[0]
        bn = upsample_bn(bn, out_channel)
        target_bn = {
            target_key: bn[key]
            for key, target_key in zip(bn_keys, target_bn_keys)
        }
        target_state_dict.update(target_bn)
        # record upsampled keys and target keys
        upsampled_keys.append(prefix)
        changed_target_keys.append(target_prefix)

    def upsample_conv_byprefix(prefix, target_prefix=None):
        if not cfg.TRAIN.LOAD_CONV:
            return
        if target_prefix is None:
            target_prefix = prefix
        target_state_dict[target_prefix + 'weight'] = upsample_conv(
            state_dict[prefix + 'weight'],
            target_state_dict[target_prefix + 'weight'].size())
        # record upsampled keys and target keys
        upsampled_keys.append(prefix)
        changed_target_keys.append(target_prefix)

    def upsample_weighted_conv_byprefix(prefix, target_prefix=None):
        if not cfg.TRAIN.LOAD_CONV:
            return
        if target_prefix is None:
            target_prefix = prefix
        target_state_dict[target_prefix + 'weight'] = upsample_wconv(
            state_dict[prefix + 'weight'], state_dict[prefix + 'Win'],
            state_dict[prefix + 'Wout'],
            target_state_dict[target_prefix + 'weight'].size())
        # record upsampled keys and target keys
        upsampled_keys.append(prefix)
        changed_target_keys.append(target_prefix)

    def upsample_fc_byprefix(prefix, target_prefix=None):
        if target_prefix is None:
            target_prefix = prefix
        weight, bias = upsample_fc(
            state_dict[prefix + 'weight'], state_dict[prefix + 'bias'],
            target_state_dict[target_prefix + 'weight'].size()[1],
            target_state_dict[target_prefix + 'weight'].size()[0])
        target_state_dict[target_prefix + 'weight'] = weight
        target_state_dict[target_prefix + 'bias'] = bias
        # record upsampled keys and target keys
        upsampled_keys.append(prefix)
        changed_target_keys.append(target_prefix)

    def read_stageinfo(keys, stages):
        depth = [0 for _ in range(len(stages))]
        for i in range(len(depth)):
            stage_keys = [
                int(key.split('.')[1][1:]) for key in keys
                if key.startswith(stages[i])
            ]
            depth[i] = max(stage_keys)
        return depth

    depth = read_stageinfo(state_dict.keys(), stages)
    depth_target = read_stageinfo(target_state_dict.keys(), stages)

    # upsample and load stem
    for stem in stems:
        upsample_conv_byprefix(stem + '.conv.')
        if cfg.TRAIN.BN_MODE >= 2:
            upsample_bn_byprefix(stem + '.bn.')

    # upsample and load body
    for s, d, dt in zip(stages, depth, depth_target):
        # upsample the blocks
        for j in range(1, dt + 1):
            # upsample the last stage if no such stages
            block_num = j if j < d else d
            block_prefix = s + '.b' + str(block_num)
            target_block_prefix = s + '.b' + str(j)
            proj_conv = block_prefix + '.proj.'
            # upsample projection conv layer
            if (proj_conv + 'weight' in target_state_dict.keys()
                    and proj_conv + 'weight' in state_dict.keys()):
                if weighted:
                    upsample_weighted_conv_byprefix(proj_conv)
                else:
                    upsample_conv_byprefix(proj_conv)
                proj_bn_prefix = block_prefix + '.bn.'
                if cfg.TRAIN.BN_MODE >= 3:
                    upsample_bn_byprefix(proj_bn_prefix)
            block_conv_prefixs = [
                block_prefix + '.' + key for key in ['f.a.', 'f.b.', 'f.c.']
            ]
            target_block_conv_prefixs = [
                target_block_prefix + '.' + key
                for key in ['f.a.', 'f.b.', 'f.c.']
            ]
            # upsample the convs
            for prefix, target_prefix in zip(block_conv_prefixs,
                                             target_block_conv_prefixs):
                if weighted:
                    upsample_weighted_conv_byprefix(prefix, target_prefix)
                else:
                    upsample_conv_byprefix(prefix, target_prefix)
            # upsample the bns
            block_bn_prefixs = [
                block_prefix + '.' + key
                for key in ['f.a_bn.', 'f.b_bn.', 'f.c_bn.']
            ]
            target_block_bn_prefixs = [
                target_block_prefix + '.' + key
                for key in ['f.a_bn.', 'f.b_bn.', 'f.c_bn.']
            ]
            for prefix, target_prefix in zip(block_bn_prefixs,
                                             target_block_bn_prefixs):
                if cfg.TRAIN.BN_MODE >= 4:
                    upsample_bn_byprefix(prefix, target_prefix)

    # upsample and load head
    for head in heads:
        upsample_fc_byprefix(head + '.')
    ds_projection = {
        key: item
        for key, item in zip(changed_target_keys, upsampled_keys)
    }
    print("weights downsampling projection:")
    print(ds_projection)
    unwrap_model(model).load_state_dict(target_state_dict)
    optimizer.load_state_dict(
        checkpoint["optimizer_state"]) if optimizer else ()
    return 0
