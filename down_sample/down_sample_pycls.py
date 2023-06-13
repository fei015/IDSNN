import torch
from .core.weight_upsample import upsample_conv, upsample_fc, upsample_bn, upsample_wconv, merge_convs, merge_bns
from .core.config import cfg
from .core.io import pathmgr
from .core.net import unwrap_model
import ipdb

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
    target_stems = ['conv1']
    stages = ['s1', 's2', 's3', 's4']
    target_stages= ['conv2_x', 'conv3_x', 'conv4_x', 'conv5_x']
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
            stage_keys = [int(key.split('.')[1][1:]) for key in keys if key.startswith(stages[i])]
            depth[i] = max(stage_keys)
        return depth
    
    def target_read_stageinfo(keys, stages):
        depth = [0 for _ in range(len(stages))]
        for i in range(len(depth)):
            stage_keys = [int(key.split('.')[1]) for key in keys if key.startswith(stages[i])]
            depth[i] = max(stage_keys)+1
        return depth

    depth = read_stageinfo(state_dict.keys(), stages)
    target_depth = target_read_stageinfo(target_state_dict.keys(), target_stages)

    # upsample and load stem
    for stem, target_stem in zip(stems, target_stems):
        upsample_conv_byprefix(stem + '.conv.', target_stem + '.0.')
        if cfg.TRAIN.BN_MODE >= 2:
            upsample_bn_byprefix(stem + '.bn.', target_stem + '.1.bn.')

    # upsample and load body
    for s, target_s, d, dt in zip(stages, target_stages, depth, target_depth):
        # upsample the blocks
        for j in range(1, dt + 1):
            # upsample the last stage if no such stages
            block_num = j if j < d else d
            block_prefix = s + '.b' + str(block_num)
            target_block_prefix = target_s + '.' + str(j-1) + '.residual_function.'

            # project_conv暂时不做考虑
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
                target_block_prefix + key
                for key in ['0.', '3.']
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
                target_block_prefix + key
                for key in ['1.bn.', '4.bn.']
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
