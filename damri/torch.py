import os
from copy import deepcopy

import torch
from dpipe.itertools import dmap

from dpipe.torch import load_model_state, get_device, sequence_to_var, optimizer_step, to_np, \
    weighted_cross_entropy_with_logits


def load_model_state_cv3_wise(architecture, baseline_exp_path):
    val_path = os.path.abspath('.')
    exp = val_path.split('/')[-1]
    n_val = int(exp.split('_')[-1])

    n_fold = n_val // 15
    n_cv_block = n_val % 3

    path_to_pretrained_model = os.path.join(baseline_exp_path,
                                            f'experiment_{n_fold * 3 + n_cv_block}', 'model.pth')
    load_model_state(architecture, path=path_to_pretrained_model)


def load_model_state_fold_wise(architecture, baseline_exp_path, n_folds=6, modify_state_fn=None, n_first_exclude=0):
    val_path = os.path.abspath('.')
    exp = val_path.split('/')[-1]
    n_val = int(exp.split('_')[-1]) + n_first_exclude
    path_to_pretrained_model = os.path.join(baseline_exp_path, f'experiment_{n_val // (n_folds - 1)}', 'model.pth')
    load_model_state(architecture, path=path_to_pretrained_model, modify_state_fn=modify_state_fn)


def modify_state_fn_spottune(current_state, state_to_load, init_random=False):
    add_str = '_freezed'
    state_to_load_parallel = deepcopy(state_to_load)
    for key in state_to_load.keys():
        a = key.split('.')
        a[0] = a[0] + add_str
        a = '.'.join(a)
        value_to_load = torch.rand(state_to_load[key].shape).to(state_to_load[key].device) if init_random else \
                        state_to_load[key]
        state_to_load_parallel[a] = value_to_load
    return state_to_load_parallel


def load_two_models_into_spottune(module, path_base, path_post):
    val_path = os.path.abspath('.')
    exp = val_path.split('/')[-1]
    n_val = int(exp.split('_')[-1])
    path_base = os.path.join(path_base, f'experiment_{n_val // 5}', 'model.pth')
    path_post = os.path.join(path_post, f'experiment_{n_val}', 'model.pth')

    state_to_load_base = torch.load(path_base, map_location=get_device(module))
    state_to_load_post = torch.load(path_post, map_location=get_device(module))
    add_key = '_freezed'
    for key in state_to_load_base.keys():
        key_lvls = key.split('.')
        key_lvls[0] = key_lvls[0] + add_key
        key_frzd = '.'.join(key_lvls)
        state_to_load_post[key_frzd] = state_to_load_base[key]
    module.load_state_dict(state_to_load_post)


def freeze_model(model, exclude_layers=('inconv', )):
    for name, param in model.named_parameters():
        requires_grad = False
        for l in exclude_layers:
            if l in name:
                requires_grad = True
        param.requires_grad = requires_grad


def freeze_model_spottune(model):
    for name, param in model.named_parameters():
        if 'freezed' in name:
            requires_grad = False
        else:
            requires_grad = True
        param.requires_grad = requires_grad


def unfreeze_model(model):
    for params in model.parameters():
        params.requires_grad = True


def train_step(*inputs, architecture, criterion, optimizer, n_targets=1, loss_key=None,
               alpha_l2sp=None, reference_architecture=None, **optimizer_params):
    architecture.train()
    if n_targets >= 0:
        n_inputs = len(inputs) - n_targets
    else:
        n_inputs = -n_targets

    assert 0 <= n_inputs <= len(inputs)
    inputs = sequence_to_var(*inputs, device=architecture)
    inputs, targets = inputs[:n_inputs], inputs[n_inputs:]

    if alpha_l2sp is not None:
        if reference_architecture is None:
            raise ValueError('`reference_architecture` should be provided for L2-SP regularization.')

        w_diff = torch.tensor(0., requires_grad=True, dtype=torch.float32)
        w_diff.to(get_device(architecture))
        for p1, p2 in zip(architecture.parameters(), reference_architecture.parameters()):
            w_diff = w_diff + torch.sum((p1 - p2) ** 2)

        loss = criterion(architecture(*inputs), *targets) + alpha_l2sp * w_diff
    else:
        loss = criterion(architecture(*inputs), *targets)

    if loss_key is not None:
        optimizer_step(optimizer, loss[loss_key], **optimizer_params)
        return dmap(to_np, loss)

    optimizer_step(optimizer, loss, **optimizer_params)
    return to_np(loss)
