from typing import Callable

import numpy as np
import torch
from torch.nn import Module
from torch.optim import Optimizer

from dpipe.im.utils import identity
from dpipe.torch.utils import *
from dpipe.torch.model import *
from .gumbel_softmax import gumbel_softmax


def train_step_spottune(*inputs: np.ndarray, architecture_main, architecture_policy, k_reg, reg_mode, temperature,
                        criterion: Callable, optimizer_main: Optimizer, optimizer_policy: Optimizer,
                        n_targets: int = 1, loss_key: str = None, k_reg_source=None, with_source=False, alpha_l2sp=None,
                        **optimizer_params) -> np.ndarray:

    architecture_main.train()
    architecture_policy.train()

    inputs = sequence_to_var(*inputs, device=architecture_main)

    if with_source:
        inputs_target = inputs[:2]
        inputs_source = inputs[2:]
        inputs_target, targets_target = inputs_target[0], inputs_target[1]
        inputs_source, targets_source = inputs_source[0], inputs_source[1]

        #  getting the policy (source)
        probs_source = architecture_policy(inputs_source)  # [32, 16]
        action_source = gumbel_softmax(probs_source.view(probs_source.size(0), -1, 2),
                                       temperature=temperature)  # [32, 8, 2]
        policy_source = action_source[:, :, 1]  # [32, 8]

        # getting the policy (target)
        probs = architecture_policy(inputs_target)  # [32, 16]
        action = gumbel_softmax(probs.view(probs.size(0), -1, 2), temperature=temperature)  # [32, 8, 2]
        policy = action[:, :, 1]  # [32, 8]

        # forward (target)
        outputs = architecture_main.forward(inputs_target, policy)
        loss = criterion(outputs, targets_target) + reg_policy(policy=policy, k=k_reg, mode=reg_mode) +\
            reg_policy(policy=policy_source, k=k_reg_source)

    else:
        inputs, targets = inputs[0], inputs[1]

        # getting the policy (target)
        probs = architecture_policy(inputs)  # [32, 16]
        action = gumbel_softmax(probs.view(probs.size(0), -1, 2), temperature=temperature)  # [32, 8, 2]
        policy = action[:, :, 1]  # [32, 8]

        # forward (target)
        outputs = architecture_main.forward(inputs, policy)

        if alpha_l2sp is not None:
            params_unfr, params_frzd = [], []
            for n, p in architecture_main.named_parameters():
                if 'freezed' in n:
                    params_frzd.append(p)
                else:
                    params_unfr.append(p)

            w_diff = torch.tensor(0., requires_grad=True, dtype=torch.float32)
            w_diff.to(get_device(architecture_main))
            for p1, p2 in zip(params_frzd, params_unfr):
                w_diff = w_diff + torch.sum((p1 - p2) ** 2)

            loss = criterion(outputs, targets) + reg_policy(policy=policy, k=k_reg, mode=reg_mode) + alpha_l2sp * w_diff
        else:
            loss = criterion(outputs, targets) + reg_policy(policy=policy, k=k_reg, mode=reg_mode)

    optimizer_step_spottune(optimizer_main, optimizer_policy, loss, **optimizer_params)

    return to_np(loss)


def inference_step_spottune(*inputs: np.ndarray, architecture_main: Module, architecture_policy: Module, temperature,
                            use_gumbel, activation: Callable = identity) -> np.ndarray:
    """
    Returns the prediction for the given ``inputs``.

    Notes
    -----
    Note that both input and output are **not** of type ``torch.Tensor`` - the conversion
    to and from ``torch.Tensor`` is made inside this function.
    """

    architecture_main.eval()
    architecture_policy.eval()

    net_input = sequence_to_var(*inputs, device=architecture_main)

    probs = architecture_policy(*net_input)
    action = gumbel_softmax(probs.view(probs.size(0), -1, 2), use_gumbel=use_gumbel, temperature=temperature)
    policy = action[:, :, 1]

    with torch.no_grad():
        return to_np(activation(architecture_main(*net_input, policy)))


def optimizer_step_spottune(optimizer_main: Optimizer, optimizer_policy: Optimizer,
                            loss: torch.Tensor, **params) -> torch.Tensor:
    """
    Performs the backward pass with respect to ``loss``, as well as a gradient step.

    ``params`` is used to change the optimizer's parameters.

    Examples
    --------
    >>> optimizer = Adam(model.parameters(), lr=1)
    >>> optimizer_step(optimizer, loss) # perform a gradient step
    >>> optimizer_step(optimizer, loss, lr=1e-3) # set lr to 1e-3 and perform a gradient step
    >>> optimizer_step(optimizer, loss, betas=(0, 0)) # set betas to 0 and perform a gradient step

    Notes
    -----
    The incoming ``optimizer``'s parameters are not restored to their original values.
    """
    lr_main, lr_policy = params['lr_main'], params['lr_policy']

    set_params(optimizer_main, lr=lr_main)
    set_params(optimizer_policy, lr=lr_policy)

    optimizer_main.zero_grad()
    optimizer_policy.zero_grad()

    loss.backward()
    optimizer_main.step()
    optimizer_policy.step()

    return loss


def reg_policy(policy, k, mode='l1'):
    if mode == 'l1':
        reg = k * (1 - policy).sum() / torch.numel(policy)  # shape(policy) [batch_size, n_blocks]
    elif mode == 'l2':
        reg = k * torch.sqrt(((1 - policy) ** 2).sum()) / torch.numel(policy)
    else:
        raise ValueError(f'`mode` should be either `l1` or `l2`; but `{mode}` is given')
    return reg
