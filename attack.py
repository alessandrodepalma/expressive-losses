# File containing definitions for adversarial attacks
import torch
import numpy as np
from typing import Callable


def pgd_attack(model, input_lb, input_ub, loss_function: Callable, n_steps, step_size):
    # Note that loss_function is assumed to return an entry per output coordinate (so no reduction e.g., mean)

    step_size_scaling = (input_ub - input_lb) / 2
    attack_point = input_lb.clone()
    attack_loss = (-np.inf) * torch.ones(input_lb.shape[0], dtype=torch.float32, device=input_lb.device)

    with torch.enable_grad():

        # Sample uniformly in input domain
        adv_input = (torch.zeros_like(input_lb).uniform_() * (input_ub - input_lb) + input_lb).detach_()

        for i in range(n_steps):

            adv_input.requires_grad = True
            if adv_input.grad is not None:
                adv_input.grad.zero_()

            adv_outs = model(adv_input)
            obj = loss_function(adv_outs)

            attack_point = torch.where(
                (obj >= attack_loss).view((-1,) + (1,) * (input_lb.dim() - 1)),
                adv_input.detach().clone(), attack_point)
            attack_loss = torch.where(obj >= attack_loss, obj.detach().clone(), attack_loss)

            grad = torch.autograd.grad(obj.sum(), adv_input)[0]
            adv_input = adv_input.detach() + step_size * step_size_scaling * grad.sign()
            adv_input = torch.max(torch.min(adv_input, input_ub), input_lb).detach_()

    if n_steps > 1:
        adv_outs = model(adv_input)
        obj = loss_function(adv_outs)
        attack_point = torch.where(
            (obj >= attack_loss).view((-1,) + (1,) * (input_lb.dim() - 1)),
            adv_input.detach().clone(), attack_point)
    else:
        attack_point = adv_input.detach().clone()

    return attack_point
