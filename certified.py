import torch
import torch.nn as nn
from utils import *
from auto_LiRPA import BoundedTensor, BoundDataParallel
from auto_LiRPA.perturbations import *
from auto_LiRPA.bound_ops import *
import pdb
import math


def get_loss_over_lbs(lb):
    lb_padded = torch.cat([torch.zeros_like(lb[:, :1]), lb], dim=1)
    fake_labels = torch.zeros(lb.size(0), dtype=torch.long, device=lb.device)
    robust_loss_ = ce_loss(-lb_padded, fake_labels)
    return robust_loss_    


def get_C(args, data, labels):
    return get_spec_matrix(data, labels, args.num_class)


def get_bound(args, model, x, data=None, labels=None, meter=None, bounding_algorithm="IBP"):
    assert bounding_algorithm in ['IBP', 'CROWN-IBP']

    c, bound_lower, bound_upper = get_C(args, data, labels), True, False

    if bounding_algorithm == "IBP":
        lb, ub = model(x=(x,), method_opt="compute_bounds", IBP=True, C=c, method=None, no_replicas=True)
    else:
        # CROWN-IBP bound computation
        lb, ub = model.compute_bounds(
            x=(x,), IBP=True, C=c, method='backward', bound_lower=bound_lower, bound_upper=bound_upper)

    update_relu_stat(model, meter)
    return lb


def ub_robust_loss(args, model, x, data, labels, meter=None, bounding_algorithm="IBP"):

    lb = get_bound(
        args, model, x, data=data, labels=labels, meter=meter, bounding_algorithm=bounding_algorithm)
    robust_err = torch.sum((lb < 0).any(dim=1)).item() / data.size(0)
    # Pad zero at the beginning for each example, and use fake label '0' for all examples
    robust_loss = get_loss_over_lbs(lb)

    if robust_loss is not None and torch.isnan(robust_loss):
        robust_err = 1.

    return robust_loss, robust_err, lb
