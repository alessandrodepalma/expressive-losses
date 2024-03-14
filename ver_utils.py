import torch
import sys
import os
import json
import time
import gc

from tools.custom_torch_modules import Add, Mul, Flatten
import tools.bab_tools.vnnlib_utils as vnnlib_utils
from tools.bab_tools.model_utils import one_vs_all_from_model as ovalbab_1vsall_constructor, add_single_prop
import tools.bab_tools.bab_runner as ovalbab_runner
from models.utils import Flatten as shi_flatten


# Disable OVAL BaB's verbose printing.
class do_not_print:
    # Taken from https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def convert_layers(x, layers):

    def reshaper(inp, ndim):
        if inp.dim() < ndim:
            return inp.view(inp.shape + (1,) * (ndim - 1))
        else:
            return inp

    converted_layers = []
    for lay in layers:
        if isinstance(lay, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
            assert lay.track_running_stats

            ndim = x.dim() - 1
            mean = reshaper(lay.running_mean, ndim)
            var = reshaper(lay.running_var, ndim)
            weight = reshaper(lay.weight, ndim)
            bias = reshaper(lay.bias, ndim)

            op1 = Add(-mean)
            op2 = Mul(1/torch.sqrt(var + lay.eps))
            op3 = Mul(weight)
            op4 = Add(bias)
            x = torch.nn.Sequential(op1, op2, op3, op4)(x)
            converted_layers.extend([op1, op2, op3, op4])

        elif isinstance(lay, shi_flatten):
            x = lay(x)
            converted_layers.append(Flatten())
        else:
            x = lay(x)
            converted_layers.append(lay)

    return torch.nn.Sequential(*converted_layers)


def get_ovalbab_network(in_example, layers):

    model = convert_layers(in_example.cpu(), layers)
    # check that the two models coincide in forward pass
    assert (model(in_example.cpu()) - layers(in_example.cpu())).abs().max() < 1e-3

    # Assert that the model specification is currently supported.
    supported = vnnlib_utils.is_supported_model(model)
    assert supported

    layers = list(model.children())
    for clayer in layers:
        if isinstance(clayer, torch.nn.Linear):
            clayer.bias.data = clayer.bias.data.squeeze(0)
    return layers


def create_1_vs_all_verification_problem(model, y, input_bounds, max_solver_batch, inputs, c):
    with do_not_print():
        verif_layers = ovalbab_1vsall_constructor(
            torch.nn.Sequential(*model), y, domain=input_bounds, max_solver_batch=max_solver_batch, use_ib=True,
            num_classes=model[-1].weight.shape[0])

    # Assert the functional equivalence of 1_vs_all with the original network
    out_diff_min = torch.nn.Sequential(*verif_layers)(inputs)
    out = torch.nn.Sequential(*model)(inputs)
    out_diff = torch.bmm(c.cpu(), out.unsqueeze(-1)).squeeze(-1)
    assert (out_diff_min - out_diff.min()).abs() < 1e-4

    return verif_layers


def run_oval_bab(verif_layers, input_bounds, ovalbab_json_config, timeout=20):
    # Run OVAL-BaB with the configuration specified in ovalbab_json_config
    return_dict = dict()
    start_time = time.time()

    with open(ovalbab_json_config) as json_file:
        json_params = json.load(json_file)
    with do_not_print():
        ovalbab_runner.bab_from_json(
            json_params, verif_layers, input_bounds, return_dict, None, instance_timeout=timeout, start_time=start_time)
    del json_params

    bab_out, bab_nb_states = ovalbab_runner.bab_output_from_return_dict(return_dict)
    bab_time = time.time() - start_time

    torch.cuda.empty_cache()
    gc.collect()

    return bab_out == "False", bab_out != "True"

def create_1_vs_1_verification_problem(model, y, y_other, inputs, num_class):
    verif_layers = add_single_prop(model, y, y_other, num_classes=num_class)

    # Assert the functional equivalence of 1_vs_1 with the original network
    out_diff = torch.nn.Sequential(*verif_layers)(inputs)
    out = torch.nn.Sequential(*model)(inputs).squeeze(0)
    assert (out_diff - (out[y] - out[y_other])).abs() < 1e-4
    return verif_layers


def run_oval_bab_for_1_vs_1_bounds(model, inputs, y, input_bounds, ovalbab_json_config, num_class, lbs=None, timeout=20):
    # if the lbs are provided, go into ranking w/ threshold mode

    ordering = range(num_class)
    n_below_thr = num_class - 1.
    if lbs is not None:
        if y == (num_class - 1):
            lbs_with_gt = torch.cat([lbs.squeeze(0), torch.tensor([0.], device=lbs.device, dtype=lbs.dtype)])
        else:
            lbs_with_gt = torch.cat(
                [lbs.squeeze(0)[:y], torch.tensor([0.], device=lbs.device, dtype=lbs.dtype), lbs.squeeze(0)[y:]])

        ordering = list(torch.argsort(lbs_with_gt))
        decision_bound = 3.  # the influence on the loss is small after 3., wasted time. Not putting this into BaB as
        # otherwise it returns with low UBs
        n_below_thr = max(float((lbs < decision_bound).sum().item()), 1.)

    # get root IBs
    with open(ovalbab_json_config) as json_file:
        json_params = json.load(json_file)
    y_other = 0 if y != 0 else 1
    verif_layers = create_1_vs_1_verification_problem(model, y, y_other, inputs, num_class)
    ib_lbs, ib_ubs = ovalbab_runner.bab_from_json(
        json_params, verif_layers, input_bounds, {}, None, instance_timeout=0.,
        start_time=time.time(), return_bounds_if_timeout=True, decision_bound=None, return_ibs_root=True
    )
    del json_params
    ib_lbs = ib_lbs[:-1]
    ib_ubs = ib_ubs[:-1]

    outputs = [0. for _ in range(num_class)]
    for y_other in ordering:

        if y_other == y:
            continue

        # Run OVAL-BaB with the configuration specified in ovalbab_json_config
        return_dict = dict()
        with open(ovalbab_json_config) as json_file:
            json_params = json.load(json_file)

        verif_layers = create_1_vs_1_verification_problem(model, y, y_other, inputs, num_class)

        if lbs is not None:
            c_timeout = ((timeout * (num_class - 1)) / n_below_thr) if lbs_with_gt[y_other] < decision_bound else 0.
        else:
            c_timeout = timeout

        # root IBs are passed
        ovalbab_runner.bab_from_json(
            json_params, verif_layers, input_bounds, return_dict, None, instance_timeout=c_timeout,
            start_time=time.time(), return_bounds_if_timeout=True, decision_bound=None, precomputed_ibs=(ib_lbs, ib_ubs)
        )
        del json_params
        outputs[y_other] = return_dict["min_lb"]

        torch.cuda.empty_cache()
        gc.collect()

    del outputs[y]
    return torch.tensor(outputs)