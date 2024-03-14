import copy
import os
import pickle
import torch
import wandb
import json
import math
from tqdm import tqdm

from argparser import parse_args
from utils import set_seed, prepare_model
from config import load_config
from datasets import load_data
from ver_utils import get_ovalbab_network, run_oval_bab, create_1_vs_all_verification_problem
from attack import pgd_attack
from certified import get_C
from utils import compute_perturbation
from auto_LiRPA.utils import logger
from auto_LiRPA import BoundedModule


wandb_logging_interval = 100


def report(args, pbar, ver_logdir, tot_verified_corr, tot_crown_verified, tot_ibp_verified, tot_nat_ok, tot_pgd_ok,
           test_idx, tot_tests, test_data):
    """ Logs evaluation statistics to standard output. """
    pbar.set_description(
        'tot_tests: %d, oval_ver_ok: %.5lf [%d/%d], oval_nat_ok: %.5lf [%d/%d], oval_pgd_ok: %.5lf [%d/%d]' % (
            tot_tests,
            tot_verified_corr/tot_tests, tot_verified_corr, tot_tests,
            tot_nat_ok/tot_tests, tot_nat_ok, tot_tests,
            tot_pgd_ok/tot_tests, tot_pgd_ok, tot_tests,
        )
    )
    out_file = os.path.join(ver_logdir, '{}.p'.format(test_idx))
    with open(out_file, 'wb') as file:
        pickle.dump(test_data, file)

    # Log certification data onto wandb.
    if args.wandb_label is not None and ((test_idx + 1) % wandb_logging_interval == 0):
        prefix = 'oval' if not args.crown_ver else 'crown'
        wandb_dict = {
            f'{prefix}_ver_ok': tot_verified_corr/tot_tests,
            'ibp_ver_ok': tot_ibp_verified/tot_tests,
            f'{prefix}_nat_ok': tot_nat_ok/tot_tests,
            f'{prefix}_pgd_ok': tot_pgd_ok/tot_tests,
        }
        if not args.crown_ver:
            wandb_dict['crown_ver_ok'] = tot_crown_verified/tot_tests
        wandb.log(wandb_dict, step=test_idx)


def main(args):

    if torch.cuda.is_available():
        # Disable the 19-bit TF32 type, which is not precise enough for verification purposes
        # see https://pytorch.org/docs/stable/notes/cuda.html#tf32-on-ampere
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    config = load_config(args.config)
    bound_config = config['bound_params']
    logger.info('config: {}'.format(json.dumps(config)))

    # Set random seed.
    set_seed(args.seed or config['seed'])

    # Create folder for logs.
    ver_logdir = args.load.split(".")[0] + '_ver'
    if not os.path.exists(ver_logdir):
        os.makedirs(ver_logdir)

    # Load dataset and network.
    model_ori, checkpoint, epoch, best = prepare_model(args, logger, config)
    model_ori.eval()
    batch_size = (args.batch_size or config['batch_size'])
    test_batch_size = 1
    dummy_input, _, test_data = load_data(
        args, config['data'], batch_size, test_batch_size, aug=not args.no_data_aug)

    # Log certification data onto wandb.
    if args.wandb_label is not None:
        wandb.init(project="expressive-losses", group=args.wandb_label, config=args)

    # Convert net for OVAL BaB use.
    with torch.no_grad():
        torch_verif_layers = get_ovalbab_network(dummy_input, model_ori)
        torch_net = torch.nn.Sequential(*[copy.deepcopy(lay).cuda() for lay in torch_verif_layers])

    # get autolirpa model to attempt quick certification before BaB
    model_lirpa = BoundedModule(
        copy.deepcopy(model_ori), dummy_input, bound_opts=config['bound_params']['bound_opts'], custom_ops={},
        device=args.device)
    model_lirpa.eval()

    eps = args.eps or bound_config['eps']
    data_max, data_min, std = test_data.data_max, test_data.data_min, test_data.std

    pbar = tqdm(test_data, dynamic_ncols=True)
    tot_verified, tot_crown_verified, tot_ibp_verified, tot_nat_ok, tot_pgd_ok, tot_tests = 0, 0, 0, 0, 0, 0
    for test_idx, (inputs, targets) in enumerate(pbar):
        if test_idx < args.start_idx or (args.end_idx != -1 and test_idx >= args.end_idx):
            continue

        tot_tests += 1
        test_file = os.path.join(ver_logdir, '{}.p'.format(test_idx))
        if (not args.no_load) and os.path.isfile(test_file):
            with open(test_file, 'rb') as file:
                test_data = pickle.load(file)
        else:
            test_data = {}

        # Standard accuracy.
        nat_outs = torch_net(inputs.cuda()).cpu()
        nat_ok = targets.eq(nat_outs.max(dim=1)[1]).item()

        # Logging.
        tot_nat_ok += nat_ok
        test_data['ok'] = nat_ok
        if not nat_ok:
            test_data['pgd_ok'] = 0.
            test_data['ver_ok'] = 0.
            test_data['crown_ok'] = 0.
            test_data['ibp_ok'] = 0.
            report(args, pbar, ver_logdir, tot_verified, tot_crown_verified, tot_ibp_verified,
                   tot_nat_ok, tot_pgd_ok, test_idx, tot_tests, test_data)
            continue

        # Prepare to use OVAL BaB for verification
        assert inputs.shape[0] == 1, "only test_batch=1 is supported for OVAL BaB"
        # Compute the perturbation
        norm_eps = eps
        if type(norm_eps) == float:
            norm_eps = (norm_eps / std).view(1, -1, 1, 1)
        else:  # [batch_size, channels]
            norm_eps = (norm_eps.view(*norm_eps.shape, 1, 1) / std.view(1, -1, 1, 1))
        data_ub = torch.min(inputs + norm_eps, data_max)
        data_lb = torch.max(inputs - norm_eps, data_min)
        torch_input_bounds = torch.stack([data_lb, data_ub], dim=-1)
        torch_input_bounds = torch_input_bounds.squeeze(0)
        torch_targets = targets.squeeze(0)

        # Run a quick attack before BaB
        with torch.no_grad():
            adv_data = pgd_attack(
                torch_net, data_lb.cuda(), data_ub.cuda(),
                lambda x: torch.nn.CrossEntropyLoss(reduction='none')(x, targets.cuda()),
                args.test_att_n_steps, args.test_att_step_size)
            adv_outs = torch_net(adv_data.cuda()).cpu()
            adv_ok = targets.eq(adv_outs.max(dim=1)[1]).item()
        if not adv_ok:
            test_data['pgd_ok'] = 0.
            test_data['ver_ok'] = 0.
            test_data['crown_ok'] = 0.
            test_data['ibp_ok'] = 0.
            print("==========> A quick PGD attack found a vulnerability (no BaB needed)")
            report(args, pbar, ver_logdir, tot_verified, tot_crown_verified, tot_ibp_verified,
                   tot_nat_ok, tot_pgd_ok, test_idx, tot_tests, test_data)
            continue

        # Check whether the best of IBP/CROWN bounds suffice to prove the property (computes a bound per logit
        # difference, rather than one-vs-all as BaB)
        c = get_C(args, inputs.to(args.device), targets.to(args.device))
        x, data_lb, data_ub = compute_perturbation(
            args, eps, inputs.to(args.device), data_min.to(args.device), data_max.to(args.device), std.to(args.device),
            True, False)
        if args.lirpa_crown_batch is not None:
            # compute the bounds in batches to save memory
            lb = -torch.ones((1, args.num_class - 1), device=c.device)
            n_batches = int(math.ceil(args.num_class / float(args.lirpa_crown_batch)))
            for sub_batch_idx in range(n_batches):
                # compute intermediate bounds on sub-batch
                start_batch_index = sub_batch_idx * args.lirpa_crown_batch
                end_batch_index = min((sub_batch_idx + 1) * args.lirpa_crown_batch, args.num_class)
                lb[:, start_batch_index:end_batch_index], _ = model_lirpa.compute_bounds(
                    x=(x,), IBP=False, C=c[:, start_batch_index:end_batch_index, :], method='CROWN',
                    bound_lower=True, bound_upper=False)
        else:
            lb, _ = model_lirpa.compute_bounds(
                x=(x,), IBP=False, C=c, method='CROWN', bound_lower=True, bound_upper=False)
        ibplb, _ = model_lirpa(x=(x,), method_opt="compute_bounds", IBP=True, C=c, method=None, no_replicas=True)
        lb = torch.max(lb, ibplb)

        if args.debug:
            # DEBUG: check if OVAL-based and auto_LiRPA based bounds match
            from plnn.naive_approximation import NaiveNetwork
            intermediate_net = NaiveNetwork(torch_net)
            verif_domain = torch_input_bounds.cuda().unsqueeze(0)
            intermediate_net.define_linear_approximation(verif_domain, override_numerical_errors=True)
            o_lbs = intermediate_net.lower_bounds[-1].squeeze(0).cpu()
            d_lbs, _ = model_lirpa(x=(x,), method_opt="compute_bounds", IBP=True, method=None, no_replicas=True)
            assert (o_lbs - d_lbs.cpu()).abs().max() < 1e-1

        ibp_verified = ibplb.min() > 0
        test_data['ibp_ok'] = int(ibp_verified)
        tot_ibp_verified += int(ibp_verified)
        verified = lb.min() > 0
        test_data['crown_ok'] = int(verified)
        tot_crown_verified += int(verified)
        if verified:
            tot_pgd_ok += 1.
            tot_verified += 1.
            test_data['pgd_ok'] = 1.
            test_data['ver_ok'] = 1.
            print("==========> CROWN verified the property (no BaB needed)")
            report(args, pbar, ver_logdir, tot_verified, tot_crown_verified, tot_ibp_verified,
                   tot_nat_ok, tot_pgd_ok, test_idx, tot_tests, test_data)
            continue

        if args.crown_ver:
            tot_pgd_ok += 1.
            test_data['pgd_ok'] = 1.
            test_data['ver_ok'] = 0.
            print("==========> CROWN didn't verify the property")
            report(args, pbar, ver_logdir, tot_verified, tot_crown_verified, tot_ibp_verified,
                   tot_nat_ok, tot_pgd_ok, test_idx, tot_tests, test_data)
            continue

        # Make problem a 1vsall robustness verification problem
        with torch.no_grad():
            torch_verif_problem = create_1_vs_all_verification_problem(
                torch_verif_layers, torch_targets, torch_input_bounds, args.ib_batch_size, inputs, c)
        # release some memory
        torch.cuda.empty_cache()

        with torch.no_grad():
            verified, pgd_ok = run_oval_bab(
                torch_verif_problem, torch_input_bounds, args.oval_bab_config, timeout=args.oval_bab_timeout)

        tot_pgd_ok += int(pgd_ok)
        test_data['pgd_ok'] = int(pgd_ok)

        tot_verified += int(verified)
        test_data['ver_ok'] = int(verified)

        report(args, pbar, ver_logdir, tot_verified, tot_crown_verified, tot_ibp_verified,
               tot_nat_ok, tot_pgd_ok, test_idx, tot_tests, test_data)

    # Log certification data onto wandb.
    if args.wandb_label is not None:
        wandb.run.summary["model_dir"] = args.load
        wandb.run.summary["host_name"] = os.uname().nodename


if __name__ == '__main__':

    args = parse_args()
    main(args)
