import copy
import os
import torch
import wandb
import json
import math
from tqdm import tqdm

from argparser import parse_args
from utils import set_seed, prepare_model
from config import load_config
from datasets import load_data
from attack import pgd_attack
from certified import get_C, get_loss_over_lbs
from utils import compute_perturbation, ce_loss, compute_sabr_perturbation
from auto_LiRPA.utils import logger
from auto_LiRPA import BoundedModule
from ver_utils import get_ovalbab_network, run_oval_bab_for_1_vs_1_bounds


def main(args, wandb_logging_interval=1000, plot_per_sample=False, run_bab=False):

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
    batch_size = 1
    test_batch_size = 1
    dummy_input, train_data, test_data = load_data(
        args, config['data'], batch_size, test_batch_size, aug=not args.no_data_aug)

    dataset = train_data if args.log_losses_on_train else test_data

    # Log certification data onto wandb.
    if args.wandb_label is not None:
        wandb.init(project="expressive-losses", group=args.wandb_label, config=args)

    if run_bab:
        # Convert net for OVAL BaB use.
        with torch.no_grad():
            torch_verif_layers = get_ovalbab_network(dummy_input, model_ori)
    else:
        torch_verif_layers = None

    # get autolirpa model to attempt quick certification before BaB
    model_lirpa = BoundedModule(
        copy.deepcopy(model_ori), dummy_input, bound_opts=config['bound_params']['bound_opts'], custom_ops={},
        device=args.device)
    model_lirpa.eval()

    eps = args.eps or bound_config['eps']
    data_max, data_min, std = dataset.data_max, dataset.data_min, dataset.std

    pbar = tqdm(dataset, dynamic_ncols=True)
    loss_sums = {
        "nat_loss": 0,
        "adv_loss": 0,
        "ibp_loss": 0,
        "crown_loss": 0,
        "bab_loss": 0,
        "ccibp_loss": 0,
        "mtlibp_loss": 0,
        "expibp_loss": 0,
        "sabr_loss": 0,
    }
    tot_tests = 0
    for test_idx, (inputs, targets) in enumerate(pbar):
        if test_idx < args.start_idx or (args.end_idx != -1 and test_idx >= args.end_idx):
            continue

        tot_tests += 1

        # Standard accuracy.
        nat_outs = model_lirpa(inputs.cuda()).cpu()
        nat_loss = ce_loss(nat_outs, targets)
        loss_sums["nat_loss"] += nat_loss

        x, data_lb, data_ub = compute_perturbation(
            args, eps, inputs.to(args.device), data_min.to(args.device), data_max.to(args.device), std.to(args.device),
            True, False)

        # attack to compute the adversarial loss
        with torch.no_grad():
            adv_data = pgd_attack(
                model_lirpa, data_lb.cuda(), data_ub.cuda(),
                lambda x: torch.nn.CrossEntropyLoss(reduction='none')(x, targets.cuda()),
                args.test_att_n_steps, args.test_att_step_size)
            adv_outs = model_lirpa(adv_data.cuda()).cpu()
            adv_loss = ce_loss(adv_outs, targets)
            loss_sums["adv_loss"] += adv_loss

        # Check whether the best of IBP/CROWN bounds suffice to prove the property (computes a bound per logit
        # difference, rather than one-vs-all as BaB)
        c = get_C(args, inputs.to(args.device), targets.to(args.device))
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
        ibplb = ibplb.cpu()
        lb = lb.cpu()
        lb = torch.max(lb, ibplb)

        crown_loss = get_loss_over_lbs(lb)
        loss_sums["crown_loss"] += crown_loss
        ibp_loss = get_loss_over_lbs(ibplb)
        loss_sums["ibp_loss"] += ibp_loss

        if run_bab:
            with torch.no_grad():
                # Prepare to use OVAL BaB for verification
                assert inputs.shape[0] == 1, "only test_batch=1 is supported for OVAL BaB"
                torch_input_bounds = torch.stack([data_lb, data_ub], dim=-1)
                torch_input_bounds = torch_input_bounds.squeeze(0)
                torch_targets = targets.squeeze(0)
                # Run BaB (num_classes - 1) times with the provided timeout each
                bab_lbs = run_oval_bab_for_1_vs_1_bounds(
                    torch_verif_layers, inputs, torch_targets, torch_input_bounds, args.oval_bab_config,
                    args.num_class, timeout=args.oval_bab_timeout)
                bab_loss = get_loss_over_lbs(bab_lbs.unsqueeze(0))
                loss_sums["bab_loss"] += bab_loss
            # release some memory
            torch.cuda.empty_cache()
        else:
            bab_loss = 0

        # attack to compute the train-time losses (uses training args)
        attack_eps = eps * args.attack_eps_factor
        _, attack_lb, attack_ub = compute_perturbation(
            args, attack_eps, inputs, data_min, data_max, std, True, False)

        with torch.no_grad():
            train_adv_data = pgd_attack(
                model_lirpa, attack_lb.cuda(), attack_ub.cuda(),
                lambda x: torch.nn.CrossEntropyLoss(reduction='none')(x, targets.cuda()),
                args.train_att_n_steps, args.train_att_step_size)
            train_adv_outs = model_lirpa(train_adv_data.cuda()).cpu()
            train_adv_loss = ce_loss(train_adv_outs, targets)

        # cross_entropy of convex combination of IBP with natural/adversarial logits
        train_adv_diff = torch.bmm(
            get_C(args, inputs, targets),
            train_adv_outs.unsqueeze(-1)).squeeze(-1)
        ccibp_loss = get_loss_over_lbs(args.ccibp_coeff * ibplb + (1 - args.ccibp_coeff) * train_adv_diff)
        loss_sums["ccibp_loss"] += ccibp_loss
        mtlibp_loss = args.mtlibp_coeff * ibp_loss + (1 - args.mtlibp_coeff) * train_adv_loss
        loss_sums["mtlibp_loss"] += mtlibp_loss
        expibp_loss = ibp_loss ** args.expibp_coeff * train_adv_loss ** (1 - args.expibp_coeff)
        loss_sums["expibp_loss"] += expibp_loss

        sabr_x, _ = compute_sabr_perturbation(
            args, attack_eps, inputs.to(args.device), train_adv_data.to(args.device), data_min.to(args.device),
            data_max.to(args.device), std.to(args.device), True, False)
        sabr_lb, _ = model_lirpa(x=(sabr_x,), method_opt="compute_bounds", IBP=True, C=c, method=None, no_replicas=True)
        sabr_loss = get_loss_over_lbs(sabr_lb)

        # Log certification data onto wandb.
        if args.wandb_label is not None:
            if plot_per_sample:
                wandb_dict = {
                    f'sample_val_nat_loss': nat_loss,
                    f'sample_val_adv_loss': adv_loss,
                    f'sample_val_ibp_loss': ibp_loss,
                    f'sample_val_crown_loss': crown_loss,
                    f'sample_val_bab_loss': bab_loss,
                    f'sample_val_ccibp_loss': ccibp_loss,
                    f'sample_val_mtlibp_loss': mtlibp_loss,
                    f'sample_val_expibp_loss': expibp_loss,
                    f'sample_val_sabr_loss': sabr_loss,
                }
                if run_bab:
                    wandb_dict['sample_val_bab_loss'] = bab_loss
                wandb.log(wandb_dict, step=test_idx)
            elif (test_idx + 1) % wandb_logging_interval == 0:
                wandb_dict = {f'val_{k}': loss_sums[k]/tot_tests for k in loss_sums.keys()}
                wandb.log(wandb_dict, step=test_idx)

        pbar_description = "tot_tests: %d"
        pbar_list = [tot_tests]
        for k in loss_sums.keys():
            pbar_description += f", {k}: %.5lf"
            pbar_list.append(loss_sums[k]/tot_tests)
        pbar.set_description(pbar_description % tuple(pbar_list))

    # Log certification data onto wandb.
    if args.wandb_label is not None:
        wandb.run.summary["model_dir"] = args.load
        wandb.run.summary["host_name"] = os.uname().nodename


if __name__ == '__main__':

    if torch.cuda.is_available():
        # Disable the 19-bit TF32 type, which is not precise enough for verification purposes
        # see https://pytorch.org/docs/stable/notes/cuda.html#tf32-on-ampere
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    args = parse_args()
    main(args, plot_per_sample=(args.oval_bab_config is not None), run_bab=(args.oval_bab_config is not None))
