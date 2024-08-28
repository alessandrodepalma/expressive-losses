import torch
import wandb
from auto_LiRPA import BoundedModule, CrossEntropyWrapper, BoundedTensor
from auto_LiRPA.perturbations import *
from auto_LiRPA.utils import MultiAverageMeter
from auto_LiRPA.bound_ops import *
from config import load_config
from datasets import load_data
from utils import *
from manual_init import manual_init, kaiming_init
from argparser import parse_args
from certified import ub_robust_loss, get_loss_over_lbs, get_C
from attack import pgd_attack
from regularization import compute_reg, compute_L1_reg
from tqdm import tqdm

args = parse_args()

if not args.verify:
    set_file_handler(logger, args.dir)
logger.info('Arguments: {}'.format(args))


def epsilon_clipping(eps, eps_scheduler, args, train):
    if eps < args.min_eps:
        eps = args.min_eps
    if args.fix_eps or (not train):
        eps = eps_scheduler.get_max_eps()
    if args.natural:
        eps = 0.
    return eps


def train_or_test(model, model_ori, t, loader, eps_scheduler, opt):
    # Function used both for training and testing purposes

    train = opt is not None
    meter = MultiAverageMeter()

    data_max, data_min, std = loader.data_max, loader.data_min, loader.std
    if args.device == 'cuda':
        data_min, data_max, std = data_min.cuda(), data_max.cuda(), std.cuda()

    if train:
        model_ori.train(); model.train(); eps_scheduler.train()
        eps_scheduler.step_epoch()
    else:
        model_ori.eval(); model.eval(); eps_scheduler.eval()

    pbar = tqdm(loader, dynamic_ncols=True)

    for i, (data, labels) in enumerate(pbar):
        start = time.time()
        eps_scheduler.step_batch()
        eps = eps_scheduler.get_eps()
        epoch_progress = (i+1) * 1. / len(loader) if train else 1.0

        if train:
            eps *= args.train_eps_mul
            att_n_steps = args.train_att_n_steps
            att_step_size = args.train_att_step_size
            bounding_algorithm = args.bounding_algorithm
        else:
            att_n_steps = args.test_att_n_steps
            att_step_size = args.test_att_step_size
            bounding_algorithm = "IBP"  # at eval time always use IBP
        attack_eps = eps * args.attack_eps_factor

        eps = epsilon_clipping(eps, eps_scheduler, args, train)
        attack_eps = epsilon_clipping(attack_eps, eps_scheduler, args, train)

        reg = t <= args.num_reg_epochs

        # For small eps just use natural training, no need to compute LiRPA bounds
        batch_method = 'natural' if (eps < 1e-50) else 'robust'
        robust = batch_method == 'robust'

        # labels = labels.to(torch.long)
        if args.device == 'cuda':
            data, labels = data.cuda().detach().requires_grad_(), labels.cuda()

        data_batch, labels_batch = data, labels
        grad_acc = args.grad_acc_steps
        assert data.shape[0] % grad_acc == 0
        bsz = data.shape[0] // grad_acc

        for k in range(grad_acc):
            if grad_acc > 1:
                data, labels = data_batch[bsz*k:bsz*(k+1)], labels_batch[bsz*k:bsz*(k+1)]

            # Compute regular cross-entropy loss
            # NOTE: all forward passes should be carried out on the LiRPA model to avoid batch_norm stats mismatches
            output = model(data)
            regular_ce = ce_loss(output, labels)  # regular CrossEntropyLoss used for warming up
            regular_err = torch.sum(torch.argmax(output, dim=1) != labels).item() / data.size(0)

            # Compute the perturbation
            # NOTE: at validation (train=false) these losses and errors are computed on the target epsilon
            x, data_lb, data_ub = compute_perturbation(args, eps, data, data_min, data_max, std, robust, reg)
            # Run a PGD attack
            if att_n_steps is not None and att_n_steps > 0:

                if train:
                    # attack perturbation with a possibly different epsilon
                    _, attack_lb, attack_ub = compute_perturbation(
                        args, attack_eps, data, data_min, data_max, std, robust, reg)

                    # set the network in eval mode before the attack
                    model_ori.eval()
                    model.eval()
                else:
                    attack_lb = data_lb
                    attack_ub = data_ub

                with torch.no_grad():
                    adv_data = pgd_attack(
                        model, attack_lb, attack_ub,
                        lambda x: nn.CrossEntropyLoss(reduction='none')(x, labels), att_n_steps, att_step_size)
                    del attack_lb, attack_ub  # save a bit of memory

                if train:
                    # reset the network in train mode post-attack (the adversarial point is evaluated in train mode)
                    model_ori.train()
                    model.train()

                adv_output = model(adv_data)
                adv_loss = ce_loss(adv_output, labels)
                adv_err = torch.sum(torch.argmax(adv_output, dim=1) != labels).item() / data.size(0)

            else:
                adv_loss = regular_ce
                adv_err = regular_err
                adv_output = output

            # Upper bound on the robust loss (via IBP)
            # NOTE: when training, the bounding computation will use the BN statistics from the last forward pass: in
            # this case, from the adversarial points
            if robust or reg:

                if (not args.sabr) or (not train):
                    robust_loss, robust_err, lb = ub_robust_loss(
                        args, model, x, data, labels, meter=meter, bounding_algorithm=bounding_algorithm)
                else:
                    sabr_x, sabr_center = compute_sabr_perturbation(
                        args, attack_eps, data, adv_data, data_min, data_max, std, robust, reg)
                    robust_loss, robust_err, lb = ub_robust_loss(
                        args, model, sabr_x, sabr_center, labels, meter=meter, bounding_algorithm=bounding_algorithm)

            else:
                lb = robust_loss = robust_err = None

            update_meter(meter, regular_ce, robust_loss, adv_loss, regular_err, robust_err, adv_err, data.size(0))

            if train:

                if not (args.ccibp or args.mtlibp or args.sabr or args.expibp):

                    if reg:
                        loss = compute_reg(args, model, meter, eps, eps_scheduler)
                    else:
                        loss = torch.tensor(0.).to(args.device)
                    if robust:
                        loss += robust_loss
                    else:
                        # warmup phase
                        loss += regular_ce

                else:
                    if reg and args.reg_lambda > 0:
                        loss = compute_reg(args, model, meter, eps, eps_scheduler)
                    else:
                        loss = torch.tensor(0.).to(args.device)
                    loss += compute_L1_reg(args, model_ori, meter)

                    if robust:
                        if args.ccibp:
                            # cross_entropy of convex combination of IBP with natural/adversarial logits
                            adv_diff = torch.bmm(
                                get_C(args, data, labels),
                                adv_output.unsqueeze(-1)).squeeze(-1)
                            ccibp_diff = args.ccibp_coeff * lb + (1 - args.ccibp_coeff) * adv_diff
                            loss += get_loss_over_lbs(ccibp_diff)
                        elif args.mtlibp:
                            mtlibp_loss = args.mtlibp_coeff * robust_loss + (1 - args.mtlibp_coeff) * adv_loss
                            loss += mtlibp_loss
                        elif args.expibp:
                            expibp_loss = robust_loss ** args.expibp_coeff * adv_loss ** (1 - args.expibp_coeff)
                            loss += expibp_loss
                        else:
                            # sabr
                            sabr_loss = robust_loss
                            loss += sabr_loss
                    else:
                        # warmup phase
                        loss += regular_ce

                meter.update('Loss', loss.item(), data.size(0))

                loss /= grad_acc
                loss.backward()

        if train:
            grad_norm = torch.nn.utils.clip_grad_norm_(model_ori.parameters(), max_norm=args.grad_norm)
            meter.update('grad_norm', grad_norm)
            opt.step()
            opt.zero_grad()

        meter.update('wnorm', get_weight_norm(model_ori))
        meter.update('Time' , time.time() - start)

        pbar.set_description(
            ('[T]' if train else '[V]') +
            ' epoch=%d, nat_loss=%.4f, nat_ok=%.4f, adv_ok=%.4f, ver_ok=%.4f, eps=%.4f' % (
                t,
                meter.avg('CE'),
                1. - meter.avg('Err'),
                1. - meter.avg('Adv_Err'),
                1. - meter.avg('Rob_Err'),
                eps
            )
        )

    if batch_method != 'natural':
        meter.update('eps', eps)

    if train:
        log_dict = {
            'train_nat_loss': meter.avg('CE'),
            'train_nat_ok': 1. - meter.avg('Err'),
            'train_adv_ok': 1. - meter.avg('Adv_Err'),
            'train_adv_loss': meter.avg('Adv_Loss'),
            'train_ver_ok': 1. - meter.avg('Rob_Err'),
            'train_ver_loss': meter.avg('Rob_Loss'),
        }
    else:
        log_dict = {
            'val_nat_loss': meter.avg('CE'),
            'val_nat_ok': 1. - meter.avg('Err'),
            'val_adv_ok': 1. - meter.avg('Adv_Err'),
            'val_adv_loss': meter.avg('Adv_Loss'),
            'val_ver_ok': 1. - meter.avg('Rob_Err'),
            'val_ver_loss': meter.avg('Rob_Loss'),
        }
    if args.wandb_label is not None:
        wandb.log(log_dict, step=t)

    return meter, log_dict


def main(args):

    if torch.cuda.is_available() and args.disable_train_tf32:
        # Disable the 19-bit TF32 type, which is not precise enough for verification purposes, and seems to hurt
        # performance a bit for training
        # see https://pytorch.org/docs/stable/notes/cuda.html#tf32-on-ampere
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    config = load_config(args.config)
    logger.info('config: {}'.format(json.dumps(config)))
    set_seed(args.seed or config['seed'])
    model_ori, checkpoint, epoch, _ = prepare_model(args, logger, config)
    logger.info('Model structure: \n {}'.format(str(model_ori)))
    timestamp = int(time.time())
    name_prefix = args.model + "_" + args.method + f"_{timestamp}_"

    # Log training data onto wandb for monitoring purposes.
    if args.wandb_label is not None:
        wandb.init(project="expressive-losses", group=args.wandb_label, config=args)

    log_dict = {}
    custom_ops = {}
    bound_config = config['bound_params']
    batch_size = (args.batch_size or config['batch_size'])
    test_batch_size = args.test_batch_size or batch_size
    dummy_input, train_data, test_data = load_data(
        args, config['data'], batch_size, test_batch_size, aug=not args.no_data_aug)
    bound_opts = bound_config['bound_opts']

    model_ori.train()
    model = BoundedModule(model_ori, dummy_input, bound_opts=bound_opts, custom_ops=custom_ops, device=args.device)
    model_ori.to(args.device)
     
    if checkpoint is None:
        if args.manual_init:
            manual_init(args, model_ori, model, train_data)
        if args.kaiming_init:
            kaiming_init(model_ori)

    model_loss = model
    params = list(model_ori.parameters())
    logger.info('Parameter shapes: {}'.format([p.shape for p in params]))
    if args.multi_gpu:
        raise NotImplementedError('Multi-GPU is not supported yet')

    opt = get_optimizer(args, params, checkpoint)
    max_eps = args.eps or bound_config['eps']
    eps_scheduler = get_eps_scheduler(args, max_eps, train_data)
    lr_scheduler = get_lr_scheduler(args, opt)

    if epoch > 0 and not args.plot:
        # skip epochs
        eps_scheduler.train()
        for i in range(epoch):
            # FIXME Can use `last_epoch` argument of lr_scheduler
            lr_scheduler.step()
            eps_scheduler.step_epoch(verbose=False)

    if args.verify:
        start_time = time.time()
        logger.info('Inference')
        meter, log_dict = train_or_test(model, model_ori, 10000, test_data, eps_scheduler, None)
        logger.info(meter)
        timer = time.time() - start_time
    else:
        timer = 0.0
        for t in range(epoch + 1, args.num_epochs + 1):
            start_time = time.time()
            train_or_test(model, model_ori, t, train_data, eps_scheduler, opt)
            update_state_dict(model_ori, model_loss)
            epoch_time = time.time() - start_time
            timer += epoch_time
            lr_scheduler.step()
            if t % args.test_interval == 0:
                # Validation phase (performed on the target epsilon)
                with torch.no_grad():
                    meter, log_dict = train_or_test(model, model_ori, t, test_data, eps_scheduler, None)
                save(args, name_prefix, epoch=t, model=model_ori, opt=opt)

    # Log training data onto wandb for monitoring purposes.
    if args.wandb_label is not None:
        wandb.run.summary["runtime"] = timer
        wandb.run.summary["model_dir"] = os.path.join(args.dir, name_prefix) + "ckpt_last"
        wandb.run.summary["host_name"] = os.uname().nodename

    # location of the saved model (printed and saved to file)
    saved_model_dir = os.path.join(args.dir, name_prefix) + "ckpt_last"
    logger.info(f"Trained model checkpoint: {saved_model_dir}")
    with open("./trained_models_info.txt", "a") as file:
        string_summary = f"Model={saved_model_dir}, Dataset={args.config}"
        for k in log_dict:
            string_summary += f", {k}={log_dict[k]}"
        file.write(string_summary + "\n")


if __name__ == '__main__':
    main(args)
