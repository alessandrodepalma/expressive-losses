# Expressive Losses for Verified Robustness via Convex Combinations

Code associated to
[Expressive Losses for Verified Robustness via Convex Combinations](https://openreview.net/pdf?id=mzyZ4wzKlM).

Expressive losses, defined as the ability of a loss to range from lower to upper bounds to the true robust loss, 
are all you need to attain state-of-the-art verified robustness. We show that expressive losses are easily defined via convex combinations.

If you use our code in your research, please cite:
```
@inproceedings{DePalma2024,
    title={Expressive Losses for Verified Robustness via Convex Combinations},
    author={De Palma, Alessandro and Bunel, Rudy and Dvijotham, Krishnamurthy and Kumar, M. Pawan and Stanforth, Robert and Lomuscio, Alessio},
    booktitle={International Conference on Learning Representations},
    year={2024},
}
```

## Available Losses

This repository contains code to train certifiably robust networks using the following expressive losses: 

- CC-IBP [(De Palma et al., 2024)](https://openreview.net/pdf?id=mzyZ4wzKlM), taking convex combinations between adversarial logit differences and IBP lower bounds to logit differences;
- MTL-IBP [(De Palma et al., 2024)](https://openreview.net/pdf?id=mzyZ4wzKlM), taking convex combinations between the adversarial loss and the IBP loss;
- Exp-IBP [(De Palma et al., 2024)](https://openreview.net/pdf?id=mzyZ4wzKlM), where the convex combinations are taken in the space of the logarithms of the adversarial loss and the IBP loss;
- SABR [(Mueller et al., 2023)](https://openreview.net/pdf?id=7oFuxtJtUMH), which computes IBP bounds over small subsets of the perturbation region, containing adversarial examples.

## Trained Models

| Dataset      | Model | Target L-inf Perturbation | Method  | Download Link |
|--------------|------------------ |---------------------------|---------|---------------|
| MNIST        | CNN7 | 0.1                       | CC-IBP  | [model](https://sail.doc.ic.ac.uk/data/expressive-losses-models-iclr24/mnist-0.1-ccibp.pt)     |
| MNIST        | CNN7 | 0.1                       | MTL-IBP | [model](https://sail.doc.ic.ac.uk/data/expressive-losses-models-iclr24/mnist-0.1-mtlibp.pt)             |
| MNIST        | CNN7 | 0.3                       | CC-IBP   | [model](https://sail.doc.ic.ac.uk/data/expressive-losses-models-iclr24/mnist-0.3-ccibp.pt)             |
| MNIST        | CNN7 | 0.3                       | MTL-IBP | [model](https://sail.doc.ic.ac.uk/data/expressive-losses-models-iclr24/mnist-0.3-mtlibp.pt)             |
| CIFAR-10     | CNN7 | 2/255                     | CC-IBP  | [model](https://sail.doc.ic.ac.uk/data/expressive-losses-models-iclr24/cifar10-2-255-ccibp.pt)             |
| CIFAR-10     | CNN7 | 2/255                     | MTL-IBP | [model](https://sail.doc.ic.ac.uk/data/expressive-losses-models-iclr24/cifar10-2-255-mtlibp.pt)             |
| CIFAR-10     | CNN7 | 2/255                     | Exp-IBP | [model](https://sail.doc.ic.ac.uk/data/expressive-losses-models-iclr24/cifar10-2-255-expibp.pt)             |
| CIFAR-10     | CNN7 | 8/255                     | CC-IBP  | [model](https://sail.doc.ic.ac.uk/data/expressive-losses-models-iclr24/cifar10-8-255-ccibp.pt)             |
| CIFAR-10     | CNN7 | 8/255                     | MTL-IBP | [model](https://sail.doc.ic.ac.uk/data/expressive-losses-models-iclr24/cifar10-8-255-mtlibp.pt)             |
| CIFAR-10     | CNN7 | 8/255                     | Exp-IBP | [model](https://sail.doc.ic.ac.uk/data/expressive-losses-models-iclr24/cifar10-8-255-expibp.pt)             |
| TinyImageNet | CNN7 | 1/255                     | CC-IBP  | [model](https://sail.doc.ic.ac.uk/data/expressive-losses-models-iclr24/tinyimagenet-1-255-ccibp.pt)             |
| TinyImageNet | CNN7 | 1/255                     | MTL-IBP | [model](https://sail.doc.ic.ac.uk/data/expressive-losses-models-iclr24/tinyimagenet-1-255-mtlibp.pt)             |
| TinyImageNet | CNN7 | 1/255                     | Exp-IBP | [model](https://sail.doc.ic.ac.uk/data/expressive-losses-models-iclr24/tinyimagenet-1-255-expibp.pt)             |
| ImageNet64   | CNN7 | 1/255                     | CC-IBP  | [model](https://sail.doc.ic.ac.uk/data/expressive-losses-models-iclr24/imagenet64-1-255-ccibp.pt)             |
| ImageNet64   | CNN7 | 1/255                     | MTL-IBP | [model](https://sail.doc.ic.ac.uk/data/expressive-losses-models-iclr24/imagenet64-1-255-mtlibp.pt)             |
| ImageNet64   | CNN7 | 1/255                     | Exp-IBP | [model](https://sail.doc.ic.ac.uk/data/expressive-losses-models-iclr24/imagenet64-1-255-expibp.pt)             |

The code to load the above models can be found within `utils.py` (in the `prepare_model` function). See for instance how it is used in `verify.py`.
MNIST and CIFAR-10 models require `--model=cnn`, TinyImageNet and ImageNet64 models require `--model=cnn_7layer_bn_imagenet`.

## Code Setup and Dependencies

We now list the steps necessary to install the code in this repository.

```
# Create a conda environment
conda create -y -n expressive-losses python=3.7
conda activate expressive-losses
```

### PyTorch
The codebase requires a working PyTorch (and `torchvision`) installation: see [PyTorch Get Started](https://pytorch.org/get-started/).
```
# Install pytorch to this virtualenv
# (or check updated or more suitable install instructions at https://pytorch.org)
conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch
```

### auto_LiRPA
The training pipeline relies on the IBP implementation within [auto_LiRPA](https://github.com/KaidiXu/auto_LiRPA). 
In order to install its source code, execute the following:
```
git clone https://github.com/KaidiXu/auto_LiRPA.git
cd auto_LiRPA
python setup.py install
cd ..
```

### OVAL BaB

In the paper, we employ the [OVAL](https://github.com/oval-group/oval-bab) verification framework to perform branch-and-bound-based 
verification of our trained models. Please install the verification framework by following the instructions in the relative repository.

### Installation

After satisfying the above dependencies, the code can be installed by executing the following:
```
pip install .
```

## Datasets

MNIST and CIFAR-10 are downloaded through `torchvision`.

In order to download TinyImageNet, use the following command:
```
cd data/tinyimagenet
bash tinyimagenet_download.sh
```

In order to run ImageNet, download the raw images (Train and Val, 64x64, npz format) from [Image-Net.org](http://image-net.org/download-images), 
under the "Download downsampled image data (32x32, 64x64)" section, to a folder of your choice (say `imagenet_folder_path`), decompress them and then run the following data preprocessing script:
```
cd data/ImageNet64
python imagenet_data_loader.py --folder imagenet_folder_path
```

## Reproducing Table 1

The logs from the trained model can be logged onto `wandb`, if an optional `--wandb_label` is provided 
(e.g., `--wandb_label test`).

During training, model checkpoints are saved in the directory indicated in the `--dir` argument, with a filename determined by the timestamp at the start of execution.
The full model path, along with the final model statistics, are appended to a text file (`trained_models_info.txt`), with one line per trained model.
*The model path (denoted `trained_model_path` in the commands below) will be needed to correctly run model evaluation (including verification).*

Note: if encountering memory errors, use gradient accumulation (`--grad-acc-steps`).
If your GPU relies on the TensorFloat-32 by default, please append `--disable_train_tf32` to your training commands to gain in numerical precision.

### MNIST - 0.1

```
mkdir -p model_mnist
# CC-IBP
python train.py --method=fast --dir=model_mnist --scheduler_opts=start=1,length=20 --lr-decay-milestones=50,60 --num-epochs=70 --config=config/mnist_0.1.json --model=cnn --train_att_n_steps 1 --train_att_step_size 10.0 --test_att_n_steps 40 --test_att_step_size 0.1 --ccibp --l1_coeff 2e-6 --ccibp_coeff 0.1 --test-interval 10 --train-eps-mul 2.0
# MTL-IBP
python train.py --method=fast --dir=model_mnist --scheduler_opts=start=1,length=20 --lr-decay-milestones=50,60 --num-epochs=70 --config=config/mnist_0.1.json --model=cnn --train_att_n_steps 1 --train_att_step_size 10.0 --test_att_n_steps 40 --test_att_step_size 0.1 --mtlibp --l1_coeff 1e-5 --mtlibp_coeff 1e-2 --test-interval 10 --train-eps-mul 2.0
# Model evaluation (and verification)
python verify.py --config=config/mnist_0.1.json --model=cnn --ib_batch_size 2000 --oval_bab_config ./bab_configs/cnn7_naive_noearly.json --oval_bab_timeout 1800  --load ./trained_model_path --test_att_n_steps 40 --test_att_step_size 0.035
```

### MNIST - 0.3

```
mkdir -p model_mnist
# CC-IBP
python train.py --method=fast --dir=model_mnist --scheduler_opts=start=1,length=20 --lr-decay-milestones=50,60 --num-epochs=70 --config=config/mnist.json --model=cnn --train_att_n_steps 1 --train_att_step_size 10.0 --test_att_n_steps 40 --test_att_step_size 0.1 --ccibp --l1_coeff 3e-6 --ccibp_coeff 0.3 --test-interval 10 --train-eps-mul 1.334
# MTL-IBP
python train.py --method=fast --dir=model_mnist --scheduler_opts=start=1,length=20 --lr-decay-milestones=50,60 --num-epochs=70 --config=config/mnist.json --model=cnn --train_att_n_steps 1 --train_att_step_size 10.0 --test_att_n_steps 40 --test_att_step_size 0.1 --mtlibp --l1_coeff 1e-6 --mtlibp_coeff 8e-2 --test-interval 10 --train-eps-mul 1.334
# Model evaluation (and verification)
python verify.py --config=config/mnist.json --model=cnn --ib_batch_size 2000 --oval_bab_config ./bab_configs/cnn7_naive_noearly.json --oval_bab_timeout 1800 --load ./trained_model_path --test_att_n_steps 40 --test_att_step_size 0.035
```

### CIFAR10 - 2/255


```
mkdir -p model_cifar
# CC-IBP
python train.py --method=fast --dir=model_cifar --scheduler_opts=start=2,length=80 --lr-decay-milestones=120,140 --num-epochs=160 --config=config/cifar_2_255.json --model=cnn --train_att_n_steps 8 --train_att_step_size 0.25 --test_att_n_steps 40 --test_att_step_size 0.1 --ccibp --l1_coeff 3e-6 --ccibp_coeff 1e-2 --attack_eps_factor 2.1 --test-interval 20
# MTL-IBP
python train.py --method=fast --dir=model_cifar --scheduler_opts=start=2,length=80 --lr-decay-milestones=120,140 --num-epochs=160 --config=config/cifar_2_255.json --model=cnn --train_att_n_steps 8 --train_att_step_size 0.25 --test_att_n_steps 40 --test_att_step_size 0.1 --mtlibp --l1_coeff 3e-6 --mtlibp_coeff 4e-3 --attack_eps_factor 2.1 --test-interval 20
# Exp-IBP
python train.py --method=fast --dir=model_cifar --scheduler_opts=start=2,length=80 --lr-decay-milestones=120,140 --num-epochs=160 --config=config/cifar_2_255.json --model=cnn --train_att_n_steps 8 --train_att_step_size 0.25 --test_att_n_steps 40 --test_att_step_size 0.1 --expibp --l1_coeff 4e-6 --expibp_coeff 9.5e-2 --attack_eps_factor 2.1 --test-interval 20
# Model evaluation (and verification)
python verify.py --config=config/cifar_2_255.json --model=cnn  --ib_batch_size 2000 --oval_bab_config ./bab_configs/cnn7_naive.json --oval_bab_timeout 1800 --load ./trained_model_path --test_att_n_steps 40 --test_att_step_size 0.035
```

### CIFAR10 - 8/255

```
mkdir -p model_cifar
# CC-IBP
python train.py --method=fast --dir=model_cifar --scheduler_opts=start=2,length=80 --lr-decay-milestones=180,220 --num-epochs=260 --config=config/cifar.json --model=cnn --train_att_n_steps 1 --train_att_step_size 10.0 --test_att_n_steps 40 --test_att_step_size 0.1 --ccibp --l1_coeff 0 --ccibp_coeff 0.5--test-interval 20
# MTL-IBP
python train.py --method=fast --dir=model_cifar --scheduler_opts=start=2,length=80 --lr-decay-milestones=180,220 --num-epochs=260 --config=config/cifar.json --model=cnn --train_att_n_steps 1 --train_att_step_size 10.0 --test_att_n_steps 40 --test_att_step_size 0.1 --mtlibp --l1_coeff 1e-7 --mtlibp_coeff 0.5 --test-interval 20
# Exp-IBP
python train.py --method=fast --dir=model_cifar --scheduler_opts=start=2,length=80 --lr-decay-milestones=180,220 --num-epochs=260 --config=config/cifar.json --model=cnn --train_att_n_steps 1 --train_att_step_size 10.0 --test_att_n_steps 40 --test_att_step_size 0.1 --expibp --l1_coeff 0 --expibp_coeff 0.5 --test-interval 20
# Model evaluation (and verification)
python verify.py --config=config/cifar.json --model=cnn --ib_batch_size 2000 --oval_bab_config ./bab_configs/cnn7_naive.json --oval_bab_timeout 1800 --load ./trained_model_path --test_att_n_steps 40 --test_att_step_size 0.035
```


### TinyImageNet - 1/255

```
mkdir -p model_tinyimagenet
# CC-IBP
python train.py --method=fast --dir=model_tinyimagenet --scheduler_opts=start=2,length=80 --reg-lambda=0.2 --lr-decay-milestones=120,140 --num-epochs=160 --num-class 200 --batch-size=128 --config=config/tinyimagenet.ibp.json --model=cnn_7layer_bn_imagenet --train_att_n_steps 1 --train_att_step_size 10.0 --test_att_n_steps 40 --test_att_step_size 0.1 --ccibp --l1_coeff 5e-5 --ccibp_coeff 1e-2 --test-interval 20
# MTL-IBP
python train.py --method=fast --dir=model_tinyimagenet --scheduler_opts=start=2,length=80 --reg-lambda=0.2 --lr-decay-milestones=120,140 --num-epochs=160 --num-class 200 --batch-size=128 --config=config/tinyimagenet.ibp.json --model=cnn_7layer_bn_imagenet --train_att_n_steps 1 --train_att_step_size 10.0 --test_att_n_steps 40 --test_att_step_size 0.1 --mtlibp --l1_coeff 5e-5 --mtlibp_coeff 1e-2 --test-interval 20
# Exp-IBP
python train.py --method=fast --dir=model_tinyimagenet --scheduler_opts=start=2,length=80 --reg-lambda=0.2 --lr-decay-milestones=120,140 --num-epochs=160 --num-class 200 --batch-size=128 --config=config/tinyimagenet.ibp.json --model=cnn_7layer_bn_imagenet --train_att_n_steps 1 --train_att_step_size 10.0 --test_att_n_steps 40 --test_att_step_size 0.1 --expibp --l1_coeff 5e-5 --expibp_coeff 4e-2 --test-interval 20
# Model evaluation (and verification)
python verify.py --config=config/tinyimagenet.ibp.json --model=cnn_7layer_bn_imagenet --ib_batch_size 2000 --oval_bab_config ./bab_configs/cnn7_naive_lessmem.json --oval_bab_timeout 1800 --load ./trained_model_path --test_att_n_steps 40 --test_att_step_size 0.035 --num-class 200
```

### ImageNet64 - 1/255

```
mkdir -p model_imagenet64
# CC-IBP
python train.py --method=fast --config=config/imagenet64.json --dir=model_imagenet64 --model=cnn_7layer_bn_imagenet --model-params=num_class=1000 --scheduler_opts=start=2,length=20 --reg-lambda=0.2 --lr-decay-milestones=60,70 --num-epochs=80 --num-class 1000 --batch-size=128 --train_att_n_steps 1 --train_att_step_size 10.0 --test_att_n_steps 40 --test_att_step_size 0.1 --ccibp --l1_coeff 1e-5 --ccibp_coeff 5e-2 --test-interval 5 --data_loader_workers 10
# MTL-IBP
python train.py --method=fast --config=config/imagenet64.json --dir=model_imagenet64 --model=cnn_7layer_bn_imagenet --model-params=num_class=1000 --scheduler_opts=start=2,length=20 --reg-lambda=0.2 --lr-decay-milestones=60,70 --num-epochs=80 --num-class 1000 --batch-size=128 --train_att_n_steps 1 --train_att_step_size 10.0 --test_att_n_steps 40 --test_att_step_size 0.1 --mtlibp --l1_coeff 1e-5 --mtlibp_coeff 5e-2 --test-interval 5 --data_loader_workers 10
# Exp-IBP
python train.py --method=fast --config=config/imagenet64.json --dir=model_imagenet64 --model=cnn_7layer_bn_imagenet --model-params=num_class=1000 --scheduler_opts=start=2,length=20 --reg-lambda=0.2 --lr-decay-milestones=60,70 --num-epochs=80 --num-class 1000 --batch-size=128 --train_att_n_steps 1 --train_att_step_size 10.0 --test_att_n_steps 40 --test_att_step_size 0.1 --expibp --l1_coeff 1e-5 --expibp_coeff 5e-2 --test-interval 5 --data_loader_workers 10
# Model evaluation (and verification). Decrease lirpa_crown_batch in case of memory issues.
python verify.py --config=config/imagenet64.json --model=cnn_7layer_bn_imagenet --model-params=num_class=1000 --ib_batch_size 2000 --oval_bab_config ./bab_configs/cnn7_naive_lessmem.json --oval_bab_timeout 1800 --load ./trained_model_path --test_att_n_steps 40 --test_att_step_size 0.035 --num-class 1000 --lirpa_crown_batch 700
```

## Other Experiments

The ablations can be reproduced by suitably editing the commands provided above: see section 7 and appendices F, G for the relative hyper-parameters.
On top of the default seed, we used seeds (`--seed`) 0, 1 and 2 for the repetitions in appendix G.8.

Losses for trained models can be computed by using `log_losses.py` as follows (Exp-IBP example on CIFAR-10 2/255):
```
python log_losses.py --config=config/cifar_2_255.json --model=cnn --ib_batch_size 2000 --train_att_n_steps 1 --train_att_step_size 10.0 --attack_eps_factor 2.1 --test_att_n_steps 40 --test_att_step_size 0.035 --load ./trained_model_path --expibp_coeff 3e-1
```

In order to use the validation split in the training/evaluation process for the experiments from section 6.3, the user needs to append `--valid_share 0.8`. 
We now provide examples for SABR, which satisfies our definition of expressivity, to also show how it can be run from our codebase (it can be adapted to CC/MTL/Exp-IBP using the commands above), first training then computing BaB loss and errors.
```
# SABR - 2/255 example
python train.py --method=fast --dir=model_cifar --scheduler_opts=start=2,length=80 --lr-decay-milestones=120,140 --num-epochs=160 --config=config/cifar_2_255.json --model=cnn --train_att_n_steps 8 --train_att_step_size 0.25 --test_att_n_steps 40 --test_att_step_size 0.1 --attack_eps_factor 2.1 --test-interval 20 --valid_share 0.8 --sabr --l1_coeff 3e-6 --sabr_coeff 3e-2
python get_tuning_stats.py --config=config/cifar_2_255.json --model=cnn --ib_batch_size 2000 --train_att_n_steps 8 --train_att_step_size 0.25 --attack_eps_factor 2.1 --test_att_n_steps 8 --test_att_step_size 0.25 --load ./trained_model_path --valid_share 0.8 --oval_bab_config ./bab_configs/cnn7_naive.json --oval_bab_timeout 15 --end_idx 100 --sabr --sabr_coeff 3e-2
# SABR - 8/255 example
python train.py --method=fast --dir=model_cifar --scheduler_opts=start=2,length=80 --lr-decay-milestones=180,220 --num-epochs=260 --config=config/cifar.json --model=cnn --train_att_n_steps 1 --train_att_step_size 10.0 --test_att_n_steps 40 --test_att_step_size 0.1 --test-interval 20 --valid_share 0.8 --l1_coeff 0 --sabr --sabr_coeff 0.6 
python get_tuning_stats.py --config=config/cifar.json --model=cnn --ib_batch_size 2000 --train_att_n_steps 1 --train_att_step_size 10.0 --test_att_n_steps 8 --test_att_step_size 0.25 --load ./trained_model_path --valid_share 0.8 --oval_bab_config ./bab_configs/cnn7_naive_noearly.json --oval_bab_timeout 15 --end_idx 100 --sabr --sabr_coeff 0.6
```

## Acknowledgments
We would like to thank the authors of [(Shi et al., 2021)](https://github.com/shizhouxing/Fast-Certified-Robust-Training) and [(Mueller et al., 2023)](https://github.com/eth-sri/SABR) for open-sourcing their code: we based our codebase upon the former repository, and re-implemented SABR from the latter.