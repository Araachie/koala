import argparse
import json
import random

import numpy as np

import torch
import torch.nn as nn

from torchvision import datasets
from torchvision import transforms

from tensorboard_logger import configure

from models import models
from optimizers import optimizers
from training_funcs import train, validate, lr_multiplier_functor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, help='Experiment name', default='AUTO')
    parser.add_argument('--model', type=str, help='Model name: resnet18', default='resnet18_cifar')
    parser.add_argument('--optim', type=str, help='Optimizer name: adagrad, sgd, koala-v/m...',
                        choices=list(optimizers.keys()), required=True)
    parser.add_argument('--data', type=str, help='Dataset to run the experiment on', default='cifar100')
    parser.add_argument('--batch-size', type=int, help='Batch size', default=128)
    parser.add_argument('--num-gpus', type=int, help='Number of gpus', default=1)
    parser.add_argument('--num-epochs', type=int, help='Number of epochs', default=100)
    parser.add_argument('--saving_freq', type=int, help='Frequency of model checkpoints', default=100)
    parser.add_argument('--resume-network', type=str, help='Path to model checkpoint to resume', default=None)
    parser.add_argument('--resume-epoch', type=int, help='Starting with epoch', default=None)
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--data-dir', type=str, default='./data', help='Dataset location')
    parser.add_argument('--warmup-epochs', type=int, help='Warmup epochs, set to 0 to disable', default=0)
    parser.add_argument('--scheduler', type=str, default='step', choices=['none', 'step'], help='lr scheduler type')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate for non-koala optimizers (note that the default is for SGD not Adam)')
    parser.add_argument('--momentum', type=float, default=0.9, help="SGD's momentum")
    parser.add_argument('--step-gamma', type=float, default=0.2, help="step scheduler's gamma (lr*=gamma)")
    parser.add_argument('--seed', type=int, help='random seed', default=42)
    # KOALA specific args
    parser.add_argument('--r', type=float, help='None for adaptive', default=None)
    parser.add_argument('--sw', type=float, default=0.1)
    parser.add_argument('--sv', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=0.9)
    parser.add_argument('--sigma', type=float, default=0.1)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.exp == 'AUTO':
        args.exp = f'{args.model}_{args.optim}_{args.data}_{args.batch_size}_{args.num_epochs}_' \
                   f'{args.weight_decay}_{args.warmup_epochs}_{args.scheduler}_{args.lr}_{args.momentum}_' \
                   f'{args.step_gamma}_{args.seed}_{args.r}_{args.sw}_{args.sv}_' \
                   f'{args.alpha}_{args.sigma}'
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    configure('runs/{}'.format(args.exp))

    with open(f'runs/{args.exp}/args.json', 'wt') as f:
        json.dump(vars(args), f, indent=2)

    # Configure dataset
    if args.data == 'cifar10':
        normalize = transforms.Normalize(
            mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
            std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    elif args.data == 'cifar100':
        normalize = transforms.Normalize(
            mean=[x / 255.0 for x in [129.3, 124.1, 112.4]],
            std=[x / 255.0 for x in [68.2, 65.4, 70.4]])
    else:
        raise NotImplementedError()

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    if args.data == 'cifar10':
        num_classes = 10
        dataset_cls = datasets.CIFAR10
    elif args.data == 'cifar100':
        num_classes = 100
        dataset_cls = datasets.CIFAR100
    else:
        raise NotImplementedError()
    train_loader = torch.utils.data.DataLoader(
        dataset_cls(args.data_dir, train=True, download=True, transform=transform_train),
        batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        dataset_cls(args.data_dir, train=False, transform=transform_test),
        batch_size=args.batch_size * 2, shuffle=True, num_workers=4, pin_memory=True)

    # Configure model
    model = models[args.model](num_classes=num_classes)
    torch.backends.cudnn.benchmark = True
    model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpus))).cuda()
    if args.resume_network is not None:
        model.load_state_dict(torch.load(args.resume_network))

    # Manage extra params
    is_koala = args.optim.startswith('koala')
    extra_params = dict()
    if is_koala:
        extra_params['r'] = args.r
        if args.optim == 'koala-v':
            extra_params['sigma'] = args.sigma
        else:
            extra_params['sw'] = args.sw
            extra_params['sv'] = args.sv
            extra_params['a'] = args.alpha
    else:
        extra_params['lr'] = args.lr
        if args.optim == 'sgd':
            extra_params['momentum'] = args.momentum

    # Setup optimizer
    optimizer = optimizers[args.optim](
        model.parameters(),
        weight_decay=args.weight_decay,
        **extra_params)

    # Setup scheduler
    if args.scheduler == 'none':
        milestones = tuple()
    else:
        if args.num_epochs == 100:
            milestones = (30, 60, 90)
        elif args.data == 'cifar10' and args.num_epochs == 200:
            milestones = (150,)
        elif args.data == 'cifar100' and args.num_epochs == 200:
            milestones = (60, 120, 160)
        else:
            raise NotImplementedError()
    calculate_lr = lr_multiplier_functor(len(train_loader), base_lr=1.0 if args.optim.startswith('koala') else args.lr,
                                         warmup_iters=args.warmup_epochs * len(train_loader), milestones=milestones,
                                         gamma=args.step_gamma)

    # Configure criterion
    criterion = nn.CrossEntropyLoss(reduction='none').cuda()

    # Start training
    start_epoch = 0
    if args.resume_epoch is not None:
        start_epoch = args.resume_epoch + 1

    for epoch in range(start_epoch, args.num_epochs):
        # Train for one epoch
        train(train_loader, model, criterion, optimizer, epoch,
              is_koala=is_koala, num_epochs=args.num_epochs, calculate_lr=calculate_lr)

        # Evaluate on validation set
        err1, err5 = validate(val_loader, model, criterion, epoch, num_epochs=args.num_epochs)

        if epoch % args.saving_freq == 0:
            torch.save(model.state_dict(), 'runs/{}/model_ckpt_{}'.format(args.exp, epoch))

    torch.save(model.state_dict(), 'runs/{}/model_final'.format(args.exp))

    print("Validation top1 error:", err1)
    print("Validation top5 error:", err5)


if __name__ == "__main__":
    main()
