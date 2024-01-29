# -*- coding: utf-8 -*-
# @Author  : jacob xu
# @Time    : 2023/7/15 16:48
# @File    : arg_parse.py
# @Software: PyCharm

from argparse import ArgumentParser


def args_parse( default=False ):
    """Command-line argument parser for train."""

    parser = ArgumentParser(
        description='PyTorch implementation of protonet of gout comorbidities  '
    )

        # 主要运行参数
    # parser.add_argument("data", metavar="DIR", help="path to dataset")
    parser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.4)
    parser.add_argument('--update_step', type=int, help='task-level inner update steps', default=3)

    parser.add_argument(
        "-j",
        "--workers",
        default=32,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 32)",
    )
    parser.add_argument(
        "--epochs", default=200, type=int, metavar="N", help="number of total epochs to run"
    )
    parser.add_argument(
        "--start-epoch",
        default=0,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=256,
        type=int,
        metavar="N",
        help="mini-batch size (default: 256), this is the total "
        "batch size of all GPUs on the current node when "
        "using Data Parallel or Distributed Data Parallel",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.15,
        type=float,
        metavar="LR",
        help="initial learning rate",
        dest="lr",
    )
    parser.add_argument(
        "--schedule",
        default=[120, 160],
        nargs="*",
        type=int,
        help="learning rate schedule (when to drop lr by 10x)",
    )
    parser.add_argument(
        "--momentum", default=0.2, type=float, metavar="M", help="momentum of SGD solver"
    )
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "-p",
        "--print-freq",
        default=10,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--world-size",
        default=-1,
        type=int,
        help="number of nodes for distributed training",
    )
    # parser.add_argument(
    #     "--rank", default=-1, type=int, help="node rank for distributed training"
    # )
    # parser.add_argument(
    #     "--dist-url",
    #     default="tcp://224.66.41.62:23456",
    #     type=str,
    #     help="url used to set up distributed training",
    # )
    # parser.add_argument(
    #     "--dist-backend", default="nccl", type=str, help="distributed backend"
    # )
    parser.add_argument(
        "--seed", default=42, type=int, help="seed for initializing training. "
    )
    parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
    parser.add_argument(
        "--multiprocessing-distributed",
        action="store_true",
        help="Use multi-processing distributed training to launch "
        "N processes per node, which has N GPUs. This is the "
        "fastest way to use PyTorch for either single node or "
        "multi node data parallel training",
    )

    # moco specific configs:
    parser.add_argument(
        "--moco-dim", default=128, type=int, help="feature dimension (default: 128)"
    )
    parser.add_argument(
        "--moco-k",
        default=65536,
        type=int,
        help="queue size; number of negative keys (default: 65536)",
    )

    # 动量encoder的设置，这里的设置相当于当前的输出和上一个的输出的比例是0.999，当前的模型的状态只有0.001的影响
    parser.add_argument(
        "--moco-m",
        default=0.999,
        type=float,
        help="moco momentum of updating key encoder (default: 0.999)",
    )

    # 这里是loss的温度
    parser.add_argument(
        "--moco-t", default=0.07, type=float, help="softmax temperature (default: 0.07)"
    )

    # options for moco v2
    parser.add_argument("--mlp", action="store_true", help="use mlp head")
    parser.add_argument(
        "--aug-plus", action="store_true", help="use moco v2 data augmentation"
    )
    parser.add_argument("--cos", action="store_true", help="use cosine lr schedule")

    return parser.parse_args()
