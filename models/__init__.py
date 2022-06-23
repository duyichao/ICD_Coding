#!/usr/bin/env/python
"""
@author: ycdu
@mail: ycdu666@gmail.com
@IDE: PyCharm
@file: __init__.py.py
@time: 2020/9/1 3:18 下午
@desc: 
"""
import torch

from .HMRCNN import HMRCNN


def pick_model(args, dicts, label_inp_idx, label_inp_mask):
    Y = len(dicts['ind2c'])
    if args.model == 'HMRCNN':
        model = HMRCNN(args, Y, dicts)
    else:
        raise RuntimeError("wrong model name")

    if args.test_model:
        sd = torch.load(args.test_model)
        model.load_state_dict(sd)
    if args.gpu >= 0:
        model.cuda(args.gpu)
    return model
