#!/usr/bin/env/python
"""
@author: ycdu
@mail: ycdu666@gmail.com
@IDE: PyCharm
@file: HMRCNN.py
@time: 2020/09/10 7:47 下午
@desc: 
"""
from collections import OrderedDict
from math import floor

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import xavier_uniform_ as xavier_uniform

from models.modules.base_module import WordRep
from models.modules.residual_module import ResUnit
from utils.data_helper import get_loss


class HMRCNN(nn.Module):
    def __init__(self, args, Y, dicts):
        super(HMRCNN, self).__init__()

        # dropout
        self.embed_drop = nn.Dropout(p=args.embed_dropout)
        self.conv_drop = args.conv_dropout
        self.att_drop = nn.Dropout(p=args.att_dropout)

        # 词表示
        self.word_rep = WordRep(args, Y, dicts)

        # 卷积层
        self.conv = nn.ModuleList()
        filter_sizes = args.filter_size.split(',')
        self.filter_num = len(filter_sizes)
        for filter_size in filter_sizes:
            filter_size = int(filter_size)
            one_channel = nn.ModuleList()
            tmp = nn.Conv1d(self.word_rep.feature_size, self.word_rep.feature_size, kernel_size=filter_size,
                            padding=int(floor(filter_size / 2)))
            xavier_uniform(tmp.weight)
            one_channel.add_module('baseconv', tmp)

            conv_dimension = self.word_rep.conv_dict[args.conv_layer]
            for idx in range(args.conv_layer):
                tmp = ResUnit(conv_dimension[idx], conv_dimension[idx + 1], filter_size, 1, True, self.conv_drop)
                one_channel.add_module('resconv-{}'.format(idx), tmp)
            self.conv.add_module('channel-{}'.format(filter_size), one_channel)

        # 注意力层+输出
        self.level = args.level
        self.n_labels = []
        for level in range(1, int(self.level)):
            k = 'ind2l' + str(level + 1)
            self.n_labels.append(len(dicts[k]))
        self.n_labels.append(Y)
        self.att_size = args.att_size  # attention 中间维度
        self.level_proj_size = args.level_proj_size
        self.projectors = nn.ModuleList(
            [nn.Linear(self.n_labels[level], self.level_proj_size[level], bias=False) for level in range(self.level)])
        self.W = nn.ModuleList(
            [nn.Linear(self.filter_num * args.num_filter_maps, args.att_size, bias=False) for _ in range(self.level)])
        self.U = nn.ModuleList(
            [nn.Linear(args.att_size, self.n_labels[level], bias=False) for level in range(self.level)])
        self.final = nn.ModuleList(
            [nn.Linear(self.filter_num * args.num_filter_maps + int(self.level_proj_size[level-1] if level > 0 else 0),
                       self.n_labels[level], bias=True) for level in range(self.level)])

        self.loss_function = [nn.BCEWithLogitsLoss() for _ in range(self.level)]

        self.init_weight()

    def init_weight(self):
        for w in self.W:
            xavier_uniform(w.weight)
        for u in self.U:
            xavier_uniform(u.weight)
        for f in self.final:
            xavier_uniform(f.weight)
        for p in self.projectors:
            xavier_uniform(p.weight)

    def freeze_net(self):
        for p in self.word_rep.embed.parameters():
            p.requires_grad = False

    def multi_res_conv(self, x):
        x = self.word_rep(x)
        x = x.transpose(1, 2)

        conv_result = []
        for conv in self.conv:
            tmp = x
            for idx, md in enumerate(conv):
                if idx == 0:
                    tmp = torch.tanh(md(tmp))
                else:
                    tmp = md(tmp)
            tmp = tmp.transpose(1, 2)
            conv_result.append(tmp)
        x = torch.cat(conv_result, dim=2)
        return x

    def hie_att(self, x):
        p = None  # pre_level_projection
        weighted_outputs, att_weights, att_rep, att_rep1 = [], [], [], []
        for level in range(self.level):
            z = torch.tanh(self.W[level](x))
            uz = self.U[level](z)
            alpha = F.softmax(uz.transpose(1, 2), dim=2)
            # alpha = self.att_drop(alpha)
            m = alpha.matmul(x)
            att_rep.append(m)
            if p is not None:
                m = torch.cat((m, p.unsqueeze(1).repeat(1, m.shape[1], 1)), dim=2)
            att_rep1.append(m)

            y = self.final[level].weight.mul(m).sum(dim=2).add(self.final[level].bias)

            p = self.projectors[level](torch.sigmoid(y))
            p = torch.sigmoid(p)
            weighted_outputs.append(y)
            att_weights.append(alpha)
        return weighted_outputs, att_weights, att_rep, att_rep1

    def forward(self, x, target, level1, level2, level3, text_inputs=None):
        # pass
        if self.level == 2:
            true_labels = [level1, target]
        elif self.level == 3:
            true_labels = [level2, level3, target]
        elif self.level == 4:
            true_labels = [level1, level2, level3, target]
        loss_list = []
        x = self.multi_res_conv(x)
        y, _, att_rep, att_rep1 = self.hie_att(x)
        for level in range(self.level):
            pred = y[level]
            loss_list.append(self.loss_function[level](pred, true_labels[level]))
        if self.level == 2:
            loss = get_loss(loss_list, [len(level1), len(target)])
        if self.level == 3:
            loss = get_loss(loss_list, [len(level2), len(level3), len(target)])
        elif self.level == 4:
            loss = get_loss(loss_list, [len(level1), len(level2), len(level3), len(target)])
        return y, loss
