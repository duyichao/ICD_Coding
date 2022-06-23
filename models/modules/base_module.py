#!/usr/bin/env/python
"""
@author: ycdu
@mail: ycdu666@gmail.com
@IDE: PyCharm
@file: base_module.py
@time: 2020/9/2 2:28 下午
@desc:
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_ as xavier_uniform

from utils.data_helper import load_embeddings


class WordRep(nn.Module):
    def __init__(self, args, Y, dicts):
        super(WordRep, self).__init__()

        self.gpu = args.gpu

        if args.embed_file:
            args.logger.info("loading pretrained embeddings from {}".format(args.embed_file))
            W = torch.Tensor(load_embeddings(args.embed_file))
            self.embed = nn.Embedding(W.size()[0], W.size()[1], padding_idx=0)
            self.embed.weight.data = W.clone()
        else:
            # add 2 to include UNK and PAD
            self.embed = nn.Embedding(len(dicts['w2ind']) + 2, args.embed_size, padding_idx=0)
        self.feature_size = self.embed.embedding_dim

        self.embed_drop = nn.Dropout(p=args.dropout)

        self.conv_dict = {1: [self.feature_size, args.num_filter_maps]}

    def forward(self, x, target=None, text_inputs=None):
        """
        :param x: inputs_ids, list, batch*max_len
        :param target: labels
        :param text_inputs: text_inputs
        :return: x, shape:batch_size*max_len*embed_dim 16*2500*100
        """
        # 通过one hot向量对input_ids进行嵌入
        features = [self.embed(x)]
        x = torch.cat(features, dim=2)
        x = self.embed_drop(x)
        return x


class OutputLayer(nn.Module):
    def __init__(self, args, Y, dicts, input_size):
        super(OutputLayer, self).__init__()

        self.level = args.level

        self.U = nn.Linear(input_size, Y)
        self.final = nn.Linear(input_size, Y)
        self.loss_function = nn.BCEWithLogitsLoss()

        self.init_weight()

    def init_weight(self):
        xavier_uniform(self.U.weight)
        xavier_uniform(self.final.weight)

    def forward(self, x, target, text_inputs=None):
        # U:8922*300 x:16*2500*300
        alpha = F.softmax(self.U.weight.matmul(x.transpose(1, 2)), dim=2)
        m = alpha.matmul(x)
        y = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)
        loss = self.loss_function(y, target)
        return y, loss


class CodeWiseAttention(nn.Module):
    def __init__(self):
        super(CodeWiseAttention, self).__init__()

    def forward(self, x, label_feature):
        """
        :param x: batch*max_len*embed_size  2500*100
        :param label_feature: len(labels)*embed_size 8922*100
        :return:
        """
        alpha = F.softmax(label_feature.matmul(x.transpose(1, 2)), dim=2)
        # 8922*2500->8922*100
        m = alpha.matmul(x)

        return m

