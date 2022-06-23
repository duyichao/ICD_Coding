#!/usr/bin/env/python
"""
@author: ycdu
@mail: ycdu666@gmail.com
@IDE: PyCharm
@file: data_helper.py
@time: 2020/9/1 11:42 下午
@desc:
"""
import csv

import json
import numpy as np
import torch
from allennlp.modules.elmo import batch_to_ids
from collections import defaultdict

import scipy.sparse as sp

from torch.utils.data import Dataset

from utils.options import args


def load_embeddings(embed_file):
    W = []
    with open(embed_file) as ef:
        for line in ef:
            line = line.rstrip().split()
            vec = np.array(line[1:]).astype(np.float)
            vec = vec / float(np.linalg.norm(vec) + 1e-6)
            W.append(vec)
        # UNK embedding, gaussian randomly initialized
        print("adding unk embedding")
        vec = np.random.randn(len(W[-1]))
        vec = vec / float(np.linalg.norm(vec) + 1e-6)
        W.append(vec)
    W = np.array(W)
    return W


def load_vocab_dict(args, vocab_file):
    vocab = set()

    with open(vocab_file, 'r') as vocabfile:
        for i, line in enumerate(vocabfile):
            line = line.rstrip()
            if line != '':
                vocab.add(line.strip())

    ind2w = {i + 1: w for i, w in enumerate(sorted(vocab))}
    w2ind = {w: i for i, w in ind2w.items()}

    return ind2w, w2ind


def load_full_codes(train_path, mimic2_dir, version='mimic3'):
    if version == 'mimic2':
        ind2c = defaultdict(str)
        codes = set()
        with open(mimic2_dir, 'r') as f:
            r = csv.reader(f)
            next(r)
            for row in r:
                codes.update(set(row[-1].split(';')))
        codes = set([c for c in codes if c != ''])
        ind2c = defaultdict(str, {i: c for i, c in enumerate(sorted(codes))})
    else:
        codes = set()
        for split in ['train', 'dev', 'test']:
            with open(train_path.replace('train', split), 'r') as f:
                lr = csv.reader(f)
                next(lr)
                for row in lr:
                    for code in row[3].split(';'):
                        codes.add(code)
        codes = set([c for c in codes if c != ''])
        ind2c = defaultdict(str, {i: c for i, c in enumerate(sorted(codes))})
    return ind2c


def load_label_index(label_index_file):
    f = open(label_index_file, 'r')
    lines = f.readlines()
    f.close()
    id2c = dict()
    c2id = dict()
    for line in lines:
        line = line.rstrip().split('\t')
        id2c[int(line[0])] = line[1]
        c2id[line[1]] = int(line[0])
    return c2id, id2c


def prepare_instance_hie(dicts, filename, args, max_length):
    ind2w, w2ind, ind2c, c2ind = dicts['ind2w'], dicts['w2ind'], dicts['ind2c'], dicts['c2ind']
    instances = []
    num_labels = len(dicts['ind2c'])

    def transform(label_idx, label, c2idx):
        for l in label:
            if l in c2idx.keys():
                code = int(c2idx[l])
                label_idx[code] = 1
        return label_idx

    with open(filename, 'r') as json_file:
        rows = json_file.readlines()
        if args.level == 2:
            ind2l1, l2ind1 = dicts['ind2l1'], dicts['l2ind1']
            for row in rows:
                data = json.loads(row)
                level1_idx = transform(np.zeros(len(ind2l1)), data['level1'], l2ind1)
                level2_idx = transform(np.zeros(num_labels), data['level2'], c2ind)
                labels_idx = [level1_idx, level2_idx]
                tokens_ = data['TEXT']
                tokens_id, tokens = [], []
                for token in tokens_:
                    if token == '[CLS]' or token == '[SEP]':
                        continue
                    tokens.append(token)
                    token_id = w2ind[token] if token in w2ind else len(w2ind) + 1
                    tokens_id.append(token_id)

                if len(tokens) > max_length:
                    tokens = tokens[:max_length]
                    tokens_id = tokens_id[:max_length]

                dict_instance = {'level1': level1_idx,
                                 'label': level2_idx,
                                 # 'tokens': tokens,
                                 "tokens_id": tokens_id}
                instances.append(dict_instance)
        elif args.level == 4:
            ind2l1, l2ind1 = dicts['ind2l1'], dicts['l2ind1']
            ind2l2, l2ind2 = dicts['ind2l2'], dicts['l2ind2']
            ind2l3, l2ind3 = dicts['ind2l3'], dicts['l2ind3']
            for row in rows:
                data = json.loads(row)
                level1_idx = transform(np.zeros(len(ind2l1)), data['level1'], l2ind1)
                level2_idx = transform(np.zeros(len(ind2l2)), data['level2'], l2ind2)
                level3_idx = transform(np.zeros(len(ind2l3)), data['level3'], l2ind3)
                level4_idx = transform(np.zeros(num_labels), data['level4'], c2ind)
                labels_idx = [level1_idx, level2_idx, level3_idx, level4_idx]
                tokens_ = data['TEXT']
                tokens_id, tokens = [], []
                for token in tokens_:
                    if token == '[CLS]' or token == '[SEP]':
                        continue
                    tokens.append(token)
                    token_id = w2ind[token] if token in w2ind else len(w2ind) + 1
                    tokens_id.append(token_id)

                if len(tokens) > max_length:
                    tokens = tokens[:max_length]
                    tokens_id = tokens_id[:max_length]

                dict_instance = {'level1': level1_idx,
                                 'level2': level2_idx,
                                 'level3': level3_idx,
                                 'label': level4_idx,
                                 # 'tokens': tokens,
                                 "tokens_id": tokens_id}
                instances.append(dict_instance)
        elif args.level == 3:
            ind2l1, l2ind1 = dicts['ind2l1'], dicts['l2ind1']
            ind2l2, l2ind2 = dicts['ind2l2'], dicts['l2ind2']
            ind2l3, l2ind3 = dicts['ind2l3'], dicts['l2ind3']
            for row in rows:
                data = json.loads(row)
                level1_idx = transform(np.zeros(len(ind2l1)), data['level1'], l2ind1)
                level2_idx = transform(np.zeros(len(ind2l2)), data['level2'], l2ind2)
                level3_idx = transform(np.zeros(len(ind2l3)), data['level3'], l2ind3)
                level4_idx = transform(np.zeros(num_labels), data['level4'], c2ind)
                tokens_ = data['TEXT']
                tokens_id, tokens = [], []
                for token in tokens_:
                    if token == '[CLS]' or token == '[SEP]':
                        continue
                    tokens.append(token)
                    token_id = w2ind[token] if token in w2ind else len(w2ind) + 1
                    tokens_id.append(token_id)

                if len(tokens) > max_length:
                    tokens = tokens[:max_length]
                    tokens_id = tokens_id[:max_length]

                dict_instance = {
                    # 'level1': level1_idx,
                    'level2': level2_idx,
                    'level3': level3_idx,
                    'label': level4_idx,
                    # 'tokens': tokens,
                    "tokens_id": tokens_id}
                instances.append(dict_instance)

    return instances


def load_label_inputs_hie(args, dicts):
    """
    按照层次结构加载label的文本以及mask
    :param args:
    :param dicts:
    :return:
    """
    label_desc_inputs_list, label_desc_mask_list = [], []
    keys = ['ind2c']
    for level in list(range(args.level - 1))[::-1]:
        keys.insert(0, 'ind2l' + str(level + 1))
    for key in keys:
        label_index = list(dicts[key].values())
        f = open(args.label_desc_path, 'r')
        lines = f.readlines()[1:]
        f.close()
        label_desc = {line.rstrip().split('\t')[0]: line.rstrip().split('\t')[1].split(' ') for line in lines}
        label_desc_inputs, label_desc_mask = [], []
        max_len = 0
        len_all = 0
        for ind in label_index:
            # TODO  label index
            if ind[:3] == ind[-3:] and ind not in ['00-00']:
                ind = ind[:3]
            elif '-' in ind:
                if str(ind).startswith('E'):
                    ind = ind + '.9'
                else:
                    ind = ind + '.99'
            tmp = [dicts['w2ind'][item] if item in dicts['w2ind'].keys() else len(list(dicts['w2ind'].keys())) + 1
                   for item in label_desc[ind]]
            label_desc_inputs.append(tmp)
            label_desc_mask.append(np.ones(len(tmp)))
            if len(tmp) > max_len:
                max_len = len(tmp)
            len_all += len(tmp)
        label_desc_inputs = torch.LongTensor(pad_sequence(label_desc_inputs, max_len)).to(args.device)
        label_desc_mask = torch.FloatTensor(pad_sequence(label_desc_mask, max_len)).to(args.device)
        label_desc_inputs_list.append(label_desc_inputs)
        label_desc_mask_list.append(label_desc_mask)

    return label_desc_inputs_list, label_desc_mask_list


#     return label_desc_inputs.to(args.device), label_desc_mask.to(args.device)

def load_lookups_hie(args):
    ind2w, w2ind = load_vocab_dict(args, args.vocab)
    ind2c_list = load_full_codes_hie(args.data_path, '%s/proc_dsums.csv' % args.MIMIC_2_DIR, args,
                                     version=args.version)
    if args.level == 4:
        ind2l1, ind2l2, ind2l3, ind2c = ind2c_list[0], ind2c_list[1], ind2c_list[2], ind2c_list[3]
        l2ind1 = {c: i for i, c in ind2l1.items()}
        l2ind2 = {c: i for i, c in ind2l2.items()}
        l2ind3 = {c: i for i, c in ind2l3.items()}
        c2ind = {c: i for i, c in ind2c.items()}
        dicts = {'ind2w': ind2w, 'w2ind': w2ind,
                 'ind2l1': ind2l1, 'l2ind1': l2ind1,
                 'l2ind2': l2ind2, 'ind2l2': ind2l2,
                 'l2ind3': l2ind3, 'ind2l3': ind2l3,
                 'ind2c': ind2c, 'c2ind': c2ind,
                 }
    elif args.level == 3:
        ind2l1, ind2l2, ind2l3, ind2c = ind2c_list[0], ind2c_list[1], ind2c_list[2], ind2c_list[3]
        l2ind1 = {c: i for i, c in ind2l1.items()}
        l2ind2 = {c: i for i, c in ind2l2.items()}
        l2ind3 = {c: i for i, c in ind2l3.items()}
        c2ind = {c: i for i, c in ind2c.items()}
        dicts = {'ind2w': ind2w, 'w2ind': w2ind,
                 'ind2l1': ind2l1, 'l2ind1': l2ind1,
                 'l2ind2': l2ind2, 'ind2l2': ind2l2,
                 'l2ind3': l2ind3, 'ind2l3': ind2l3,
                 'ind2c': ind2c, 'c2ind': c2ind,
                 }
    else:
        ind2l1, ind2c = ind2c_list[0], ind2c_list[1]
        l2ind1 = {c: i for i, c in ind2l1.items()}
        c2ind = {c: i for i, c in ind2c.items()}
        dicts = {'ind2w': ind2w, 'w2ind': w2ind,
                 'ind2l1': ind2l1, 'l2ind1': l2ind1,
                 'ind2c': ind2c, 'c2ind': c2ind,
                 }


    return dicts


def load_full_codes_hie(train_path, mimic2_dir, args, version='mimic3'):
    if version == 'mimic2':
        ind2c = defaultdict(str)
        codes = set()
        with open(mimic2_dir, 'r') as f:
            r = csv.reader(f)
            # header
            next(r)
            for row in r:
                codes.update(set(row[-1].split(';')))
        codes = set([c for c in codes if c != ''])
        ind2c = defaultdict(str, {i: c for i, c in enumerate(sorted(codes))})
    else:
        codes1, codes2, codes3, codes = set(), set(), set(), set()
        for split in ['train', 'dev', 'test']:
            with open(train_path.replace('train', split), 'r') as f:
                lines = f.readlines()
                if args.level == 4:
                    for line in lines:
                        line = json.loads(line)
                        codes1 = codes1.union(set(line['level1']))
                        codes2 = codes2.union(set(line['level2']))
                        codes3 = codes3.union(set(line['level3']))
                        codes = codes.union(set(line['level4']))
                    ind2c1 = defaultdict(str, {i: c for i, c in enumerate(sorted(codes1))})
                    ind2c2 = defaultdict(str, {i: c for i, c in enumerate(sorted(codes2))})
                    ind2c3 = defaultdict(str, {i: c for i, c in enumerate(sorted(codes3))})
                    ind2c = defaultdict(str, {i: c for i, c in enumerate(sorted(codes))})
                    ind2c_list = [ind2c1, ind2c2, ind2c3, ind2c]
                # elif args.level == 2:
                elif args.level == 3:
                    for line in lines:
                        line = json.loads(line)
                        codes1 = codes1.union(set(line['level1']))
                        codes2 = codes2.union(set(line['level2']))
                        codes3 = codes3.union(set(line['level3']))
                        codes = codes.union(set(line['level4']))
                    ind2c1 = defaultdict(str, {i: c for i, c in enumerate(sorted(codes1))})
                    ind2c2 = defaultdict(str, {i: c for i, c in enumerate(sorted(codes2))})
                    ind2c3 = defaultdict(str, {i: c for i, c in enumerate(sorted(codes3))})
                    ind2c = defaultdict(str, {i: c for i, c in enumerate(sorted(codes))})
                    # ind2c_list = [ind2c1, ind2c2, ind2c]
                    # ind2c_list = [ind2c2, ind2c3, ind2c]
                    ind2c_list = [ind2c1, ind2c2, ind2c3, ind2c]
                else:
                    for line in lines:
                        line = json.loads(line)
                        codes1 = codes1.union(set(line['level1']))
                        codes = codes.union(set(line['level2']))
                    ind2c1 = defaultdict(str, {i: c for i, c in enumerate(sorted(codes1))})
                    ind2c = defaultdict(str, {i: c for i, c in enumerate(sorted(codes))})
                    ind2c_list = [ind2c1, ind2c]
    return ind2c_list


class MyDataset(Dataset):

    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]


def pad_sequence(x, max_len, type=np.int):
    padded_x = np.zeros((len(x), max_len), dtype=type)
    for i, row in enumerate(x):
        padded_x[i][:len(row)] = row

    return padded_x


def my_collate_hie(x):
    level = args.level
    words = [x_['tokens_id'] for x_ in x]

    seq_len = [len(w) for w in words]
    max_seq_len = max(seq_len)

    inputs_id = pad_sequence(words, max_seq_len)

    labels = [x_['label'] for x_ in x]
    if level == 2:
        level1 = [x_['level1'] for x_ in x]
        level2 = np.array([])
        level3 = np.array([])
        return inputs_id, labels, level1, level2, level3
    elif level == 3:
        level1 = np.array([])
        level2 = [x_['level2'] for x_ in x]
        level3 = [x_['level3'] for x_ in x]
        return inputs_id, labels, level1, level2, level3
    elif level == 4:
        level1 = [x_['level1'] for x_ in x]
        level2 = [x_['level2'] for x_ in x]
        level3 = [x_['level3'] for x_ in x]
        return inputs_id, labels, level1, level2, level3
    else:
        return inputs_id, labels

def get_loss(loss_list, n_training_labels):
    loss = None
    n_total_label = 0
    for i in range(len(loss_list)):
        n_label = n_training_labels[i]
        n_total_label += n_label
        if loss is None:
            loss = n_label * loss_list[i]
        else:
            loss += n_label * loss_list[i]
    return loss / n_total_label


#################################
# process adj matrix
#################################

def load_code_cooc(args, dicts: dict, to_sp=True, norm=True, to_tensor=True):
    cooc_adj = np.zeros((len(dicts['c2ind']), len(dicts['c2ind'])))
    for name in ['train', 'dev', 'test']:
        data_path = args.data_path.replace('train', name)
        # print(data_path)
        f = open(data_path, 'r')
        # lines = f.readlines()
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            labels = row[3].split(';')
            lids = [dicts['c2ind'][label] for label in labels]
            i, j = 0, 0
            while i < len(lids):
                # j += 1
                j = i + 1
                lid1 = lids[i]
                cooc_adj[lid1, lid1] += 1  # 自身出现次数
                while j < len(lids):
                    lid2 = lids[j]
                    # print(i, j, lid1, cooc_adj[lid1, lid1])
                    cooc_adj[lid1, lid2] += 1
                    cooc_adj[lid2, lid1] += 1
                    # print(lid1, lid2)
                    j += 1
                i += 1
    print(cooc_adj)
    if to_sp:
        cooc_adj = sp.csr_matrix(cooc_adj)
    if norm:
        cooc_adj = normalize_row(cooc_adj)
    if to_tensor:
        cooc_adj = sparse_mx_to_torch_sparse_tensor(cooc_adj)
    # exit(0)
    return cooc_adj


def normalize_row(mx, is_sparse=True):
    """Row-normalize sparse matrix."""
    if not is_sparse:
        mx = sp.coo_matrix(mx)
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_sys(mx, is_sparse=True):
    """Symmetrically normalize adjacency matrix."""
    if not is_sparse:
        mx = sp.coo_matrix(mx)
    rowsum = np.array(mx.sum(1))  # D
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()  # D^-0.5
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)  # D^-0.5
    return mx.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()  # D^-0.5AD^0.5


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.Tensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
