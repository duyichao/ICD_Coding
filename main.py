#!/usr/bin/env/python
"""
@author: ycdu
@mail: ycdu666@gmail.com
@IDE: PyCharm
@file: main.py
@time: 2020/9/1 9:48 上午
@desc: 
"""

import csv
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import geoopt as gt
import torch
import torch.optim as optim
from texttable import Texttable
from prettytable import PrettyTable
from torch.utils.data import DataLoader

from adabelief_pytorch import AdaBelief
from ranger_adabelief import RangerAdaBelief

from models import pick_model
from trainer_hie import train, test
from utils.common import early_stop
from utils.data_helper import   MyDataset, \
    prepare_instance_hie, load_lookups_hie, my_collate_hie, load_label_inputs_hie
from utils.metrics import save_everything
from utils.options import args
from utils.logger import logger_func


# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


def main():
    logger = logger_func(args=args)
    args.logger = logger

    if args.random_seed != 0:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)

    if args.gpu != -1:
        args.device = torch.device('cuda:' + str(args.gpu) if int(args.gpu) >= 0 else 'cpu')
    logger.info('device:' + str(args.device))

    csv.field_size_limit(sys.maxsize)

    # 加载词典
    dicts = load_lookups_hie(args)
    label_desc_inputs, label_desc_mask = load_label_inputs_hie(args, dicts)

    # 加载模型
    model = pick_model(args, dicts, label_desc_inputs, label_desc_mask)
    model.to(device=args.device)
    logger.info('\n' + str(model))

    # 选择优化器
    betas = (0.9, 0.999)
    if args.test_model:
        optimizer = None
    elif 'hyp' in args.model:
        optimizer = gt.optim.RiemannianAdam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
    elif args.optimizer == 'AdaBelief':
        optimizer = AdaBelief(model.parameters(), lr=args.lr, eps=1e-12, betas=betas, weight_decay=args.weight_decay)
        # optimizer = RangerAdaBelief(model.parameters(), lr=args.lr, eps=1e-12, betas=betas,
        #                             weight_decay=args.weight_decay)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=betas, weight_decay=args.weight_decay)

    # 冻结训练词向量
    if not args.tune_wordemb:
        model.freeze_net()

    # 评价指标
    metrics_hist = defaultdict(lambda: [])
    metrics_hist_te = defaultdict(lambda: [])
    metrics_hist_tr = defaultdict(lambda: [])
    metrics_hist1 = defaultdict(lambda: [])
    metrics_hist_te1 = defaultdict(lambda: [])
    metrics_hist_tr1 = defaultdict(lambda: [])
    metrics_hist2 = defaultdict(lambda: [])
    metrics_hist_te2 = defaultdict(lambda: [])
    metrics_hist_tr2 = defaultdict(lambda: [])
    metrics_hist3 = defaultdict(lambda: [])
    metrics_hist_te3 = defaultdict(lambda: [])
    metrics_hist_tr3 = defaultdict(lambda: [])

    # 准备训练，验证，测试实例
    prepare_instance_func = prepare_instance_hie

    train_instances = prepare_instance_func(dicts, args.data_path, args, args.MAX_LENGTH)
    logger.info("train_instances {}".format(len(train_instances)))
    if args.version != 'mimic2':
        dev_instances = prepare_instance_func(dicts, args.data_path.replace('train', 'dev'), args, args.MAX_LENGTH)
        logger.info("dev_instances {}".format(len(dev_instances)))
    else:
        dev_instances = None
    test_instances = prepare_instance_func(dicts, args.data_path.replace('train', 'test'), args, args.MAX_LENGTH)
    logger.info("test_instances {}".format(len(test_instances)))

    # DataLoader
    collate_func = my_collate_hie
    train_loader = DataLoader(MyDataset(train_instances),
                              args.batch_size,
                              shuffle=True,
                              num_workers=args.loader_workers,
                              pin_memory=True,
                              collate_fn=collate_func)
    if args.version != 'mimic2':
        dev_loader = DataLoader(MyDataset(dev_instances), 1, shuffle=False, num_workers=args.loader_workers,
                                pin_memory=True,
                                collate_fn=collate_func)
    else:
        dev_loader = None
    test_loader = DataLoader(MyDataset(test_instances), 1, shuffle=False, num_workers=args.loader_workers,
                             pin_memory=True,
                             collate_fn=collate_func)

    test_only = args.test_model is not None

    for epoch in range(args.n_epochs):

        # 创建存储模型的文件夹
        if epoch == 0 and not args.test_model:
            model_dir = os.path.join(args.MODEL_DIR,
                                     '_'.join([args.model, time.strftime('%b_%d_%H_%M_%S', time.localtime())]))
            os.makedirs(model_dir)
        elif args.test_model:
            model_dir = os.path.dirname(os.path.abspath(args.test_model))

        # 训练模型
        if not test_only:
            epoch_start = time.time()
            losses = train(args, logger, model, optimizer, epoch, args.gpu, train_loader)
            loss = np.mean(losses)
            epoch_finish = time.time()
            logger.info("epoch finish in %.2fs, loss: %.4f" % (epoch_finish - epoch_start, loss))
        else:
            loss = np.nan

        # 验证集选择模型参数
        fold = 'test' if args.version == 'mimic2' else 'dev'
        dev_instances = test_instances if args.version == 'mimic2' else dev_instances
        dev_loader = test_loader if args.version == 'mimic2' else dev_loader
        if epoch == args.n_epochs - 1:
            logger.info("last epoch: testing on dev and test sets")
            test_only = True

        # test on dev
        evaluation_start = time.time()
        metrics = test(args, logger, model, args.data_path, fold, args.gpu, dicts, dev_loader)
        evaluation_finish = time.time()
        logger.info("evaluation finish in %.2fs" % (evaluation_finish - evaluation_start))
        if test_only or epoch == args.n_epochs - 1:
            metrics_te = test(args, logger, model, args.data_path, "test", args.gpu, dicts, test_loader)
        else:
            metrics_te = defaultdict(float)
        metrics_tr = {'loss': loss}
        metrics_all = (metrics[0], metrics_te, metrics_tr)

        for name in metrics_all[0].keys():
            metrics_hist[name].append(metrics_all[0][name])
        for name in metrics_all[1].keys():
            metrics_hist_te[name].append(metrics_all[1][name])
        for name in metrics_all[2].keys():
            metrics_hist_tr[name].append(metrics_all[2][name])
        metrics_hist_all = (metrics_hist, metrics_hist_te, metrics_hist_tr)

        save_everything(args, metrics_hist_all, model, model_dir, None, args.criterion, test_only, hie_level=0)

        metrics_all1 = (metrics[1], metrics_te, metrics_tr)
        for name in metrics_all1[0].keys():
            metrics_hist1[name].append(metrics_all1[0][name])
        for name in metrics_all1[1].keys():
            metrics_hist_te1[name].append(metrics_all1[1][name])
        for name in metrics_all1[2].keys():
            metrics_hist_tr1[name].append(metrics_all1[2][name])
        metrics_hist_all1 = (metrics_hist1, metrics_hist_te1, metrics_hist_tr1)

        save_everything(args, metrics_hist_all1, model, model_dir, None, args.criterion, test_only, hie_level=1)

        if args.level == 4:
            metrics_all2 = (metrics[2], metrics_te, metrics_tr)

            for name in metrics_all2[0].keys():
                metrics_hist2[name].append(metrics_all2[0][name])
            for name in metrics_all2[1].keys():
                metrics_hist_te2[name].append(metrics_all2[1][name])
            for name in metrics_all2[2].keys():
                metrics_hist_tr2[name].append(metrics_all2[2][name])
            metrics_hist_all2 = (metrics_hist2, metrics_hist_te2, metrics_hist_tr2)

            save_everything(args, metrics_hist_all2, model, model_dir, None, args.criterion, test_only, hie_level=2)

            metrics_all3 = (metrics[3], metrics_te, metrics_tr)
            for name in metrics_all3[0].keys():
                metrics_hist3[name].append(metrics_all3[0][name])
            for name in metrics_all3[1].keys():
                metrics_hist_te3[name].append(metrics_all3[1][name])
            for name in metrics_all3[2].keys():
                metrics_hist_tr3[name].append(metrics_all3[2][name])
            metrics_hist_all3 = (metrics_hist3, metrics_hist_te3, metrics_hist_tr3)

            save_everything(args, metrics_hist_all3, model, model_dir, None, args.criterion, test_only, hie_level=3)

        elif args.level == 3:
            metrics_all2 = (metrics[2], metrics_te, metrics_tr)

            for name in metrics_all2[0].keys():
                metrics_hist2[name].append(metrics_all2[0][name])
            for name in metrics_all2[1].keys():
                metrics_hist_te2[name].append(metrics_all2[1][name])
            for name in metrics_all2[2].keys():
                metrics_hist_tr2[name].append(metrics_all2[2][name])
            metrics_hist_all2 = (metrics_hist2, metrics_hist_te2, metrics_hist_tr2)

            save_everything(args, metrics_hist_all2, model, model_dir, None, args.criterion, test_only, hie_level=2)

        sys.stdout.flush()

        if test_only:
            break

        if args.criterion in metrics_hist.keys():
            if early_stop(metrics_hist, args.criterion, args.patience):
                # stop training, do tests on test and train sets, and then stop the script
                logger.info("%s hasn't improved in %d epochs, early stopping..." % (args.criterion, args.patience))
                test_only = True
                if args.level:
                    args.test_model = '%s/model_%d_best_%s.pth' % (model_dir, (args.level - 1), args.criterion)
                else:
                    args.test_model = '%s/model_best_%s.pth' % (model_dir, args.criterion)
                model = pick_model(args, dicts, label_desc_inputs, label_desc_mask)


if __name__ == '__main__':
    main()
