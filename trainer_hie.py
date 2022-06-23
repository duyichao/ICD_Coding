#!/usr/bin/env/python
"""
@author: ycdu
@mail: ycdu666@gmail.com
@IDE: PyCharm
@file: trainer_hie.py
@time: 2020/9/2 4:44 下午
@desc:
"""

import torch
import numpy as np
from utils.metrics import all_metrics, print_metrics


def train(args, logger, model, optimizer, epoch, gpu, data_loader):
    logger.info("EPOCH %d" % epoch)

    losses = []

    model.train()

    # loader
    data_iter = iter(data_loader)
    num_iter = len(data_loader)
    for i in range(num_iter):
        inputs_id, labels, level1, level2, level3 = next(data_iter)
        inputs_id, labels, = torch.LongTensor(inputs_id), torch.FloatTensor(labels)
        level1, level2, level3 = torch.FloatTensor(level1), torch.FloatTensor(level2), torch.FloatTensor(level3)
        if gpu >= 0:
            inputs_id, labels = inputs_id.to(args.device), labels.to(args.device)
            level1, level2, level3 = level1.to(args.device), level2.to(args.device), level3.to(args.device)
            # label_inputs.to(args.device), label_inputs_len.to(args.device)

        output_, loss = model(inputs_id, labels, level1, level2, level3)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return losses


def test(args, logger, model, data_path, fold, gpu, dicts, data_loader):
    pass
    filename = data_path.replace(' train', fold)
    num_labels = len(dicts['ind2c'])

    y1, yhat1, yhat_raw1, hids1, losses1 = [], [], [], [], []
    y2, yhat2, yhat_raw2, hids3, losses2 = [], [], [], [], []
    y3, yhat3, yhat_raw3, hids3, losses3 = [], [], [], [], []
    y, yhat, yhat_raw, hids, losses = [], [], [], [], []

    model.eval()

    # data_loader
    data_iter = iter(data_loader)
    num_iter = len(data_loader)
    for i in range(num_iter):
        with torch.no_grad():
            inputs_id, labels, level1, level2, level3 = next(data_iter)
            inputs_id, labels, = torch.LongTensor(inputs_id), torch.FloatTensor(labels)
            level1, level2, level3 = torch.FloatTensor(level1), torch.FloatTensor(level2), torch.FloatTensor(level3)
            if gpu != -1:
                inputs_id, labels = inputs_id.to(args.device), labels.to(args.device)
                level1, level2, level3 = level1.to(args.device), level2.to(args.device), level3.to(args.device)

            output_, loss = model(inputs_id, labels, level1, level2, level3)

            losses.append(loss.item())

            output = torch.sigmoid(output_[-1])
            output = output.data.cpu().numpy()

            target_data = labels.data.cpu().numpy()

            yhat_raw.append(output)
            output = np.round(output)
            y.append(target_data)
            yhat.append(output)

            if args.level == 2:
                output1 = torch.sigmoid(output_[0])
                output1 = output1.data.cpu().numpy()
                target_data1 = level1.data.cpu().numpy()
                yhat_raw1.append(output1)
                output1 = np.round(output1)
                y1.append(target_data1)
                yhat1.append(output1)

            elif args.level == 4:
                output1 = torch.sigmoid(output_[0])
                output1 = output1.data.cpu().numpy()
                target_data1 = level1.data.cpu().numpy()
                yhat_raw1.append(output1)
                output1 = np.round(output1)
                y1.append(target_data1)
                yhat1.append(output1)

                output2 = torch.sigmoid(output_[1])
                output2 = output2.data.cpu().numpy()
                target_data2 = level2.data.cpu().numpy()
                yhat_raw2.append(output2)
                output2 = np.round(output2)
                y2.append(target_data2)
                yhat2.append(output2)

                output3 = torch.sigmoid(output_[2])
                output3 = output3.data.cpu().numpy()
                target_data3 = level3.data.cpu().numpy()
                yhat_raw3.append(output3)
                output3 = np.round(output3)
                y3.append(target_data3)
                yhat3.append(output3)

            elif args.level == 3:
                # output1 = torch.sigmoid(output_[0])
                # output1 = output1.data.cpu().numpy()
                # target_data1 = level1.data.cpu().numpy()
                # yhat_raw1.append(output1)
                # output1 = np.round(output1)
                # y1.append(target_data1)
                # yhat1.append(output1)
                #
                # output2 = torch.sigmoid(output_[1])
                # output2 = output2.data.cpu().numpy()
                # target_data2 = level2.data.cpu().numpy()
                # yhat_raw2.append(output2)
                # output2 = np.round(output2)
                # y2.append(target_data2)
                # yhat2.append(output2)

                # l2 l3
                output2 = torch.sigmoid(output_[0])
                output2 = output2.data.cpu().numpy()
                target_data2 = level2.data.cpu().numpy()
                yhat_raw2.append(output2)
                output2 = np.round(output2)
                y2.append(target_data2)
                yhat2.append(output2)

                output3 = torch.sigmoid(output_[1])
                output3 = output3.data.cpu().numpy()
                target_data3 = level3.data.cpu().numpy()
                yhat_raw3.append(output3)
                output3 = np.round(output3)
                y3.append(target_data3)
                yhat3.append(output3)

    y = np.concatenate(y, axis=0)
    yhat = np.concatenate(yhat, axis=0)
    yhat_raw = np.concatenate(yhat_raw, axis=0)

    k = 5 if num_labels in [50, 100, 150, 200] else [8, 15]
    metrics = all_metrics(yhat, y, k=k, yhat_raw=yhat_raw)
    print_metrics(metrics, logger=logger)
    metrics['loss_%s' % fold] = np.mean(losses)

    if args.level == 2:
        y1 = np.concatenate(y1, axis=0)
        yhat1 = np.concatenate(yhat1, axis=0)
        yhat_raw1 = np.concatenate(yhat_raw1, axis=0)

        metrics1 = all_metrics(yhat1, y1, k=k, yhat_raw=yhat_raw1)
        print_metrics(metrics1, logger=logger)
        metrics1['loss_%s' % fold] = np.mean(losses)
        return [metrics1, metrics]

    elif args.level == 3:
        # y1 = np.concatenate(y1, axis=0)
        # yhat1 = np.concatenate(yhat1, axis=0)
        # yhat_raw1 = np.concatenate(yhat_raw1, axis=0)
        #
        # metrics1 = all_metrics(yhat1, y1, k=k, yhat_raw=yhat_raw1)
        # print_metrics(metrics1, logger=logger)
        # metrics1['loss_%s' % fold] = np.mean(losses)

        y2 = np.concatenate(y2, axis=0)
        yhat2 = np.concatenate(yhat2, axis=0)
        yhat_raw2 = np.concatenate(yhat_raw2, axis=0)

        metrics2 = all_metrics(yhat2, y2, k=k, yhat_raw=yhat_raw2)
        print_metrics(metrics2, logger=logger)
        metrics2['loss_%s' % fold] = np.mean(losses)

        y3 = np.concatenate(y3, axis=0)
        yhat3 = np.concatenate(yhat3, axis=0)
        yhat_raw3 = np.concatenate(yhat_raw3, axis=0)

        metrics3 = all_metrics(yhat3, y3, k=k, yhat_raw=yhat_raw3)
        print_metrics(metrics3, logger=logger)
        metrics3['loss_%s' % fold] = np.mean(losses)

        # return [metrics1, metrics2, metrics]
        return [metrics2, metrics3, metrics]

    elif args.level == 4:
        y1 = np.concatenate(y1, axis=0)
        yhat1 = np.concatenate(yhat1, axis=0)
        yhat_raw1 = np.concatenate(yhat_raw1, axis=0)

        metrics1 = all_metrics(yhat1, y1, k=k, yhat_raw=yhat_raw1)
        print_metrics(metrics1, logger=logger)
        metrics1['loss_%s' % fold] = np.mean(losses)

        y2 = np.concatenate(y2, axis=0)
        yhat2 = np.concatenate(yhat2, axis=0)
        yhat_raw2 = np.concatenate(yhat_raw2, axis=0)

        metrics2 = all_metrics(yhat2, y2, k=k, yhat_raw=yhat_raw2)
        print_metrics(metrics2, logger=logger)
        metrics2['loss_%s' % fold] = np.mean(losses)

        y3 = np.concatenate(y3, axis=0)
        yhat3 = np.concatenate(yhat3, axis=0)
        yhat_raw3 = np.concatenate(yhat_raw3, axis=0)

        metrics3 = all_metrics(yhat3, y3, k=k, yhat_raw=yhat_raw3)
        print_metrics(metrics3, logger=logger)
        metrics3['loss_%s' % fold] = np.mean(losses)

        return [metrics1, metrics2, metrics3, metrics]

    return [metrics]
