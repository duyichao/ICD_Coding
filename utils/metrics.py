#!/usr/bin/env/python
"""
@author: ycdu
@mail: ycdu666@gmail.com
@IDE: PyCharm
@file: metrics.py
@time: 2020/9/2 11:49 上午
@desc:
"""
import json
import torch
import numpy as np
from texttable import Texttable
from prettytable import PrettyTable


def save_metrics(metrics_hist_all, model_dir, hie_level):
    if hie_level:
        with open(model_dir + '/hie_' + str(hie_level) + "_metrics.json", 'w') as metrics_file:
            # concatenate dev, train metrics into one dict
            data = metrics_hist_all[0].copy()
            data.update({"%s_te" % name: val for (name, val) in metrics_hist_all[1].items()})
            data.update({"%s_tr" % name: val for (name, val) in metrics_hist_all[2].items()})
            json.dump(data, metrics_file, indent=1)
    else:
        with open(model_dir + "/metrics.json", 'w') as metrics_file:
            # concatenate dev, train metrics into one dict
            data = metrics_hist_all[0].copy()
            data.update({"%s_te" % name: val for (name, val) in metrics_hist_all[1].items()})
            data.update({"%s_tr" % name: val for (name, val) in metrics_hist_all[2].items()})
            json.dump(data, metrics_file, indent=1)


def save_everything(args, metrics_hist_all, model, model_dir, params, criterion, evaluate=False, hie_level=None):
    save_metrics(metrics_hist_all, model_dir, hie_level)

    if not evaluate:
        # save the model with the best criterion metric
        if not np.all(np.isnan(metrics_hist_all[0][criterion])):
            if criterion == 'loss_dev':
                eval_val = np.nanargmin(metrics_hist_all[0][criterion])
            else:
                eval_val = np.nanargmax(metrics_hist_all[0][criterion])

            if eval_val == len(metrics_hist_all[0][criterion]) - 1:
                sd = model.cpu().state_dict()
                if hie_level:
                    torch.save(sd, model_dir + '/model_' + str(hie_level) + "_best_%s.pth" % criterion)
                else:
                    torch.save(sd, model_dir + "/model_best_%s.pth" % criterion)
                if args.gpu >= 0:
                    model.cuda(args.gpu)
    print("saved metrics, params, model to directory %s\n" % (model_dir))


def print_metrics(metrics, logger):
    p1, p2 = Texttable().set_precision(5), Texttable().set_precision(5)
    if "auc_macro" in metrics.keys():
        p1.add_row(['', 'accuracy', 'precision', 'recall', 'f1-score', 'auc'])
        # p1.add_row(
        #     ['macro', "%.5f" % metrics["acc_macro"], "%.5f" % metrics["prec_macro"], "%.5f" % metrics["rec_macro"],
        #      "%.5f" % metrics["f1_macro"], "%.5f" % metrics["auc_macro"]])
        p1.add_row(['macro', metrics["acc_macro"], metrics["prec_macro"], metrics["rec_macro"], metrics["f1_macro"],
                    metrics["auc_macro"]])
    else:
        p1.add_row(['', 'accuracy', 'precision', 'recall', 'f1-score'])
        p1.add_row(['macro', metrics["acc_macro"], metrics["prec_macro"], metrics["rec_macro"], metrics["f1_macro"]])
    if "auc_micro" in metrics.keys():
        p1.add_row(['micro', metrics["acc_micro"], metrics["prec_micro"], metrics["rec_micro"], metrics["f1_micro"],
                    metrics["auc_micro"]])
    else:
        p1.add_row(['micro', metrics["acc_micro"], metrics["prec_micro"], metrics["rec_micro"], metrics["f1_micro"]])
    logger.info('\n' + p1.draw())

    for metric, val in metrics.items():
        if metric.find("rec_at") != -1:
            p2.add_row([metric, "%.6f" % val])
    logger.info('\n' + p2.draw())
    """
    metrics = {'acc_macro': 54456, 'prec_macro': 1212.11, 'rec_macro': 1.0111, 'f1_macro': 121.11, 'auc_macro': 12.11,
               'acc_micro': 45.11, 'prec_micro': 121.11, 'rec_micro': 545.4545, 'f1_micro': 44.4154, 'auc_micro': 12.11,
               'prec_at_8': 15}
    """


def union_size(yhat, y, axis):
    # axis=0 for label-level union (macro). axis=1 for instance-level
    return np.logical_or(yhat, y).sum(axis=axis).astype(float)


def intersect_size(yhat, y, axis):
    # axis=0 for label-level union (macro). axis=1 for instance-level
    return np.logical_and(yhat, y).sum(axis=axis).astype(float)


def macro_accuracy(yhat, y):
    num = intersect_size(yhat, y, 0) / (union_size(yhat, y, 0) + 1e-10)
    return np.mean(num)


def macro_precision(yhat, y):
    num = intersect_size(yhat, y, 0) / (yhat.sum(axis=0) + 1e-10)
    return np.mean(num)


def macro_recall(yhat, y):
    num = intersect_size(yhat, y, 0) / (y.sum(axis=0) + 1e-10)
    return np.mean(num)


def macro_f1(yhat, y):
    prec = macro_precision(yhat, y)
    rec = macro_recall(yhat, y)
    if prec + rec == 0:
        f1 = 0.
    else:
        f1 = 2 * (prec * rec) / (prec + rec)
    return f1


def all_macro(yhat, y):
    return macro_accuracy(yhat, y), macro_precision(yhat, y), macro_recall(yhat, y), macro_f1(yhat, y)


def micro_accuracy(yhatmic, ymic):
    return intersect_size(yhatmic, ymic, 0) / union_size(yhatmic, ymic, 0)


def micro_precision(yhatmic, ymic):
    return intersect_size(yhatmic, ymic, 0) / yhatmic.sum(axis=0)


def micro_recall(yhatmic, ymic):
    return intersect_size(yhatmic, ymic, 0) / ymic.sum(axis=0)


def micro_f1(yhatmic, ymic):
    prec = micro_precision(yhatmic, ymic)
    rec = micro_recall(yhatmic, ymic)
    if prec + rec == 0:
        f1 = 0.
    else:
        f1 = 2 * (prec * rec) / (prec + rec)
    return f1


def all_micro(yhatmic, ymic):
    return micro_accuracy(yhatmic, ymic), micro_precision(yhatmic, ymic), micro_recall(yhatmic, ymic), micro_f1(yhatmic,
                                                                                                                ymic)


from sklearn.metrics import roc_curve, auc


def auc_metrics(yhat_raw, y, ymic):
    if yhat_raw.shape[0] <= 1:
        return
    fpr = {}
    tpr = {}
    roc_auc = {}
    # get AUC for each label individually
    relevant_labels = []
    auc_labels = {}
    for i in range(y.shape[1]):
        # only if there are true positives for this label
        if y[:, i].sum() > 0:
            fpr[i], tpr[i], _ = roc_curve(y[:, i], yhat_raw[:, i])
            if len(fpr[i]) > 1 and len(tpr[i]) > 1:
                auc_score = auc(fpr[i], tpr[i])
                if not np.isnan(auc_score):
                    auc_labels["auc_%d" % i] = auc_score
                    relevant_labels.append(i)

    # macro-AUC: just average the auc scores
    aucs = []
    for i in relevant_labels:
        aucs.append(auc_labels['auc_%d' % i])
    roc_auc['auc_macro'] = np.mean(aucs)

    # micro-AUC: just look at each individual prediction
    yhatmic = yhat_raw.ravel()
    fpr["micro"], tpr["micro"], _ = roc_curve(ymic, yhatmic)
    roc_auc["auc_micro"] = auc(fpr["micro"], tpr["micro"])

    return roc_auc


def recall_at_k(yhat_raw, y, k):
    # num true labels in top k predictions / num true labels
    sortd = np.argsort(yhat_raw)[:, ::-1]
    topk = sortd[:, :k]

    # get recall at k for each example
    vals = []
    for i, tk in enumerate(topk):
        num_true_in_top_k = y[i, tk].sum()
        denom = y[i, :].sum()
        vals.append(num_true_in_top_k / float(denom))

    vals = np.array(vals)
    vals[np.isnan(vals)] = 0.

    return np.mean(vals)


def precision_at_k(yhat_raw, y, k):
    # num true labels in top k predictions / k
    sortd = np.argsort(yhat_raw)[:, ::-1]
    topk = sortd[:, :k]

    # get precision at k for each example
    vals = []
    for i, tk in enumerate(topk):
        if len(tk) > 0:
            num_true_in_top_k = y[i, tk].sum()
            denom = len(tk)
            vals.append(num_true_in_top_k / float(denom))

    return np.mean(vals)


def all_metrics(yhat, y, k=8, yhat_raw=None, calc_auc=True):
    """
        Inputs:
            yhat: binary predictions matrix
            y: binary ground truth matrix
            k: for @k metrics
            yhat_raw: prediction scores matrix (floats)
        Outputs:
            dict holding relevant metrics
    """
    names = ["acc", "prec", "rec", "f1"]

    # macro
    macro = all_macro(yhat, y)

    # micro
    ymic = y.ravel()
    yhatmic = yhat.ravel()
    micro = all_micro(yhatmic, ymic)

    metrics = {names[i] + "_macro": macro[i] for i in range(len(macro))}
    metrics.update({names[i] + "_micro": micro[i] for i in range(len(micro))})

    # AUC and @k
    if yhat_raw is not None and calc_auc:
        # allow k to be passed as int or list
        if type(k) != list:
            k = [k]
        for k_i in k:
            rec_at_k = recall_at_k(yhat_raw, y, k_i)
            metrics['rec_at_%d' % k_i] = rec_at_k
            prec_at_k = precision_at_k(yhat_raw, y, k_i)
            metrics['prec_at_%d' % k_i] = prec_at_k
            metrics['f1_at_%d' % k_i] = 2 * (prec_at_k * rec_at_k) / (prec_at_k + rec_at_k)

        roc_auc = auc_metrics(yhat_raw, y, ymic)
        metrics.update(roc_auc)

    return metrics


def _readString(f, code):
    # s = unicode()
    s = str()
    c = f.read(1)
    value = ord(c)

    while value != 10 and value != 32:
        if 0x00 < value < 0xbf:
            continue_to_read = 0
        elif 0xC0 < value < 0xDF:
            continue_to_read = 1
        elif 0xE0 < value < 0xEF:
            continue_to_read = 2
        elif 0xF0 < value < 0xF4:
            continue_to_read = 3
        else:
            raise RuntimeError("not valid utf-8 code")

        i = 0
        # temp = str()
        # temp = temp + c

        temp = bytes()
        temp = temp + c

        while i < continue_to_read:
            temp = temp + f.read(1)
            i += 1

        temp = temp.decode(code)
        s = s + temp

        c = f.read(1)
        value = ord(c)

    return s
