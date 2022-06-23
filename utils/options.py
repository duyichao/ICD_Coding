#!/usr/bin/env/python
"""
@author: ycdu
@mail: ycdu666@gmail.com
@IDE: PyCharm
@file: options.py
@time: 2020/9/1 3:17 下午
@desc:
"""

import argparse
import sys
import time

parser = argparse.ArgumentParser()

# path
ROOT_DIR = '/data/ycdu/workspace/ICD-Coding/'
DATA_DIR = '/data/ycdu/workspace/ICD-Coding/data'
parser.add_argument("--DATA_DIR", type=str, default=DATA_DIR)
parser.add_argument("--MIMIC_3_DIR", type=str, default=DATA_DIR + '/mimic3')
parser.add_argument("--MIMIC_2_DIR", type=str, default=DATA_DIR + '/mimic2')
parser.add_argument("--MODEL_DIR", type=str, default='./predictions')
parser.add_argument("--data_path", type=str, default=DATA_DIR + '/mimic3/train_50_4_level.json')
parser.add_argument("--vocab", type=str, default=DATA_DIR + '/mimic3/vocab.csv')
parser.add_argument("--icd_path", type=str, default=DATA_DIR + '/ICD9CM.csv')
parser.add_argument("--hie_relations_path", type=str, default=DATA_DIR + '/hierarchy_relations.txt')
parser.add_argument("--train_prob_path", type=str, default=DATA_DIR + '/mimic3/train_prob.json')
parser.add_argument("--label_index_path", type=str, default=DATA_DIR + '/mimic3/label_index.txt')
parser.add_argument("--label_desc_path", type=str, default=DATA_DIR + '/mimic3/label_desc.txt')

# common
parser.add_argument("--Y", type=str, default="full", choices=["full", "50", "100", "150", "200", "500"])
parser.add_argument("--version", type=str, choices=["mimic2", "mimic3"], default="mimic3")
parser.add_argument("--MAX_LENGTH", type=int, default=3000)
parser.add_argument("--level", type=int, default=2)
parser.add_argument("--level_label_count", type=list, default=[1167, 8925])
# parser.add_argument("--level_label_count", type=list, default=[199, 1175, 5125, 8925])

# hie projector
parser.add_argument("--level_out_size", type=int, default=100)
parser.add_argument("--level_proj_size", type=list, default=[40, 50, 0])  # 25 48 50
# parser.add_argument("--level_proj_size", type=list, default=[300, 800, 0])  # 25 40 48 50

# model
parser.add_argument("--model", type=str)
parser.add_argument("--filter_size", type=str, default="3,5,9,15,19,21")
# parser.add_argument("--filter_size", type=str, default="1,3,5,7,9,11,13")
parser.add_argument("--num_filter_maps", type=int, default=50)
# parser.add_argument("--num_filter_maps", type=int, default=100)
parser.add_argument("--conv_layer", type=int, default=1)
# parser.add_argument("--embed_file", type=str, default=DATA_DIR + '/mimic3/processed_full.embed')
parser.add_argument("--embed_file", type=str)
parser.add_argument("--embed_size", type=int, default=100)
parser.add_argument("--test_model", type=str, default=None)
parser.add_argument("--att_size", type=int, default=300)

# cooc
parser.add_argument("--co_hid_size", type=int, default=300)

# hyperbolic
# parser.add_argument("---hyper_embed_path", default='/data/ycdu/workspace/Auto-Diagnosis/data/poincare_icd_embed.txt')
parser.add_argument("---hyper_embed_path", default='/data/ycdu/workspace/Auto-Diagnosis/data/lorentz_icd_embed.txt')

# dropout
parser.add_argument("--dropout", type=float, default=0.4)
parser.add_argument("--embed_dropout", type=float, default=0.4)
parser.add_argument("--conv_dropout", type=float, default=0.4)
parser.add_argument("--att_dropout", type=float, default=0.4)

# training
parser.add_argument("--n_epochs", type=int, default=500)
parser.add_argument("--patience", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=0)
parser.add_argument("--criterion", type=str, default="prec_at_8", choices=["prec_at_8", "f1_micro", "prec_at_5"])
parser.add_argument("--gpu", type=int, default=1, help='-1 if not use gpu, >=0 if use gpu')
parser.add_argument("--tune_wordemb", action="store_const", const=True, default=True)
parser.add_argument("--random_seed", type=int, default=1,
                    help='0 if randomly initialize the model, other if fix the seed')
parser.add_argument("--loader_workers", type=int, default=32)
parser.add_argument("--optimizer", type=str, default='AdaBelief')

# logger
parser.add_argument("--log_name", type=str, default='ICDlog')
parser.add_argument("--log_level", type=str, default='info')
parser.add_argument("--log_file", type=str,
                    default="./log/mimic3_" + time.strftime("%b_%d_%H_%M_%S", time.localtime()) + ".log")

args = parser.parse_args()
command = ' '.join(['python3'] + sys.argv)
args.command = command

