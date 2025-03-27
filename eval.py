from __future__ import print_function

import numpy as np

import argparse
import torch
import torch.nn as nn
import pdb
import os
import pandas as pd
from utils.utils import *
from math import floor
import matplotlib.pyplot as plt
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits
import h5py
from utils.eval_utils import *

# Training settings
parser = argparse.ArgumentParser(description='CLAM Evaluation Script')
parser.add_argument('--data_root_dir', type=str, default=None,
                    help='data directory')
parser.add_argument('--results_dir', type=str, default='./results',
                    help='relative path to results folder, i.e. '+
                    'the directory containing models_exp_code relative to project root (default: ./results)')
parser.add_argument('--save_exp_code', type=str, default=None,
                    help='experiment code to save eval results')
parser.add_argument('--seed', type=int, default=2021,
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--models_exp_code', type=str, default=None,
                    help='experiment code to load trained models (directory under results_dir containing model checkpoints')
parser.add_argument('--splits_dir', type=str, default=None,
                    help='splits directory, if using custom splits other than what matches the task (default: None)')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', 
                    help='size of model (default: small)')
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_sb_add', 'clam_mb', 'max_pool', 'mean_pool', 'max_pool_ins', 'mean_pool_ins', 'abmil', 'abmil_add', 'madmil', 'madmil_diff', 'madmil_class', 'dtfd', 'acmil', 'acmil_add'], default='clam_sb',
                    help='type of model (default: clam_sb, clam w/ single attention branch)')
parser.add_argument('--drop_out', action='store_true', default=False, 
                    help='whether model uses dropout')
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--fold', type=int, default=-1, help='single fold to evaluate')
parser.add_argument('--micro_average', action='store_true', default=False, 
                    help='use micro_average instead of macro_avearge for multiclass AUC')
parser.add_argument('--split', type=str, choices=['train', 'val', 'test', 'all'], default='test')
parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal',  'task_2_tumor_subtyping'])
### CLAM specific options
parser.add_argument('--no_inst_cluster', action='store_true', default=False,
                     help='disable instance-level clustering')
parser.add_argument('--inst_loss', type=str, choices=['svm', 'ce', None], default=None,
                     help='instance-level clustering loss function (default: None)')
parser.add_argument('--subtyping', action='store_true', default=False,
                     help='subtyping problem')
parser.add_argument('--bag_weight', type=float, default=1.0,
                    help='clam: weight coefficient for bag-level loss (default: 0.7)')
parser.add_argument('--B', type=int, default=8, help='numbr of positive/negative patches to sample for clam')
### MAD-MIL specific options
parser.add_argument('--n', type= int, default= 2, help='number of heads in the attention network')
parser.add_argument('--head_size', type= str, default= "small" , help='dimension of each head in the attention network')
### ACMIL specific options
parser.add_argument('--n_token', type= int, default= 1, help='number of tokens')
parser.add_argument('--n_masked_patch', type= int, default= 10 , help='number of masked patches')
### DTFD specific options
parser.add_argument('--drop_out2', type= float, default= 0.25, help='dropout value')
parser.add_argument('--mDim', type= int, default= 512, help='dimension of the compressed feature')
parser.add_argument('--distill', type=str, choices=['MaxMinS','MaxS', 'AFS'], default='AFS', help='distillation loss')
parser.add_argument('--in_chn', type= int, default= 1024, help='dimension of the input feature')

args = parser.parse_args()

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

args.save_dir = os.path.join('./eval_results', 'EVAL_' + str(args.save_exp_code))
args.models_dir = os.path.join(args.results_dir, str(args.models_exp_code))

os.makedirs(args.save_dir, exist_ok=True)

if args.splits_dir is None:
    args.splits_dir = args.models_dir

assert os.path.isdir(args.models_dir)
assert os.path.isdir(args.splits_dir)

settings = {'task': args.task,
            'split': args.split,
            'save_dir': args.save_dir, 
            'models_dir': args.models_dir,
            'model_type': args.model_type,
            'drop_out': args.drop_out,
            'model_size': args.model_size,
            'n': args.n,
            'head_size': args.head_size,
            'n_token': args.n_token,
            'n_masked_patch': args.n_masked_patch,
            'in_chn': args.in_chn,
            'drop_out2': args.drop_out2,
            'mDim': args.mDim,}

with open(args.save_dir + '/eval_experiment_{}.txt'.format(args.save_exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print(settings)
if args.task == 'task_1_tumor_vs_normal':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/camelyon16.csv',
                            data_dir= os.path.join(args.data_root_dir, 'feat'),
                            shuffle = False, 
                            print_info = True,
                            label_dict={0: 0, 1: 1},
                            patient_strat=False,
                            ignore=[])

elif args.task == 'task_2_tumor_subtyping':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/camelyon16.csv',
                            data_dir= os.path.join(args.data_root_dir, 'feat'),
                            shuffle = False, 
                            print_info = True,
                            label_dict={0: 0, 1: 1},
                            patient_strat= False,
                            ignore=[])


else:
    raise NotImplementedError

if args.k_start == -1:
    start = 0
else:
    start = args.k_start
if args.k_end == -1:
    end = args.k
else:
    end = args.k_end

if args.fold == -1:
    folds = range(start, end)
else:
    folds = range(args.fold, args.fold+1)
ckpt_paths = [os.path.join(args.models_dir, 's_{}_checkpoint.pt'.format(fold)) for fold in folds]
datasets_id = {'train': 0, 'val': 1, 'test': 2, 'all': -1}

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    all_auc = []
    all_acc = []
    all_att = {}
    for ckpt_idx in range(len(ckpt_paths)):
        if datasets_id[args.split] < 0:
            split_dataset = dataset
        else:
            csv_path = '{}/splits_{}.csv'.format(args.splits_dir, folds[ckpt_idx])
            datasets = dataset.return_splits(from_id=False, csv_path=csv_path)
            split_dataset = datasets[datasets_id[args.split]]

        seed_torch(args.seed+ckpt_idx)    
        patient_results, att, acc, auc, df  = eval(split_dataset, args, ckpt_paths[ckpt_idx])
        all_auc.append(auc)
        all_acc.append(acc)
        all_att[ckpt_idx] = att
        df.to_csv(os.path.join(args.save_dir, 'fold_{}.csv'.format(folds[ckpt_idx])), index=False)

    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_auc, 'test_acc': all_acc})
    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(folds[0], folds[-1])
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.save_dir, save_name))
    np.save(os.path.join(args.save_dir, 'att.npy'), all_att)
