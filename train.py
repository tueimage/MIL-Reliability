from __future__ import print_function

import argparse
import pdb
import os
import math

# internal imports
from utils.file_utils import save_pkl
from utils.utils import *
from utils.core_utils import train
from datasets.dataset_generic import Generic_MIL_Dataset

# pytorch imports
import torch
import pandas as pd
import numpy as np


def main(args):
    # create results directory if necessary
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end
    
    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []
    all_test_loss= []
    all_val_loss= []
    folds = np.arange(start, end)
    print('folds:',folds)
    for i in folds:
        seed_torch(args.seed+i)
        # pdb.set_trace()
        train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False,
                csv_path='{}/splits_{}.csv'.format(args.split_dir, i))
        # train_loader = get_split_loader(train_dataset)
        datasets = (train_dataset, val_dataset, test_dataset)
        results, test_auc, val_auc, test_acc, val_acc, test_loss, val_loss  = train(datasets, i, args)
        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)
        all_test_loss.append(test_loss)
        all_val_loss.append(val_loss)
        
        #write results to pkl
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        save_pkl(filename, results)

    "Compute the average and std of the metrics"
    test_auc_ave= np.mean(all_test_auc)
    test_acc_ave= np.mean(all_test_acc)
    test_loss_ave= np.mean(all_test_loss)

    test_auc_std= np.std(all_test_auc)
    test_acc_std= np.std(all_test_acc)
    test_loss_std= np.std(all_test_loss)
    
    val_auc_ave= np.mean(all_val_auc)
    val_acc_ave= np.mean(all_val_acc)
    val_loss_ave= np.mean(all_val_loss)

    val_auc_std= np.std(all_val_auc)
    val_acc_std= np.std(all_val_acc)
    val_loss_std= np.std(all_val_loss)
    
    
    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_test_auc,
        'val_auc': all_val_auc, 'test_acc': all_test_acc, 'val_acc' : all_val_acc, 'test_loss' : all_test_loss, 'val_loss' : all_val_loss})

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.results_dir, save_name))

    print('\n Val:\n loss ± std: {0:.3f} ± {1:.3f}'.format(val_loss_ave, val_loss_std))

    print('\n\n Test:\n auc ± std: {0:.3f} ± {1:.3f}, acc ± std: {2:.3f} ± {3:.3f}'.format(
        test_auc_ave, test_auc_std, test_acc_ave, test_acc_std))
    
    print('\n\n Misc:\n auc ± std (val): {0:.3f} ± {1:.3f}, acc ± std (val): {2:.3f} ± {3:.3f},'
          'loss ± std (test): {4:.3f} ± {5:.3f} \n'.format(val_auc_ave, val_auc_std, val_acc_ave, val_acc_std, test_loss_ave, test_loss_std))
      

    
# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--data_root_dir', type=str, default=None,
                    help='data directory')
parser.add_argument('--max_epochs', type=int, default=50,
                    help='maximum number of epochs to train (default: 50)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--label_frac', type=float, default=1.0,
                    help='fraction of training labels (default: 1.0)')
parser.add_argument('--reg', type=float, default=1e-2,
                    help='weight decay (default: 1e-5)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', type=int, default=5, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=5, help='end fold (default: -1, first fold)')
parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
parser.add_argument('--split_dir', type=str, default=None,
                    help='manually specify the set of splits to use, '
                    +'instead of infering from the task and label_frac argument (default: None)')
parser.add_argument('--log_data', action='store_true', default=True, help='log data using tensorboard')
parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')
parser.add_argument('--opt', type=str, choices = ['adam', 'sgd'], default='adam')
parser.add_argument('--drop_out', action='store_true', default=False, help='enabel dropout (p=0.25)')
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce'], default='ce',
                     help='slide-level classification loss function (default: ce)')
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_sb_add', 'clam_mb', 'max_pool', 'mean_pool', 'max_pool_ins', 'mean_pool_ins', 'abmil', 'abmil_add',  'abmil_con', 'madmil', 'madmil_diff', 'madmil_class','dtfd', 'acmil', 'acmil_add'], default='clam_sb',
                    help='type of model (default: clam_sb, clam w/ single attention branch)')
parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
parser.add_argument('--model_size', type=str, choices=['small', 'big', 'tiny'], default='small', help='size of model, does not affect mil')
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
args.drop_out=True
args.early_stopping =True
args.weighted_sample =True
args.inst_loss ="svm"
args.task = "task_1_tumor_vs_normal"
args.split_dir = "task_camelyon16/"
args.csv_path = './dataset_csv/camelyon16.csv'
args.data_root_dir = "/home/bme001/20215294/Data/CAM/Cam_ostu_20x/"
sub_feat_dir = 'feat'
args.use_drop_out = True
args.bag_weight = 0.7
args.seed = 2021
args.k = 5
args.k_end = 5
args.subtyping= False

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


settings = {'num_splits': args.k,
            'k_start': args.k_start,
            'k_end': args.k_end,
            'task': args.task,
            'max_epochs': args.max_epochs,
            'results_dir': args.results_dir,
            'lr': args.lr,
            'experiment': args.exp_code,
            'reg': args.reg,
            'label_frac': args.label_frac,
            'bag_loss': args.bag_loss,
            'seed': args.seed,
            'model_type': args.model_type,
            'model_size': args.model_size,
            "use_drop_out": args.drop_out,
            'weighted_sample': args.weighted_sample,
            'opt': args.opt,
            'n': args.n,
            'head_size': args.head_size,
            'n_token': args.n_token,
            'n_masked_patch': args.n_masked_patch,
            'in_chn': args.in_chn,
            'drop_out2': args.drop_out2,
            'mDim': args.mDim,}

if args.model_type in ['clam_sb', 'clam_sb_add', 'clam_mb']:
   settings.update({'bag_weight': args.bag_weight,
                    'inst_loss': args.inst_loss,
                    'B': args.B})

print('\nLoad Dataset')

if args.task == 'task_1_tumor_vs_normal':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(
                            csv_path = args.csv_path,
                            data_dir = os.path.join(args.data_root_dir, sub_feat_dir),
                            shuffle = False,
                            seed = args.seed,
                            print_info = True,
                            label_dict={0: 0, 1: 1},
                            patient_strat=False,
                            ignore=[])

elif args.task == 'task_2_tumor_subtyping':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(
                            csv_path = args.csv_path,
                            data_dir = os.path.join(args.data_root_dir, sub_feat_dir),
                            shuffle = False,
                            seed = args.seed,
                            print_info = True,
                            label_dict={0: 0, 1: 1},
                            patient_strat=False,
                            ignore=[])

    if args.model_type in ['clam_sb', 'clam_sb_add', 'clam_mb']:
        assert args.subtyping

else:
    raise NotImplementedError

if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed))
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

if args.split_dir is None:
    args.split_dir = os.path.join('splits', args.task+'_{}'.format(int(args.label_frac*100)))
else:
    args.split_dir = os.path.join('splits', args.split_dir)

print('split_dir: ', args.split_dir)
assert os.path.isdir(args.split_dir)

settings.update({'split_dir': args.split_dir})


with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))

if __name__ == "__main__":
    # pdb.set_trace()
    results = main(args)
    
    print("finished!")
    print("end script")


