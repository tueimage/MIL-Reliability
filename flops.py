
import argparse
from calflops import calculate_flops
from utils.utils import *
from models.model_abmil import ABMIL, MADMIL, MADMIL_CLASS
from models.model_abmil_add import ABMIL_ADD
from models.model_clam import CLAM_MB, CLAM_SB
from models.model_clam_add import CLAM_SB_ADD
from models.model_acmil import ACMIL
from models.model_acmil_add import ACMIL_ADD
from models.model_pool import Max_Pool, Mean_Pool
from models.model_pool_instance import Max_Pool_Ins, Mean_Pool_Ins
from models.model_dtfd import DimReduction, Attention, Classifier_1fc, Attention_with_Classifier


# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for FLOPs computation')
parser.add_argument('--drop_out', action='store_true', default=False, help='enabel dropout (p=0.25)')
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_sb_add', 'clam_mb', 'max_pool', 'mean_pool', 'max_pool_ins', 'mean_pool_ins', 'abmil', 'abmil_add',  'abmil_con', 'madmil', 'madmil_diff', 'madmil_class','dtfd', 'acmil', 'acmil_add'], default='clam_sb',
                    help='type of model (default: clam_sb, clam w/ single attention branch)')
parser.add_argument('--model_size', type=str, choices=['small', 'big', 'tiny'], default='small', help='size of model, does not affect mil')
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

args.drop_out=True
args.use_drop_out = True
args.subtyping= False
args.inst_loss ="svm"
args.n_classes=2

print('\nInit Model...', end=' ')
if args.model_type != 'dtfd':
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
    if args.model_type == 'clam' and args.subtyping:
        model_dict.update({'subtyping': True})
    
    if args.model_size is not None and args.model_type != 'mil':
        model_dict.update({"size_arg": args.model_size})
    
    if args.model_type in ['clam_sb', 'clam_sb_add', 'clam_mb']:
        if args.subtyping:
            model_dict.update({'subtyping': True})
        
        if args.B > 0:
            model_dict.update({'k_sample': args.B})
        
        if args.inst_loss == 'svm':
            from topk.svm import SmoothTop1SVM
            instance_loss_fn = SmoothTop1SVM(n_classes = 2)
            if device.type == 'cuda':
                instance_loss_fn = instance_loss_fn.cuda()
        else:
            instance_loss_fn = nn.CrossEntropyLoss()

        if args.model_type =='clam_sb':
            model = CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn)
        elif args.model_type == 'clam_sb_add':
            model = CLAM_SB_ADD(**model_dict, instance_loss_fn=instance_loss_fn)    
        elif args.model_type == 'clam_mb':
            model = CLAM_MB(**model_dict, instance_loss_fn=instance_loss_fn)
        else:
            raise NotImplementedError   
        
    elif args.model_type == 'abmil':
        model = ABMIL(**model_dict)           
    elif args.model_type == 'abmil_add':
        model = ABMIL_ADD(**model_dict)          
    elif args.model_type == 'acmil':
        model = ACMIL(**model_dict, n_token= args.n_token, n_masked_patch= args.n_masked_patch)    
    elif args.model_type == 'acmil_add':
        model = ACMIL_ADD(**model_dict, n_token= args.n_token, n_masked_patch= args.n_masked_patch)          
    elif args.model_type == 'madmil':
        model = MADMIL(**model_dict, n= args.n, head_size= args.head_size)     
    elif args.model_type == 'madmil_diff':
        model = MADMIL(**model_dict, n= args.n, head_size= args.head_size)  
    elif args.model_type == 'madmil_class':
        model = MADMIL_CLASS(**model_dict, n= args.n, head_size= args.head_size)                               
    elif args.model_type == 'max_pool':
        model = Max_Pool(**model_dict)   
    elif args.model_type == 'max_pool_ins':
        model = Max_Pool_Ins(**model_dict)           
    elif args.model_type == 'mean_pool':
        model = Mean_Pool(**model_dict) 
    elif args.model_type == 'mean_pool_ins':
        model = Mean_Pool_Ins(**model_dict)     


    model.relocate()
    print('Done!')
    print_network(model)
   
elif args.model_type == 'dtfd':
    dimReduction = DimReduction(args.in_chn, args.mDim)
    attention = Attention(args.mDim)
    classifier = Classifier_1fc(args.mDim, args.n_classes, droprate= 0)
    attCls = Attention_with_Classifier(L=args.mDim, num_cls=args.n_classes, droprate=args.drop_out2)

    dimReduction.relocate()
    attention.relocate()
    classifier.relocate()
    attCls.relocate()
    print('Done!')
    
    print_network(dimReduction)
    print_network(attention)
    print_network(classifier)
    print_network(attCls)

input_shape = (120, args.in_chn)
flops, macs, params = calculate_flops(model=model, 
                                      input_shape=input_shape,
                                      output_as_string=True,
                                      output_precision=4)
print("FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))
