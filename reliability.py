
import numpy as np
from glob import glob
import pandas as pd
import argparse
from sklearn.metrics import average_precision_score
from sklearn.metrics import mutual_info_score
from scipy import stats

# Generic settings
parser = argparse.ArgumentParser(description='Configurations for computing the quantitive metrics of reliability')
parser.add_argument('--model_name', type=str, default="CLAM_SB", help='Name of the model')
parser.add_argument('--att_path', type=str, default="/home/bme001/20215294/Implementations/Next/eval_results/", help='Path to the attention scores')
parser.add_argument('--anno_path', type=str, default="/home/bme001/20215294/Data/CAM/Cam_ostu_20x/annos/", help='Path to the annotations')
"Multihead specific settings"
parser.add_argument('--number_head', type=int, default=-1, help='Number of head')
parser.add_argument('--average', type=str, default=False, help='Average the attention scores of the heads or not')
"ADD specific settings"
parser.add_argument('--att_s', type=str, default=False, help='Attention scores of the ADD model or not')
"Parse the arguments"
args = parser.parse_args()

def main(args): 
    "Load the attention scores"
    att_scores = np.load(args.att_path + "EVAL_" + args.model_name + "_eval" + "/att.npy", allow_pickle=True).item()
    folds= len(att_scores.keys())

    "Load the whole annotations"
    anno_files = glob(args.anno_path + "*.npy")

    all_average_mi = []
    all_average_spear = []
    all_average_pr_auc = []
    all_average_pr_auc = []

    "Compute the metrics for each fold"
    for fold in range(folds):
        all_names = []
        all_mi = []
        all_spear = []
        all_pr_auc = []
        for anno_file in anno_files:
            try:
                name= anno_file.split("/")[-1].split(".")[0]

                "Load the annotation and attention score"
                anno = np.load(anno_file).squeeze()

                if "ACMIL" in args.model_name or "MADMIL" in args.model_name:
                    if "ADD" in args.model_name:
                        if args.att_s == "True":
                            if args.average == "True":
                                att = np.mean(att_scores[fold][name][2][:, :, :], axis=1).squeeze()    
                            else:
                                att = att_scores[fold][name][2][:, args.number_head, :].squeeze()
                        else:
                            y_pred = att_scores[fold][name][1][0][0]
                            att = att_scores[fold][name][0][:, y_pred].squeeze()                                
                    else:
                        if args.average == "True":
                            att = np.mean(att_scores[fold][name][2][:, :, :], axis=1).squeeze()    
                        else:
                            att = att_scores[fold][name][2][:, args.number_head, :].squeeze()
                else:
                    if args.att_s == "True":
                        att = att_scores[fold][name][2].squeeze()
                    else:
                        y_pred = att_scores[fold][name][1][0][0]
                        att = att_scores[fold][name][0][:, y_pred].squeeze()

                "Sigmoid function for the attention scores of the models"
                att = 1 / (1 + np.exp(-att))  

                "Mutual Information between attention score and the annotation"
                c_xy = np.histogram2d(anno, att, bins= 1000)[0]
                mi = mutual_info_score(None, None, contingency=c_xy)
                all_mi.append(round(mi, 4))

                "Spearmans correlation between attention score and the annotation"
                spear = stats.spearmanr(anno, att)[0]
                all_spear.append(round(spear, 4))
                
                "Precision-Recall AUC between attention score and the annotation"
                pr_auc = average_precision_score(anno, att)
                all_pr_auc.append(round(pr_auc, 4))

                "Append the name"
                all_names.append(name)
                
            except:
                pass
        "Print the average of the metrics"
        average_mi = np.mean(all_mi)
        average_spear = np.mean(all_spear)
        average_pr_auc = np.mean(all_pr_auc)

        "Append the average values"
        all_average_mi.append(average_mi)
        all_average_spear.append(average_spear)
        all_average_pr_auc.append(average_pr_auc)

        all_names.append("Average")
        all_mi.append(round(average_mi, 4))
        all_spear.append(round(average_spear, 4))
        all_pr_auc.append(round(average_pr_auc, 4))

        "print the rounded values"
        if "ACMIL" in args.model_name  or "MADMIL" in args.model_name:
            print("Fold: ", fold, ", Head: ", args.number_head, ", Average MI: ", round(average_mi, 4),
            ", Average Spear: ", round(average_spear, 4), 
            ", Average PR_AUC: ", round(average_pr_auc, 4))
        else:  
            print("Fold: ", fold, ", Average MI: ", round(average_mi, 4),
            ", Average Spear: ", round(average_spear, 4), 
            ", Average PR_AUC: ", round(average_pr_auc, 4))

        final_df = pd.DataFrame({'names': all_names, 'MI': all_mi, 'SPEAR': all_spear, 'PR_AUC': all_pr_auc})

        "Save the dataframe"
        if "ACMIL" in args.model_name  or "MADMIL" in args.model_name:
            final_df.to_csv(args.att_path + "EVAL_" + args.model_name + "_eval" + "/inter_{}_head{}_average{}_fold_{}.csv".format(args.model_name, args.number_head, args.average, fold), index=False)
        else:
            final_df.to_csv(args.att_path + "EVAL_" + args.model_name + "_eval" + "/inter_{}_fold_{}.csv".format(args.model_name, fold), index=False)

    "Print the average and std of the metrics"
    if "ACMIL" in args.model_name  or "MADMIL" in args.model_name:
        print("\nHead: ", args.number_head, ", Average MI ± std: {0:.4f} ± {1:.4f} ".format(np.mean(all_average_mi), np.std(all_average_mi)),
             ", Average Spear ± std: {0:.4f} ± {1:.4f} ".format(np.mean(all_average_spear), np.std(all_average_spear)),
            ", Average PR_AUC ± std: {0:.4f} ± {1:.4f} ".format(np.mean(all_average_pr_auc), np.std(all_average_pr_auc)))
    else:
        print("\nAverage MI ± std: {0:.4f} ± {1:.4f} ".format(np.mean(all_average_mi), np.std(all_average_mi)),
             ", Average Spear ± std: {0:.4f} ± {1:.4f} ".format(np.mean(all_average_spear), np.std(all_average_spear)),
            ", Average PR_AUC ± std: {0:.4f} ± {1:.4f} ".format(np.mean(all_average_pr_auc), np.std(all_average_pr_auc)))

    "Save the average values"
    if "ACMIL" in args.model_name  or "MADMIL" in args.model_name:
        final_df_model = pd.DataFrame({'Model': args.model_name, 'Head': args.number_head, 'Average': args.average, 'Average MI ± std': "{0:.4f} ± {1:.4f}".format(np.mean(all_average_mi), np.std(all_average_mi)),
             'Average Spear ± std': "{0:.4f} ± {1:.4f}".format(np.mean(all_average_spear), np.std(all_average_spear)),
            'Average PR_AUC ± std': "{0:.4f} ± {1:.4f}".format(np.mean(all_average_pr_auc), np.std(all_average_pr_auc))}, index= [0])
        final_df_model.to_csv(args.att_path + "EVAL_" + args.model_name + "_eval" + "/inter_{}_head{}_average{}_average.csv".format(args.model_name, args.number_head, args.average), index=False)    
    else:        
        final_df_model = pd.DataFrame({'Model': args.model_name, 'Average MI ± std': "{0:.4f} ± {1:.4f}".format(np.mean(all_average_mi), np.std(all_average_mi)),
             'Average Spear ± std': "{0:.4f} ± {1:.4f}".format(np.mean(all_average_spear), np.std(all_average_spear)),
            'Average PR_AUC ± std': "{0:.4f} ± {1:.4f}".format(np.mean(all_average_pr_auc), np.std(all_average_pr_auc))}, index= [0])
        final_df_model.to_csv(args.att_path + "EVAL_" + args.model_name + "_eval" + "/inter_{}_average.csv".format(args.model_name), index=False)    

if __name__ == "__main__":
    main(args)
    print("Done!")