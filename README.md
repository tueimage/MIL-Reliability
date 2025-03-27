# MIL-Reliability
__Quantitative Evaluation of Multiple Instance Learning
Reliability For WSIs Classification__

![alt text](https://github.com/tueimage/MIL-Reliability/raw/main/framework.png)
_<sup>The overall framework for evaluating the reliability of MIL models.</sup>_


This is a PyTorch implementation for computing the reliability of MIL models.



**Data Preparation**

For the preprocessing of datasets, we adhere to [CLAM's](https://github.com/mahmoodlab/CLAM) steps. For more details, please refer to the paper.


**Training**

The training can be done for different models and datasets with proper arguments.

```
python train.py --data_root_dir feat-directory ... --lr 1e-4 --reg 1e-5 --seed 2021 --k 5 --k_end 5 --split_dir task_camelyon16 --model_type abmil --task task_1_tumor_vs_normal --csv_path ./dataset_csv/camelyon16.csv --exp_code ABMIL  
```

**Evaluation**

After training, the model can be evaluated to compute and store patch scores using the following command:

```
python eval.py --drop_out --k 5 --k_start 0 --k_end -1  --models_exp_code ABMIL_s2021 --save_exp_code ABMIL_eval --task task_1_tumor_vs_normal --model_type abmil --results_dir results --data_root_dir ... ```
```

**Reliability**

Finally, reliability scores for the model can be calculated across all folds.

```
python reliability.py --model_name ABMIL --att_path ... --anno_path ... ```
```


**Reference**

Please consider citing the following paper if you find our work useful for your project.

```
@misc{,
      title={Quantitative Evaluation of Multiple Instance Learning Reliability For WSIs Classification}, 
      author={Hassan Keshvarikhojasteh},
      year={2024},
      eprint={2409.11110},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
