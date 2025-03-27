# MIL-Reliability
__Quantitative Evaluation of Multiple Instance Learning
Reliability For WSIs Classification__


![alt text](https://github.com/tueimage/MIL-Reliability/blob/main/model.png)
_<sup>The overall framework for evaluating the reliability of MIL models.</sup>_


This is a PyTorch implementation for computing the reliability of MIL models.



**Data Preparation**

For the preprocessing of datasets, we adhere to [CLAM's](https://github.com/mahmoodlab/CLAM) steps. For more details, please refer to the paper.


**Training**

The training can be done for different models and datasets with proper arguments.

```
python train.py --data_root_dir feat-directory --lr 1e-4 --reg 1e-5 --seed 2021 --k 5 --k_end 5 --split_dir task_camelyon16 --model_type abmil --task task_1_tumor_vs_normal --csv_path ./dataset_csv/camelyon16.csv --exp_code ABMIL  
```


**Reference**

Please consider citing the following paper if you find our work useful for your project.

```
@misc{,
      title={Quantitative Evaluation of Multiple Instance Learning Reliability For WSIs Classification}, 
      author={},
      year={2024},
      eprint={},
      archivePrefix={},
      primaryClass={cs.CV}
}
```
