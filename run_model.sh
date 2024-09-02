# bin/bash
export num_proc_per_node=4
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python train_rewardmodel.py