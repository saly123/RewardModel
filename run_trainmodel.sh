#!/bin/bash
# python xx.py使用的python版本是实例默认的版本,需要指定镜像中python的具体路径来执行指定版本的python
/home/powerop/work/conda/envs/python39/bin/python3.10 /home/powerop/work/reward_model/RewardModel/train_rewardmodel.py


# 多卡执行
# 1) torch.distributed.launch
# nproc_per_node * nnodes = total number of GPUs
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr="192.168.1.1" --master_port=12345 train_rewardmodel.py

# 2) torchrun
torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 --master_addr="0.0.0.0" --master_port=8000 train_rewardmodel.py