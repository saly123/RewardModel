#!/bin/bash
# python xx.py使用的python版本是实例默认的版本,需要指定镜像中python的具体路径来执行指定版本的python
/home/powerop/work/conda/envs/python39/bin/python3.10 /home/powerop/work/reward_model/RewardModel/train_rewardmodel.py


# 多卡执行
# 1) torch.distributed.launch
# nproc_per_node * nnodes = total number of GPUs
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr="192.168.1.1" --master_port=12345 train_rewardmodel.py

# 2) torchrun
torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 --master_addr="0.0.0.0" --master_port=8000 train_rewardmodel.py

# Number of GPUs per GPU worker
GPUS_PER_NODE=$(python -c 'import torch; print(torch.cuda.device_count())')

# Number of GPU workers, for single-worker training, please set to 1
NNODES=${NNODES:-1}

# The rank of this worker, should be in {0, ..., WORKER_CNT-1}, for single-worker training, please set to 0
NODE_RANK=${NODE_RANK:-0}

# The ip address of the rank-0 worker, for single-worker training, please set to localhost
MASTER_ADDR=${MASTER_ADDR:-localhost}

# The port for communication
MASTER_PORT=${MASTER_PORT:-6001}
DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 --master_addr="0.0.0.0" --master_port=8080 train_rewardmodel.py


NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  python train_rewardmodel.py