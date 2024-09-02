#!/bin/bash
torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 --master_addr="0.0.0.0" --master_port=8000 train_rewardmodel.py