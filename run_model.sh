#!/bin/bash
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  python train_rewardmodel.py