# encoding: utf-8
# ddp分布式训练模型
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import torch
import logging
import gc
from utils import dataprocess
from config.arguments import parser
from model.rewardmodel import RewardModel
from config.rewardmodel_config import RewardModel_Config
from data.rewardmodel_dataset import RW_Dataset
from torch.utils.data import DistributedSampler, DataLoader
from optim.adamw import AdamWeightDecayOptimizer
from optim.sgd import SGD
from utils.common_utils import save_model_partweight
import os
import torch.distributed as dist
import torch.nn as nn
from datetime import datetime

now = datetime.now()
now_str = str(now.year) + str(now.month) + str(now.day) + str(now.hour) + str(now.minute) + str(now.second)

local_rank = int(os.environ["LOCAL_RANK"])  # torchrun传过来的运行卡id
print(f'local_rank: {local_rank}')
torch.cuda.set_device(local_rank)  # 设置当前卡id
dist.init_process_group(backend="nccl")  # 初始化分布式训练环境


def evaluate_rewardmodel(config, reward_model, tokenizer, global_step):
    eval_data = dataprocess.load_data(config.evaldata_path)
    eval_dataset = RW_Dataset(eval_data, tokenizer, config)
    eval_datasampler = DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_datasampler,
                                 batch_size=config.per_device_train_batch_size,
                                 collate_fn=eval_dataset.collate_wrapper)

    reward_model.eval()
    eval_num = 0
    eval_loss = 0
    eval_acc = 0
    with torch.no_grad():
        for batch in tqdm(eval_dataloader):
            result = reward_model(**batch)
            loss = result["loss"].item()
            acc = result["acc"]
            eval_loss += loss
            eval_acc += acc
            eval_num += len(batch["input_ids"])
            del loss
            torch.cuda.empty_cache()
            gc.collect()

        logging.info(
            f"===============eval data, global step: {global_step}, eval data size: {eval_num}, eval data loss: {eval_loss / eval_num}, eval data acc: {eval_acc / eval_num}")
        print(
            f"===============eval data, global step: {global_step}, eval data size: {eval_num}, eval data loss: {eval_loss / eval_num}, eval data acc: {eval_acc / eval_num}")


def train_rewardmodel(config):
    traindata = dataprocess.load_data(config.traindata_path)
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() and config.use_cuda else torch.device(
        "cpu")  # 当前local rank对应的卡
    config.device = device
    device_cnt = torch.cuda.device_count()
    print(f"device_cnt: {device_cnt}")
    print(f"device: {device}")
    print(f"config.model_path: {config.model_path}")
    model = AutoModel.from_pretrained(config.model_path)  # base model
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, use_fast=True, padding_side="right")
    # model.to(device)
    # model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    rewardmodel = RewardModel(tokenizer, model, config)
    rewardmodel.to(device)

    global_step = 0
    restore_step = 50
    evaluate_step = 50

    params = rewardmodel.named_parameters()
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # freeze part of parameters
    # just train last MLP ==> reward_model.weight
    for name, param in params:
        if name != "reward_model.weight":
            param.requires_grad = False

    # parallel放到requires_grad之后，放在前面的话，requires_grad会失效？
    rewardmodel = nn.parallel.DistributedDataParallel(rewardmodel, device_ids=[local_rank], output_device=local_rank)

    optimizer_grouped_parameters = [
        {'params': [p for n, p in params if
                    not any(nd in n for nd in no_decay)],
         'weight_decay': config.weigth_decay_rate},
        {'params': [p for n, p in params if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    if config.optimizer_method == "AdamWeightDecayOptimizer":
        # AdamW
        optimizer = AdamWeightDecayOptimizer(optimizer_grouped_parameters, lr=config.learning_rate)

    else:
        # SGD
        optimizer = SGD(params, lr=config.learning_rate)

    rewardmodel.train()
    optimizer.zero_grad()

    train_loss = 0
    train_acc = 0
    train_cnt = 0
    local_step = 0

    while global_step < config.global_step:
        train_dataset = RW_Dataset(traindata, tokenizer, config)
        train_datasampler = DistributedSampler(train_dataset)  # 将数据集切分后分给每张卡
        train_dataloader = DataLoader(train_dataset, sampler=train_datasampler,
                                      batch_size=config.per_device_train_batch_size,
                                      collate_fn=train_dataset.collate_wrapper)
        train_dataloader.sampler.set_epoch(global_step)  # 相当于sampler的seed，保证不同卡上的seed一致
        for batch in tqdm(train_dataloader):
            rewardmodel.train()
            train_cnt += len(batch["input_ids"]) // 2
            from datetime import datetime
            start = datetime.now()
            result = rewardmodel(**batch)
            print(f'train reward model: {datetime.now() - start}')
            loss = result["loss"]
            loss.backward()
            train_loss += loss.item()
            train_acc = result["acc"]
            local_step += 1

            logging.info(
                f"=================local rank:{local_rank}, global step: {global_step}, train loss: {train_loss / train_cnt}, train acc: {train_acc / train_cnt}")
            print(
                f"=================local rank: {local_rank}, global step: {global_step}, train loss: {train_loss / train_cnt}, train acc: {train_acc / train_cnt}")

            # 显存回收
            del loss
            torch.cuda.empty_cache()
            gc.collect()

            if local_step % config.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                local_step = 0
                global_step += 1
            # print(f"key: {rewardmodel.state_dict().keys()}")
            # 多卡运行，weight的key值会带卡的编号
            # print(f"key: {rewardmodel.state_dict().keys()}")
            if global_step % restore_step == 1:  # 只在rank=0的卡上进行weight restore   & torch.distributed.get_rank() == 0
                # print(f"if clause key: {rewardmodel.state_dict().keys()}")
                # save model
                save_model_partweight(config.output_dir, rewardmodel, weight_key="module.reward_model.weight",
                                      file_name=config.file_name + now_str + "_globalstep_" + str(
                                          global_step) + "_acc_" + str(
                                          train_acc) + "_cnt_" + str(train_cnt) + ".pt", metric=train_loss / train_cnt,
                                      max_save=config.max_save, type=config.type)

            if global_step % evaluate_step == 0:
                # evaluate
                evaluate_rewardmodel(config, rewardmodel, tokenizer, global_step)


if __name__ == '__main__':
    # import os
    # os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
    parsed_arguments = parser.parse_args()
    config = RewardModel_Config.init_from_parsed_args(parsed_arguments)
    print(f"config: {config.to_json_string()}")
    # import sys,transformers
    # print(sys.path)
    # print(transformers.__version__)
    train_rewardmodel(config)
    # traindata = dataprocess.load_data(config.traindata_path)
    #
    # device = torch.device("cuda" if torch.cuda.is_available() and config.use_cuda else "cpu")
    # model = AutoModel.from_pretrained(config.model_path)
    # tokenizer = AutoTokenizer.from_pretrained(config.model_path, use_fast=True, padding_side="right")
    # model.to(device)
