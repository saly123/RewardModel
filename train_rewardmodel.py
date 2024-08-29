# encoding: utf-8
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import torch
import logging

from utils import dataprocess
from config.arguments import parser
from model.rewardmodel import RewardModel
from config.rewardmodel_config import RewardModel_Config
from data.rewardmodel_dataset import RW_Dataset
from torch.utils.data import RandomSampler, DataLoader
from optim.adamw import AdamWeightDecayOptimizer
from optim.sgd import SGD
from utils.common_utils import save_model_partweight


def evaluate_rewardmodel(config, reward_model, tokenizer, global_step):
    eval_data = dataprocess.load_data(config.traindata_path)
    eval_dataset = RW_Dataset(eval_data, tokenizer, config)
    eval_datasampler = RandomSampler(eval_dataset)
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

        logging.info(
            f"===============eval data, global step: {global_step}, eval data size: {eval_num}, eval data loss: {eval_loss / eval_num}, eval data acc: {eval_acc / eval_num}")


def train_rewardmodel(config):
    traindata = dataprocess.load_data(config.traindata_path)

    device = torch.device("cuda" if torch.cuda.is_available() and config.use_cuda else "cpu")
    print(f"config.model_path: {config.model_path}")
    model = AutoModel.from_pretrained(config.model_path)
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, use_fast=True, padding_side="right")
    model.to(device)
    rewardmodel = RewardModel(tokenizer, model)
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

    while global_step < config.global_step:
        train_dataset = RW_Dataset(traindata, tokenizer, config)
        train_datasampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_datasampler,
                                      batch_size=config.per_device_train_batch_size,
                                      collate_fn=train_dataset.collate_wrapper)
        for batch in tqdm(train_dataloader):
            rewardmodel.train()
            train_cnt += len(batch["input_ids"])
            result = rewardmodel(**batch)
            loss = result["loss"]
            loss.backward()
            train_loss += loss.item()
            train_acc = result["acc"]
            global_step += 1
            logging.info(
                f"=================global step: {global_step}, train loss: {train_loss / train_cnt}, train acc: {train_acc / train_cnt}")

            if global_step % evaluate_step == 0:
                # evaluate
                evaluate_rewardmodel(config, rewardmodel, tokenizer)
            if global_step % restore_step == 0:
                # save model
                save_model_partweight(config.output_dir, model, weight_key="reward_model.weight",
                                      file_name=config.file_name + "_globalstep_"+ str(global_step) +"_acc_"+ str(train_acc) +"_cnt_" +str(train_cnt)+ ".pt",
                                      metric= train_loss / train_cnt, max_save=config.max_save,
                                      type=config.type)




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
