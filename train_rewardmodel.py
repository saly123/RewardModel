# encoding: utf-8
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import torch
from torch.optim import adamw as AdamWeightDecayOptimizer
from torch.optim import sgd as SGD
import logging

from utils import dataprocess
from config.arguments import parser
from model.rewardmodel import RewardModel
from config.rewardmodel_config import RewardModel_Config
from data.rewardmodel_dataset import RW_Dataset
from torch.utils.data import RandomSampler, DataLoader



# from optim.adamw import AdamWeightDecayOptimizer
# from optim.sgd import SGD


def evaluate_rewardmodel(config, reward_model, tokenizer, global_step):
    eval_data = dataprocess.load_data(config.traindata_path)
    eval_dataset = RW_Dataset(eval_data, tokenizer)
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
            acc = result["acc"].item()
            eval_loss += loss
            eval_acc += acc
            eval_num += len(batch["input_ids"])

        logging.info(
            f"===============eval data, global step: {global_step}, eval data size: {eval_num}, eval data loss: {eval_loss / eval_num}, eval data acc: {eval_acc / eval_num}")


def train_rewardmodel(config):
    traindata = dataprocess.load_data(config.traindata_path)

    device = torch.device("cuda" if torch.cuda.is_available() and config.use_cuda else "cpu")
    model = AutoModel.from_pretrained(config.model_path)
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, use_fast=True, padding_side="right")
    model.to(device)
    rewardmodel = RewardModel(model, tokenizer)
    rewardmodel.to(device)

    params = rewardmodel.named_parameters()
    # freeze part of parameters
    # just train last MLP ==> reward_model.weight
    for name, param in rewardmodel.named_parameters():
        if name != "reward_model.weight":
            param.requires_grad = False

    if config.optimizer_method == "AdamWeightDecayOptimizer":
        # AdamW
        optimizer = AdamWeightDecayOptimizer(params, lr=config.learning_rate)

    else:
        # SGD
        optimizer = SGD(params, lr=config.learning_rate)

    global_step = 0
    restore_step = 0
    rewardmodel.train()
    optimizer.zero_grad()

    train_loss = 0
    train_acc = 0
    train_cnt = 0

    while global_step < config.global_step:
        train_dataset = RW_Dataset(traindata, tokenizer)
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
            train_acc = result["acc"].item()
            global_step += 1
            logging.info(
                f"=================global step: {global_step}, train loss: {train_loss / train_cnt}, train acc: {train_acc / train_cnt}")

            if global_step % restore_step == 0:
                # evaluate
                evaluate_rewardmodel(config, rewardmodel, tokenizer)


if __name__ == '__main__':
    # import os
    # os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
    parsed_arguments = parser.parse_args()
    config = RewardModel_Config.init_from_parsed_args(parsed_arguments)
    train_rewardmodel(config)
    # traindata = dataprocess.load_data(config.traindata_path)
    # 
    # device = torch.device("cuda" if torch.cuda.is_available() and config.use_cuda else "cpu")
    # model = AutoModel.from_pretrained(config.model_path)
    # tokenizer = AutoTokenizer.from_pretrained(config.model_path, use_fast=True, padding_side="right")
    # model.to(device)