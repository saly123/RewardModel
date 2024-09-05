from data.rewardmodel_dataset import RW_Dataset
from model.rewardmodel import RewardModel
from config.rewardmodel_config import RewardModel_Config
from transformers import AutoModel, AutoTokenizer
from utils.common_utils import restore_partweight_from_checkpoint
from tqdm import tqdm
from utils import dataprocess
from torch.utils.data import RandomSampler, DataLoader
import torch


def inference(datapath, reward_model):
    eval_data = dataprocess.load_data(datapath)
    eval_data = eval_data[:10]
    print(f'eval_data : {eval_data}')

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
            eval_num += len(batch["input_ids"]) // 2
            chosen_reward = result["used_chosen_r"]
            reject_reward = result["used_rejected_r"]
            print(f'chosen reward: {chosen_reward}')
            print(f'reject reward: {reject_reward}')
        print(
            f"===============eval data, eval data size: {eval_num}, eval data loss: {eval_loss / eval_num}, eval data acc: {eval_acc / eval_num}")
        print(f"================chosen_reward: {chosen_reward}, reject_reward: {reject_reward}")


if __name__ == '__main__':
    config = RewardModel_Config()
    basemodel = AutoModel.from_pretrained(config.model_path)
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, use_fast=True, padding_side="right")
    rewardmodel = RewardModel(tokenizer, basemodel, config)
    restore_partweight_from_checkpoint(rewardmodel, config, config.inference_checkpint)

    inference_datapath = config.evaldata_path

    inference(inference_datapath, rewardmodel)
