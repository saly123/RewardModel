# reward参考代码参考：https://blog.csdn.net/quoniammm/article/details/136525409
import torch.nn
# 后续需要研究基于DPO或者是PPO的RLHF
from torch import nn
import math


class RewardModel(nn.Module):
    def __init__(self, config, compute_fp32_loss=False):
        '''
        tokenizer: model tokenizer
        basemodel: model(可以是任一transformer模型，会基于该base model的最后一层隐层输出计算reward)
        num_padding_at_begining: int, 不同的模型可能具有不同的填充或分词器行为。 具体来说，OPT 模型族中的 tokenizer 总是在开头添加一个 padding token，这会影响我们对评分 token 的选择。 因此，我们需要考虑到这一点。
        num_padding_at_begining 个人理解应该是标记特殊padding token的数量，当pad的数量超过这个数时，才认为是真正的padding
        compute_fp32_loss: bool, 是否计算fp32的loss
        '''
        super().__init__()
        self.config = config
        self.compute_fp32_loss = compute_fp32_loss
        self.reward_model = nn.Linear(self.config.hidden_size, 1, bias=False).to(self.config.device)  # [bts, hidden_size] => [bts, 1]

    # gradient checkpointing：一种节省显存的训练方式[但是会增加时间]——根据策略将计算图上的一部分激活值保存，其余部分丢弃。被保存的那部分在反向传播的时候可以直接使用，被丢弃的那部分在反向传播的时候就需要重新计算激活值
    def gradient_checkpoint_enable(self):
        self.model.gradient_checkpointing_enable()

    def gradient_checkpoint_disable(self):
        self.model.gradient_checkpointing_disable()

    def forward(self, input_tensor=None, divergenceid_tensor = None, endid_tensor = None , chosenendid_tensor = None, rejectendid_tensor = None,  past_key_value=None):
        '''
        past_key_value: 记录之前时间步的key和value，在处理较长序列或者是将模型应用到文本生成任务的时候，可以提高计算效率
        use_cache = True：等价于禁用gradient checkpoint
        past_key_value: transformers版本有更新，新的参数是Cache(tuple)

        '''
        loss = None
        # print(f'input_ids: {input_ids}')
        # print(f'attention_mask: {attention_mask}')
        # print(input_ids[0] == input_ids[1])
        #
        # print(f'input_ids shape: {input_ids.shape}')
        # print(f'attention_mask shape: {attention_mask.shape}')
        # print(f'forward model device: {self.model.device}')
        # print(f'reward model device: {self.reward_model.device}')

        rewards = self.reward_model(input_tensor).squeeze(
            -1)  # 移除tensor最后一个维度（如果最后一个维度的shape=1，则直接移除；如果不是1，则不改变原来的tensor）
        bs = input_tensor.shape[0] // 2

        chosen_mean_score = []
        rejected_mean_score = []

        # chosen_tensor = input_tensor[:bs]
        # rejected_tensor = input_tensor[bs:]
        chosen_rewards = rewards[:bs]
        rejected_rewards = rewards[bs:]

        print(f'reward device: {rewards.device}')
        print(f'input_ids device: {input_tensor.device}')
        loss = 0.
        acc = 0
        # 这部分可以挪到dataset中，collate中直接将truncate之后一样的删除掉！！
        for i in range(bs):
            chosen_reward = chosen_rewards[i]
            rejected_reward = rejected_rewards[i]
            divergence_ind = divergenceid_tensor[i].item()
            end_ind = endid_tensor[i].item()
            chosen_ind = chosenendid_tensor[i].item()
            rejected_ind = rejectendid_tensor[i].item()

            used_chosen_r = chosen_reward[divergence_ind:end_ind]  # 每个token一个reward value，长度不一样的padding也算在内
            used_rejected_r = rejected_reward[divergence_ind:end_ind]

            chosen_mean_score.append(chosen_reward[chosen_ind - 1])  # 非 padding的last token reward
            rejected_mean_score.append(rejected_reward[rejected_ind - 1])

            if self.compute_fp32_loss:
                used_chosen_r = used_chosen_r.float()
                used_rejected_r = used_rejected_r.float()
            # Bradeley-Terry model
            loss += torch.nn.functional.logsigmoid(
                used_chosen_r - used_rejected_r).mean()  # 用对应位置的token reward差值的mean作为loss。也可以将最后一个token的reward差值的mean作为loss

            # 将不同response的token response reward均值作为该response的整体reward，chose reward - reject reward > 0.5视为模型正确==>正负样本差异大
            diff = torch.nn.functional.sigmoid(used_chosen_r).mean() - torch.nn.functional.sigmoid(
                used_rejected_r).mean()
            if diff > 0.5:
                acc += 1
            return {"loss": - loss, "acc": acc, "chosen_mean_score": chosen_mean_score,
                    "rejected_mean_score": rejected_mean_score, "used_chosen_r": used_chosen_r,
                    "used_rejected_r": used_rejected_r}
