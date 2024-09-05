# reward参考代码参考：https://blog.csdn.net/quoniammm/article/details/136525409
import torch.nn
# 后续需要研究基于DPO或者是PPO的RLHF
from torch import nn
import math


class RewardModel(nn.Module):
    def __init__(self, tokenizer, basemodel,config, num_padding_at_begining=0, compute_fp32_loss=False):
        '''
        tokenizer: model tokenizer
        basemodel: model(可以是任一transformer模型，会基于该base model的最后一层隐层输出计算reward)
        num_padding_at_begining: int, 不同的模型可能具有不同的填充或分词器行为。 具体来说，OPT 模型族中的 tokenizer 总是在开头添加一个 padding token，这会影响我们对评分 token 的选择。 因此，我们需要考虑到这一点。
        num_padding_at_begining 个人理解应该是标记特殊padding token的数量，当pad的数量超过这个数时，才认为是真正的padding
        compute_fp32_loss: bool, 是否计算fp32的loss
        '''
        super().__init__()
        self.tokenizer = tokenizer
        self.model = basemodel
        self.model.to(config.device)
        # print(f'base model device: {self.model.device}')
        self.config = config
        self.num_padding_at_begining = num_padding_at_begining  # 在序列的开始处添加padding token的数量。
        self.compute_fp32_loss = compute_fp32_loss
        self.reward_model = nn.Linear(self.config.hidden_size, 1, bias=False).to(self.config.device)  # [bts, hidden_size] => [bts, 1]
        self.PAD_ID = tokenizer.pad_token_id

    # gradient checkpointing：一种节省显存的训练方式[但是会增加时间]——根据策略将计算图上的一部分激活值保存，其余部分丢弃。被保存的那部分在反向传播的时候可以直接使用，被丢弃的那部分在反向传播的时候就需要重新计算激活值
    def gradient_checkpoint_enable(self):
        self.model.gradient_checkpointing_enable()

    def gradient_checkpoint_disable(self):
        self.model.gradient_checkpointing_disable()

    def forward(self, input_ids=None, past_key_value=None, attention_mask=None, position_ids=None, head_mask=None,
                inputs_embeds=None, use_cache=False):
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

        from datetime import datetime
        start = datetime.now()
        transformer_outputs = self.model(input_ids, attention_mask=attention_mask,
                                         inputs_embeds=inputs_embeds,
                                         use_cache=use_cache)  # qwen2 model没有入参past_key_value
        print(f'forward model time: {datetime.now() - start}')
        # bts × seq_len × hidden_size
        # to device加速
        hidden_states = transformer_outputs[0].to(self.config.device)  # 只取模型的hidden_states，past_key_value先不取
        # bts*seq_len*1 ==> bts*seq_len（每一个token都有一个reward值）

        start = datetime.now()
        rewards = self.reward_model(hidden_states).squeeze(
            -1)  # 移除tensor最后一个维度（如果最后一个维度的shape=1，则直接移除；如果不是1，则不改变原来的tensor）

        print(f'reward model time: {datetime.now() - start}')

        chosen_mean_score = []
        rejected_mean_score = []

        # 由于 reward的模型是一个输入，对应一个chosen和一个rejected
        bs = input_ids.shape[0] // 2  # rejected和chosen各一半
        seq_len = input_ids.shape[1]

        chosen_ids = input_ids[:bs]
        rejected_ids = input_ids[bs:]
        chosen_rewards = rewards[:bs]
        rejected_rewards = rewards[bs:]

        print(f'reward device: {rewards.device}')
        print(f'input_ids device: {input_ids.device}')

        # 计算rejected和chosen的pairwise loss ——只对padding之前的chosen和rejected的不同token 计算loss 用于backpropagation


        # 下面的代码耗时较长，建议优化到dataset！！
        loss = 0.
        acc = 0
        # 这部分可以挪到dataset中，collate中直接将truncate之后一样的删除掉！！
        for i in range(bs):
            chosen_id = chosen_ids[i]
            rejected_id = rejected_ids[i]
            # print(f"chosen id: {chosen_id}")
            # print(f"reject id: {rejected_id}")

            chosen_reward = chosen_rewards[i]
            rejected_reward = rejected_rewards[i]
            chosen_inds = (chosen_id == self.PAD_ID).nonzero()  # chosen内容中padding的所有index
            # print(f'chosen_inds: {chosen_inds}')
            chosen_ind = chosen_inds[self.num_padding_at_begining].item() if len(
                chosen_inds) > self.num_padding_at_begining else seq_len  # chosen中剔除开头默认的padding，之后的第一个padding位置——表示句子结束
            # print(f'chosen_inds: {chosen_inds}')
            check_divergence = (chosen_id != rejected_id).nonzero()  # 找到不同的token的下标index

            if len(check_divergence) == 0:
                # 没有不同的token
                rejected_ind = chosen_ind  # 第一个非padding的tokenid
                end_ind = rejected_reward.shape[-1]
                divergence_ind = end_ind - 1

            else:
                # 剔除由padding导致的不同token
                rejected_inds = (rejected_id == self.PAD_ID).nonzero()
                rejected_ind = rejected_inds[self.num_padding_at_begining].item() if len(
                    rejected_inds) > self.num_padding_at_begining else seq_len  # reject中非开头默认的padding，后面的第一个padding（表示到句子结束）
                end_ind = max(chosen_ind, rejected_ind)  # 取两个里面最大的结束位置作为实际的结束位置
                divergence_ind = check_divergence[0]  # 开始不一样的index

            # print(f'check_divergence: {check_divergence}')
            # print(f'chosen_id : {chosen_id}')
            # print(f'rejected_id: {rejected_id}')
            # 
            # print(f'chosen_reward : {chosen_reward}')
            # print(f'rejected_reward: {rejected_reward}')

            assert divergence_ind > 0

            # 开始计算chosen_reward和rejected_reward不相同部分的loss

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
