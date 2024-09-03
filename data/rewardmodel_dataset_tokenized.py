from torch.utils.data import Dataset
import torch


class RW_Dataset(Dataset):
    def __init__(self, data, tokenizer, model, config, num_padding_at_begining, truncation=True, padding="max_length"):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = config.max_length
        self.truncation = truncation
        self.padding = padding
        self.config = config
        self.model = model
        self.model.to(self.config.device)
        self.PAD = self.tokenizer.pad_token_id
        self.num_padding_at_begining = num_padding_at_begining

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        return text

    def __encode_token__(self, text):
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True
        )
        return {
            'ori_data': text,
            'input_ids': torch.as_tensor(inputs.input_ids, dtype=torch.long).to(self.config.device),
            'attention_mask': torch.as_tensor(inputs.attention_mask, dtype=torch.long).to(self.config.device)
        }

    def __convert_inputid(self, chosen_list, reject_list, chosen_attention_mask, reject_attention_mask):
        bts = len(chosen_list)
        for i in range(bts):
            chosen_list[i] = chosen_list[i][:self.max_length]
            reject_list[i] = reject_list[i][:self.max_length]
            chosen_attention_mask[i] = chosen_attention_mask[i][:self.max_length]
            reject_attention_mask[i] = reject_attention_mask[i][:self.max_length]

        input_tensor = torch.cat(
            (torch.as_tensor(chosen_list, dtype=torch.long), torch.as_tensor(reject_list, dtype=torch.long)), dim=0)
        return input_tensor

    def __convert_token_hidden_state__(self, chosen_token, chosen_atten_mask, reject_token, reject_atten_mask):
        if torch.equal(chosen_token, reject_token):
            return None
        seq_len = chosen_token.shape[-1]

        chosen_hidden_state = self.model(chosen_token, attention_mask=chosen_atten_mask)[0].to(self.config.device)
        reject_hidden_state = self.model(reject_token, attention_mask=reject_atten_mask)[0].to(self.config.device)
        chosen_padding_ids = (chosen_token == self.PAD).nonzero()
        chosen_end_id = chosen_padding_ids[self.num_padding_at_begining] if len(
            chosen_padding_ids) > self.num_padding_at_begining else seq_len

        divergence = (chosen_token != reject_token).nonzero()

        if len(divergence) == 0:
            reject_end_id = chosen_end_id
            end_id = chosen_token.shape[-1]
            divergence_id = end_id - 1
        else:
            reject_padding_ids = (reject_token == self.PAD).nonzero()
            reject_end_id = reject_padding_ids[self.num_padding_at_begining] if len(
                reject_padding_ids) > self.num_padding_at_begining else seq_len
            end_id = max(chosen_end_id, reject_end_id)
            divergence_id = divergence[0]

        assert divergence_id > 0
        return chosen_hidden_state, reject_hidden_state, divergence_id, end_id, chosen_end_id, reject_end_id

    def collate_wrapper(self, batch):
        '''
        :param batch: {"query":,"response","rejected_response"}
        :return:
        '''
        chosen_list = []
        chosen_attention_mask = []
        reject_list = []
        reject_attention_mask = []
        chosen_hidden = []
        reject_hidden = []
        divergence_ids = []
        end_ids = []
        chosen_end_ids = []
        reject_end_ids = []
        oridata_list = []
        from datetime import datetime

        # now = datetime.now()

        for data in batch:
            chosen_text = data["query"] + "输出是：" + data["response"]
            reject_text = data["query"] + "输出是：" + data["rejected_response"]
            # chosen_text = data["response"]
            # reject_text = data["rejected_response"]
            # print(f"chosen_text: {chosen_text}")
            # print(f"reject_text: {reject_text}")

            # encode_start = datetime.now()

            # tensor
            chosen_id = self.__encode_token__(chosen_text)["input_ids"]
            chosen_mask = self.__encode_token__(chosen_text)["attention_mask"]

            reject_id = self.__encode_token__(reject_text)["input_ids"]
            reject_mask = self.__encode_token__(reject_text)["attention_mask"]
            divergence_hidden_state = self.__convert_token_hidden_state__(chosen_id, chosen_mask, reject_id,
                                                                          reject_mask)
            if divergence_hidden_state is None:
                continue

            # chosen_list.append(chosen_id)
            # chosen_attention_mask.append(chosen_mask)
            # reject_list.append(reject_id)
            # reject_attention_mask.append(reject_mask)
            chosen_hidden.append(divergence_hidden_state[0])
            reject_hidden.append(divergence_hidden_state[1])
            divergence_ids.append(divergence_hidden_state[2])
            end_ids.append(divergence_hidden_state[3])
            chosen_end_ids.append(divergence_hidden_state[4])
            reject_end_ids.append(divergence_hidden_state[5])

        input_tensor = torch.cat(
            (torch.as_tensor(chosen_hidden, dtype=torch.long), torch.as_tensor(reject_hidden, dtype=torch.long)), dim=0)
        divergenceid_tensor = torch.as_tensor(divergence_ids, dtype=torch.long)
        endid_tensor = torch.as_tensor(end_ids, dtype=torch.long)
        chosenendid_tensor = torch.as_tensor(chosen_end_ids, dtype=torch.long)
        rejectendid_tensor = torch.as_tensor(reject_end_ids, dtype=torch.long)

        return {"input_tensor": input_tensor.to(self.config.device),
                "divergenceid_tensor": divergenceid_tensor.to(self.config.device),
                "endid_tensor": endid_tensor.to(self.config.device),
                "chosenendid_tensor": chosenendid_tensor.to(self.config.device),
                "rejectendid_tensor": rejectendid_tensor.to(self.config.device)}
