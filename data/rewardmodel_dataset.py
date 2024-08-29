from torch.utils.data import Dataset
import torch


class RW_Dataset(Dataset):
    def __init__(self, data, tokenizer, config, truncation=True, padding="max_length"):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = config.max_length
        self.truncation = truncation
        self.padding = padding
        self.config = config

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
            'input_ids': inputs.input_ids,
            'attention_mask': inputs.attention_mask
        }

    def collate_wrapper(self, batch):
        '''
        :param batch: {"query":,"response","rejected_response"}
        :return:
        '''
        chosen_list = []
        chosen_attention_mask = []
        reject_list = []
        reject_attention_mask = []
        oridata_list = []

        for data in batch:
            chosen_text = data["query"] + data["response"]
            reject_text = data["query"] + data["rejected_response"]
            print(f"chosen_text: {chosen_text}")
            print(f"reject_text: {reject_text}")
            chosen_list.append(self.__encode_token__(chosen_text)["input_ids"])
            chosen_attention_mask.append(self.__encode_token__(chosen_text)["attention_mask"])

            reject_list.append(self.__encode_token__(reject_text)["input_ids"])
            reject_attention_mask.append(self.__encode_token__(reject_text)["attention_mask"])

        # tensor

        # return {"chosen_input": torch.as_tensor(chosen_list, dtype=torch.long),
        #         "chosen_mask": torch.as_tensor(chosen_attention_mask, dtype=torch.long),
        #         "reject_input": torch.as_tensor(reject_list, dtype=torch.long),
        #         "reject_mask": torch.as_tensor(reject_attention_mask, dtype=torch.long)}

        input_tensor = torch.cat((torch.as_tensor(chosen_list , dtype=torch.long),torch.as_tensor(reject_list, dtype=torch.long)),dim = 0)
        input_mask = torch.cat((torch.as_tensor(chosen_attention_mask, dtype=torch.long),torch.as_tensor(reject_attention_mask, dtype=torch.long)),dim = 0)
        return {"input_ids": input_tensor.to(self.config.device), "attention_mask": input_mask.to(self.config.device)}
