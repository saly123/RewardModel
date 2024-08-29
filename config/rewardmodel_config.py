import json


class RewardModel_Config(object):
    def __init__(self):
        self.model_path = "/tmp/local/qwen/Qwen2-7B-Instruct"
        self.base_model = None
        self.specidal_tokens = ""
        self.tokenizer = None
        self.learning_rate = 5e-5
        self.per_device_train_batch_size = 1
        self.per_device_eval_batch_size = 1
        self.traindata_path = "/home/powerop/work/business_rlhf/traindata_dpo_20240828.jsonl"
        self.evaldata_path = "/home/powerop/work/business_rlhf/val91_traindata_dpo_20240828.jsonl"
        self.global_step = 100
        self.use_cuda = False
        self.optimizer_method = "AdamWeightDecayOptimizer"
        self.weigth_decay_rate = 0.01
        self.device = "cpu"
        self.max_length = 4096

    @classmethod
    def init_from_parsed_args(cls, parsed_args):
        cfg = cls()
        args_dict = vars(parsed_args)
        for k in args_dict:
            if args_dict[k] is not None:
                setattr(cfg, k, args_dict[k])
        return cfg

    @classmethod
    def init_from_json(cls, file_json):
        cfg = cls()
        args_dict = dict(json.load(open(file_json, "r", encoding="utf-8")))
        for k in args_dict:
            if args_dict[k] is not None:
                setattr(cfg, k, args_dict[k])
        return cfg

    def to_json_string(self):
        return json.dumps(self.__dict__, indent=4)
