import json


class RewardModel_Config(object):
    def __init__(self):
        self.model_path = "/tmp/local/qwen/Qwen2-7B-Instruct"
        self.base_model = None
        self.specidal_tokens = ""
        self.tokenizer = None
        self.learning_rate = None
        self.per_device_train_batch_size = 1
        self.per_device_eval_batch_size = 1
        self.traindata_path = ""
        self.evaldata_path = ""
        self.global_step = 100
        self.use_cuda = False
        self.optimizer_method = "AdamWeightDecayOptimizer"

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
