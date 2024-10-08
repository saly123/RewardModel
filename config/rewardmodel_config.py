import json


class RewardModel_Config(object):
    def __init__(self):
        # self.model_path = "/tmp/local/qwen/Qwen2-7B-Instruct"
        self.model_path = "/ll_dev/qwen/Qwen2-7B-Instruct"
        self.base_model = None
        self.specidal_tokens = ""
        self.tokenizer = None
        self.learning_rate = 5e-4
        self.per_device_train_batch_size = 16
        self.per_device_eval_batch_size = 16
        # self.traindata_path = "/home/powerop/work/business_rlhf/traindata_dpo_20240828.jsonl"
        # self.evaldata_path = "/home/powerop/work/business_rlhf/val91_traindata_dpo_20240828.jsonl"

        self.traindata_path = "/ll_dev/data/traindata_dpo_20240828.jsonl"
        self.evaldata_path = "/ll_dev/data/val91_traindata_dpo_20240828.jsonl"

        self.global_step = 3000
        self.use_cuda = True
        self.optimizer_method = "AdamWeightDecayOptimizer"
        self.weigth_decay_rate = 0.01
        self.device = "cpu"
        self.max_length = 4096
        # self.output_dir = "/home/powerop/work/reward_model/rewardmodel_checkpoint"
        self.output_dir = "/ll_dev/reward_model/rewardmodel_checkpoint"
        self.file_name = "train_rewardmodel20240829"
        self.max_save = 20
        self.type = "model"
        self.gradient_accumulation_steps = 1
        # self.inference_checkpint = "/home/powerop/work/reward_model/rewardmodel_checkpoint/train_rewardmodel20240829_globalstep_1_acc_0_cnt_4.pt"
        self.inference_checkpint = "/ll_dev/reward_model/rewardmodel_checkpoint/train_rewardmodel20249320350_globalstep_2801_acc_0_cnt_22336.pt"

        self.hidden_size = 3584



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
