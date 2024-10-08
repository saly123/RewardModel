from argparse import ArgumentParser

parser = ArgumentParser()
# parser.add_argument("--model_path", type=str, default= "/tmp/local/qwen/Qwen2-7B-Instruct")

parser.add_argument("--model_path", type=str, default="/ll_dev/qwen/Qwen2-7B-Instruct")

parser.add_argument("--learning_rate", type=float, default=5e-4)
parser.add_argument("--per_device_train_batch_size", type=int, default=8)
parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
# parser.add_argument("--traindata_path", type=str, default="/home/powerop/work/business_rlhf/traindata_dpo_20240828.jsonl")
# parser.add_argument("--evaldata_path", type=str, default="/home/powerop/work/business_rlhf/val91_traindata_dpo_20240828.jsonl")

parser.add_argument("--traindata_path", type=str, default="/ll_dev/data/traindata_dpo_20240828.jsonl")
parser.add_argument("--evaldata_path", type=str, default="/ll_dev/data/val91_traindata_dpo_20240828.jsonl")

parser.add_argument("--global_step", type=int, default=3000)
parser.add_argument("--use_cuda", type=bool, default=True)
parser.add_argument("--optimizer_method", type=str, default="AdamWeightDecayOptimizer")
parser.add_argument("--weigth_decay_rate", type=float, default=0.01)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--max_length", type=int, default=4096)

parser.add_argument("--output_dir", type=str, default="/ll_dev/reward_model/rewardmodel_checkpoint")
# parser.add_argument("--output_dir", type=str, default="/home/powerop/work/reward_model/rewardmodel_checkpoint")

parser.add_argument("--file_name", type=str, default="train_rewardmodel")
parser.add_argument("--max_save", type=int, default=20)
parser.add_argument("--type", type=str, default="model")
parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
# parser.add_argument("--inference_checkpint", type=str,
#                     default="/home/powerop/work/reward_model/rewardmodel_checkpoint/train_rewardmodel20240829_globalstep_1_acc_0_cnt_4.pt")

parser.add_argument("--inference_checkpint", type=str,
                    default="/ll_dev/reward_model/rewardmodel_checkpoint/train_rewardmodel20249320350_globalstep_2801_acc_0_cnt_22336.pt")

parser.add_argument("--hidden_size", type=int, default=3584)
