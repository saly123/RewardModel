from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--model_path", type=str, default= "/tmp/local/qwen/Qwen2-7B-Instruct")
parser.add_argument("--learning_rate", type=float, default=5e-5)
parser.add_argument("--per_device_train_batch_size", type=int, default=4)
parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
parser.add_argument("--traindata_path", type=str, default="/tmp/local/qwen/data/train.json")
parser.add_argument("--evaldata_path", type=str, default="/tmp/local/qwen/data/eval.json")
parser.add_argument("--global_step", type=int, default=100)
parser.add_argument("--use_cuda", type=bool, default=True)
parser.add_argument("--optimizer_method", type=str, default="AdamWeightDecayOptimizer")
