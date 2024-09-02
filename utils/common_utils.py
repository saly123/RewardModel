import logging
import os
import re
import torch
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve
from transformers import AutoModel, AutoTokenizer


def restore_from_checkpoint(model, dir_checkpoint, strict=True, type='model'):
    if type == 'model':
        pattern = re.compile(r'pytorch_model_([0-9].*)\.pt')
    elif type == 'optimizer':
        pattern = re.compile(r'pytorch_optimizer_([0-9].*)\.pt')
    else:
        raise ValueError('Argument `type` must be either `model` or `optimizer`.')

    record_checkpoint_model = os.path.join(dir_checkpoint, type)
    with open(record_checkpoint_model, 'r', encoding='utf-8') as r:
        file_pts = r.readlines()
    restore_pt = file_pts[-1].strip().split()[0]
    match = re.search(pattern, restore_pt)
    global_step = int(match.groups()[0])

    if type == 'model':
        state_dict = torch.load(restore_pt)
        model.load_state_dict(state_dict, strict=strict)
    elif type == 'optimizer':
        state_dict = torch.load(restore_pt)
        model.load_state_dict(state_dict)
    else:
        raise ValueError('Argument `type` must be either `model` or `optimizer`.')

    logging.info(f'{type} weights restored from {restore_pt}.')
    return global_step


def save_pytorch_model(output_dir, model, file_name, metric, max_save, type):
    logging.info(f'** ** * Saving {type}: {file_name} * ** ** ')
    os.makedirs(output_dir, exist_ok=True)
    file_checkpoint = os.path.join(output_dir, type)
    if os.path.exists(file_checkpoint):
        with open(file_checkpoint, 'r', encoding='utf-8') as r:
            file_lists = r.readlines()
    else:
        file_lists = []

    file_lists.sort(key=lambda line: float(line.split()[1]), reverse=True)
    while len(file_lists) > max_save:
        file = file_lists.pop()
        try:
            os.remove(file.split()[0].strip())
        except:
            logging.info(f'Failed to remove ckpt {file}.')

    model_to_save = model.module if hasattr(model, 'module') else model
    file_ckpt = os.path.join(output_dir, file_name)
    torch.save(model_to_save.state_dict(), file_ckpt)
    file_lists.sort()
    file_lists.append(file_ckpt + ' ' + str(metric) + '\n')
    open(file_checkpoint, 'w').write(''.join(file_lists))

def save_model_partweight(output_dir, model, weight_key, file_name, metric, max_save, type):
    logging.info(f'** ** * Saving {type}: {file_name} * ** ** ')
    print(f'save output_dir: {output_dir}')
    os.makedirs(output_dir, exist_ok=True)
    file_checkpoint = os.path.join(output_dir, type)
    if os.path.exists(file_checkpoint):
        with open(file_checkpoint, 'r', encoding='utf-8') as r:
            file_lists = r.readlines()
    else:
        file_lists = []

    file_lists.sort(key=lambda line: float(line.split()[1]), reverse=True)
    while len(file_lists) > max_save:
        file = file_lists.pop()
        try:
            os.remove(file.split()[0].strip())
        except:
            logging.info(f'Failed to remove ckpt {file}.')

    # model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save = model
    file_ckpt = os.path.join(output_dir, file_name)
    torch.save(model_to_save.state_dict()[weight_key], file_ckpt)
    file_lists.sort()
    file_lists.append(file_ckpt+ ' ' + str(metric) + '\n')
    open(file_checkpoint, 'w').write(''.join(file_lists))
    

def restore_partweight_from_checkpoint(model, config,  dir_checkpoint_pt):
    state_dict = torch.load(dir_checkpoint_pt)
    load_pt_key = "reward_model.weight"
    basemodel = AutoModel.from_pretrained(config.model_path)
    basemodel_weight = basemodel.state_dict()
    model_statedict = model.state_dict()

    for k,v in model_statedict.items():
        if k == load_pt_key:
            model_statedict[k] = state_dict
        else:
            model_statedict[k] = basemodel_weight[k.replace("model.","")]
    model.load_state_dict(model_statedict)







def generate_roc_curve(predict_outputs, labels):
    fpr, tpr, thresholds_roc = roc_curve(labels, predict_outputs, pos_label=1)
    plt.plot(fpr, tpr, marker='.')
    plt.show()
    auc_value = auc(fpr, tpr)
    print(auc_value)


def generate_pr_curve(predict_outputs, labels):
    precision, recall, thresholds_pr = precision_recall_curve(labels, predict_outputs)

    plt.plot(precision, recall, marker='.')
    plt.show()
