# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import codecs
import json
import random
import argparse
import pandas as pd
from tqdm import tqdm, trange
from sklearn import metrics, preprocessing
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import BertModel, BertTokenizer, AdamW, WarmupLinearSchedule
from transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
import matplotlib
from matplotlib import pyplot as plt
from dataset import *
from util import *

def val(filename, model, processor, args, tokenizer, device):
    '''模型验证

    Args:
        model: 模型
	processor: 数据读取方法
	args: 参数表
	label_list: 所有可能类别
	tokenizer: 分词方法
	device

    Returns:
        f1: F1值
    '''
 # Run prediction for full data
    model.eval()
    entity_data, dev_data, label_list = processor.get_dev_examples(os.path.join(args.data_dir, filename))
    eval_dataloader = get_dataloader(entity_data, args, args.eval_batch_size, label_list, tokenizer)
    probs = None
    gt = np.zeros((0,), dtype=np.int32)
    acc = 0
    entity_embs = []
    entity_label_ids = []
    for batch in eval_dataloader:
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        with torch.no_grad():
            _, pooled_output = model(input_ids, input_mask, segment_ids)
            entity_embs.append(pooled_output)
            entity_label_ids.append(label_ids)
    entity_embs = torch.cat(tuple(entity_embs))
    # entity_embs = F.normalize(entity_embs)
    entity_label_ids = torch.cat(tuple(entity_label_ids))   # index2labelid
    entity_label_ids = entity_label_ids.sort()[1]           # labelid2index

    eval_dataloader = get_dataloader(dev_data, args, args.eval_batch_size, label_list, tokenizer)
    for batch in eval_dataloader:
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        with torch.no_grad():
            _, pooled_output = model(input_ids, input_mask, segment_ids)
            # pooled_output = F.normalize(pooled_output)
            logits = F.softmax(pooled_output.mm(entity_embs.t()))
            prob, pred = logits.max(1)
            ground_truth = entity_label_ids[label_ids]
            acc += (pred == ground_truth).sum().item()
            if probs is None:
                probs = logits.cpu().numpy()
            else:
                probs = np.vstack((probs, logits.cpu().numpy()))
            gt = np.hstack((gt, ground_truth.cpu().numpy()))
    acc = acc / len(dev_data)
    print(acc)
    print(prob)
    one_hot_encoder = preprocessing.OneHotEncoder(sparse=False)
    gt = one_hot_encoder.fit_transform(gt.reshape(-1, 1))
    auc = metrics.roc_auc_score(gt, probs, average='macro')
    print(auc)
    # f1 = np.mean(metrics.f1_score(predict, gt, average=None))
    # print(f1)

    return auc


def test(filename, model, processor, args, tokenizer, device):
    '''模型验证

    Args:
        model: 模型
	processor: 数据读取方法
	args: 参数表
	label_list: 所有可能类别
	tokenizer: 分词方法
	device

    Returns:
        f1: F1值
    '''
 # Run prediction for full data
    model.eval()
    entity_data, dev_data, label_list = processor.get_dev_examples(os.path.join(args.data_dir, filename))
    eval_dataloader = get_dataloader(entity_data, args, args.eval_batch_size, label_list, tokenizer)
    probs = None
    gt = np.zeros((0,), dtype=np.int32)
    acc = 0
    entity_embs = []
    entity_label_ids = []
    for batch in eval_dataloader:
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        with torch.no_grad():
            _, pooled_output = model(input_ids, input_mask, segment_ids)
            entity_embs.append(pooled_output)
            entity_label_ids.append(label_ids)
    entity_embs = torch.cat(tuple(entity_embs))
    entity_label_ids = torch.cat(tuple(entity_label_ids))   # index2labelid
    entity_label_ids = entity_label_ids.sort()[1]           # labelid2index

    eval_dataloader = get_dataloader(dev_data, args, args.eval_batch_size, label_list, tokenizer)
    for batch in eval_dataloader:
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        with torch.no_grad():
            _, pooled_output = model(input_ids, input_mask, segment_ids)
            # pooled_output = F.normalize(pooled_output)
            logits = F.softmax(pooled_output.mm(entity_embs.t()))
            prob, pred = logits.max(1)
            ground_truth = entity_label_ids[label_ids]
            acc += (pred == ground_truth).sum().item()
            if probs is None:
                probs = logits.cpu().numpy()
            else:
                probs = np.vstack((probs, logits.cpu().numpy()))
            gt = np.hstack((gt, ground_truth.cpu().numpy()))
    acc = acc / len(dev_data)
    print(acc)
    print(prob)
    f_scores = []
    best_f, best_threshold = 0, -1
    thresholds = []
    one_hot_encoder = preprocessing.OneHotEncoder(sparse=False)
    for threshold in range(75, 100):
        threshold /= 100
        prob = np.max(probs, 1)
        pred = np.argmax(probs, 1)
        to_pred = prob > args.threshold
        pred = pred[to_pred]
        prob = prob[to_pred]
        y_true = gt[to_pred]
        if len(y_true) == 0:
            continue
        precision = metrics.precision_score(y_true, pred, average='macro')
        if precision < 0.9: continue
        f = metrics.fbeta_score(y_true, pred, 0.8, average='macro')
        if best_f < f:
            best_f = f
            best_threshold = threshold
        f_scores.append(f)
        thresholds.append(threshold)

    print(best_f, best_threshold)
    plt.plot(thresholds, f_scores)
    plt.show()


    return best_f


def classify(filename, model, processor, args, tokenizer, device):
    '''模型测试

    Args:
        model: 模型
	processor: 数据读取方法
	args: 参数表
	label_list: 所有可能类别
	tokenizer: 分词方法
	device

    '''
    model.eval()
    entity_data, test_data, label_list = processor.get_test_examples(os.path.join(args.data_dir, filename))
    eval_dataloader = get_dataloader(entity_data, args, args.eval_batch_size, label_list, tokenizer)

    entity_embs = []
    for batch in tqdm(eval_dataloader):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        with torch.no_grad():
            _, pooled_output = model(input_ids, input_mask, segment_ids)
            entity_embs.append(pooled_output)
    entity_embs = torch.cat(tuple(entity_embs))

    preds, probs = [], []
    test_dataloader = get_dataloader(test_data, args, args.eval_batch_size, label_list, tokenizer)
    for index, batch in enumerate(tqdm(test_dataloader)):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        with torch.no_grad():
            _, pooled_output = model(input_ids, input_mask, segment_ids)
            # pooled_output = F.normalize(pooled_output)
            logits = F.softmax(pooled_output.mm(entity_embs.t()))
            prob, pred = logits.max(1)
            # to_pred = prob > args.threshold
            preds += pred.tolist()
            probs += prob.tolist()
    preds = [entity_data[i].guid for i in preds]
    guids = [example.guid for example in test_data]
    df = pd.DataFrame(index=guids, data={'target': preds, 'prob': probs})
    df.to_csv('data/pred.csv', index=True, header=False)


def main():
    # ArgumentParser对象保存了所有必要的信息，用以将命令行参数解析为相应的python数据类型
    parser = argparse.ArgumentParser()

    # required parameters
    # 调用add_argument()向ArgumentParser对象添加命令行参数信息，这些信息告诉ArgumentParser对象如何处理命令行参数
    parser.add_argument("--data_dir",
                        default='data',
                        type=str,
                        # required = True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model",
                        default='bert-base-multilingual-cased',
                        type=str,
                        # required = True,
                        help="choose [bert-base-chinese] mode.")
    parser.add_argument("--bert_path",
                        default='./model/bert-base-multilingual-cased/',
                        type=str,
                        # required = True,
                        help="choose [bert-base-chinese] mode.")
    parser.add_argument("--task_name",
                        default='MyPro',
                        type=str,
                        # required = True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default='checkpoints/',
                        type=str,
                        # required = True,
                        help="The output directory where the model checkpoints will be written")
    parser.add_argument("--model_save_pth",
                        default='checkpoints/bert_classification.pth',
                        type=str,
                        # required = True,
                        help="The output directory where the model checkpoints will be written")

    # other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="字符串最大长度")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="训练模式")
    parser.add_argument("--do_eval",
                        default=True,
                        action='store_true',
                        help="验证模式")
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="英文字符的大小写转换，对于中文来说没啥用")
    parser.add_argument("--train_batch_size",
                        default=64,
                        type=int,
                        help="训练时batch大小")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="验证时batch大小")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="Adam初始学习步长")
    parser.add_argument("--num_train_epochs",
                        default=20.0,
                        type=float,
                        help="训练的epochs次数")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for."
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="用不用CUDA")
    parser.add_argument("--local_rank",
                        default=-1,
                        type=int,
                        help="local_rank for distributed training on gpus.")
    parser.add_argument("--seed",
                        default=777,
                        type=int,
                        help="初始化时的随机数种子")
    parser.add_argument("--gradient_accumulation_steps",
                        default=1,
                        type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--optimize_on_cpu",
                        default=False,
                        action='store_true',
                        help="Whether to perform optimization and keep the optimizer averages on CPU.")
    parser.add_argument("--fp16",
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit.")
    parser.add_argument("--loss_scale",
                        default=128,
                        type=float,
                        help="Loss scaling, positive power of 2 values can improve fp16 convergence.")
    parser.add_argument("--cuda",
                        default='4',
                        type=str,
                        help="cuda id or ids")
    parser.add_argument("--threshold",
                        default=0.9,
                        type=float,
                        help="cuda id or ids")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda



    # 对模型输入进行处理的processor，git上可能都是针对英文的processor
    processors = {'mypro': MyPro}

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend='nccl')
        if args.fp16:
            logger.info("16-bits training currently not supported in distributed training")
            args.fp16 = False  # (see https://github.com/pytorch/pytorch/pull/13496)
    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
    #     raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name](args.data_dir)
    label_list = processor.get_labels()

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir + '/train.csv')
        num_train_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    # Prepare model
    model = BertModel.from_pretrained(args.bert_path,
                                      cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(
                                      args.local_rank))

    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    if args.fp16:
        param_optimizer = [(n, param.clone().detach().to('cpu').float().requires_grad_()) \
                           for n, param in model.named_parameters()]
    elif args.optimize_on_cpu:
        param_optimizer = [(n, param.clone().detach().to('cpu').requires_grad_()) \
                           for n, param in model.named_parameters()]
    else:
        param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    t_total = num_train_steps
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()

    # model.load_state_dict(torch.load(args.model_save_pth)['state_dict'])
    global_step = 0
    if args.do_train:
        optimizer = AdamW(optimizer_grouped_parameters,
                             lr=args.learning_rate, correct_bias=False)
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=int(args.warmup_proportion * t_total),
                                         t_total=t_total)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)
        train_features = convert_2examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer, show_exp=False)
        all_input_ids, all_input_mask, all_segment_ids, all_label_ids = [], [], [], []
        for i in range(2):
            all_input_ids.append(torch.tensor([f[i].input_ids for f in train_features], dtype=torch.long))
            all_input_mask.append(torch.tensor([f[i].input_mask for f in train_features], dtype=torch.long))
            all_segment_ids.append(torch.tensor([f[i].segment_ids for f in train_features], dtype=torch.long))
            all_label_ids.append(torch.tensor([f[i].label_id for f in train_features], dtype=torch.long))
        train_data = TensorDataset(all_input_ids[0], all_input_mask[0], all_segment_ids[0], all_label_ids[0],
                             all_input_ids[1], all_input_mask[1], all_segment_ids[1], all_label_ids[1])
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        model.train()
        best_score = 0
        flags = 0
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                input_ids1, input_mask1, segment_ids1, label_ids1, \
                input_ids2, input_mask2, segment_ids2, label_ids2 = batch
                _, pooled_output1 = model(input_ids1, input_mask1, segment_ids1)
                _, pooled_output2 = model(input_ids2, input_mask2, segment_ids2)
                scores = pooled_output1.mm(pooled_output2.t())
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(scores, torch.arange(len(scores), device=device))
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.fp16 and args.loss_scale != 1.0:
                    # rescale loss for fp16 training
                    # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
                    loss = loss * args.loss_scale
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16 or args.optimize_on_cpu:
                        if args.fp16 and args.loss_scale != 1.0:
                            # scale down gradients for fp16 training
                            for param in model.parameters():
                                if param.grad is not None:
                                    param.grad.data = param.grad.data / args.loss_scale
                        is_nan = set_optimizer_params_grad(param_optimizer, model.named_parameters(), test_nan=True)
                        if is_nan:
                            logger.info("FP16 TRAINING: Nan in gradients, reducing loss scaling")
                            args.loss_scale = args.loss_scale / 2
                            model.zero_grad()
                            continue
                        optimizer.step()
                        scheduler.step()
                        copy_optimizer_params_to_model(model.named_parameters(), param_optimizer)
                    else:
                        optimizer.step()
                        scheduler.step()
                    model.zero_grad()

            f1 = val('valid.csv', model, processor, args, tokenizer, device)
            if f1 > best_score:
                best_score = f1
                print('*acc = {}'.format(f1))
                flags = 0
                checkpoint = {
                    'state_dict': model.state_dict()
                }
                torch.save(checkpoint, args.model_save_pth)
            else:
                print('f1 score = {}'.format(f1))
                flags += 1
                if flags >= 6:
                    break

    model.load_state_dict(torch.load(args.model_save_pth)['state_dict'])
    # test('test.csv', model, processor, args, tokenizer, device)
    classify('wait.csv', model, processor, args, tokenizer, device)


if __name__ == '__main__':
    main()
