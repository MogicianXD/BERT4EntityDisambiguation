import csv
import os
import codecs
import json
import random
import argparse
import pandas as pd
from tqdm import tqdm, trange
from sklearn import metrics, preprocessing
from util import _truncate_seq_pair
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_json(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        dicts = []
        with codecs.open(input_file, 'r', 'utf-8') as infs:
            for inf in infs:
                inf = inf.strip()
                dicts.append(json.loads(inf))
        return dicts


class MyPro(DataProcessor):
    '''自定义数据读取方法，针对json文件

    Returns:
        examples: 数据集，包含index、中文文本、类别三个部分
    '''
    """Processor for the Sentiment Analysis task"""

    def __init__(self, data_dir):
        super(MyPro, self).__init__()
        file_path = os.path.join(data_dir, 'data.csv')
        self.data = pd.read_csv(file_path, encoding='utf-8', index_col='_id')
        self.data.fillna("", inplace=True)

    # 读取训练集
    def get_train_examples(self, file_path):
        train_df = pd.read_csv(file_path, sep='\t', encoding='utf-8', header=None)
        train_data = []
        for index, relation in train_df.iterrows():
            first, second = self.data.loc[relation[0]], self.data.loc[relation[1]]
            guid = relation[0]
            text_a = first['name']
            text_b = first['summary']
            label = relation[1]
            example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)

            guid = relation[1]
            text_a = second['name']
            text_b = second['summary']
            label = relation[0]
            train_data.append((example,
                               InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)))

        return train_data

    # 读取验证集
    def get_dev_examples(self, file_path):
        entity_data, dev_data = [], []
        relations = pd.read_csv(file_path, encoding='utf-8', sep='\t', header=None, index_col=0)
        relations = pd.Series(index=relations.index, data=relations[1])
        label_list = set()
                # self.data[self.data['source'] == 'douban'].iterrows():
        for index, record in \
                self.data[(self.data['source'] == 'douban') & self.data.index.isin(relations.values)].iterrows():
            guid = index
            text_a = record['name']
            text_b = record['summary']
            label = index
            label_list.add(label)
            entity_data.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        for index, entity in relations.items():
            guid = index
            record = self.data.loc[index]
            text_a = record['name']
            text_b = record['summary']
            label = entity
            dev_data.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return entity_data, dev_data, list(label_list)

    # 读取测试集
    def get_test_examples(self, file_path):
        entity_data, test_data = [], []
        items = pd.read_csv(file_path, encoding='utf-8', sep='\t', header=None)
        # self.data[self.data['source'] == 'douban'].iterrows():
        for index, record in \
                self.data[self.data['source'] == 'douban'].iterrows():
            guid = index
            text_a = record['name']
            text_b = record['summary']
            entity_data.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=''))
        for index, mention in items.iterrows():
            guid = mention[0]
            record = self.data.loc[guid]
            text_a = record['name']
            text_b = record['summary']
            test_data.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=''))
        return entity_data, test_data, ['']

    def get_labels(self):
        return self.data.index.tolist()

def _convert_examples_to_features(example, tokenizer, max_seq_length):
    tokens_a = tokenizer.tokenize(example.text_a)

    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0   0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambigiously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_ids, input_mask, segment_ids


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, show_exp=True, map_label=True):
    '''Loads a data file into a list of `InputBatch`s.
    Args:
        examples      : [List] 输入样本，包括question, label, index
        label_list    : [List] 所有可能的类别，可以是int、str等，如['book', 'city', ...]
        max_seq_length: [int] 文本最大长度
        tokenizer     : [Method] 分词方法
    Returns:
        features:
            input_ids  : [ListOf] token的id，在chinese模式中就是每个分词的id，对应一个word vector
            input_mask : [ListOfInt] 真实字符对应1，补全字符对应0
            segment_ids: [ListOfInt] 句子标识符，第一句全为0，第二句全为1
            label_id   : [ListOfInt] 将Label_list转化为相应的id表示
    '''
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    features = []
    for (ex_index, example) in enumerate(examples):
        input_ids, input_mask, segment_ids = _convert_examples_to_features(example, tokenizer, max_seq_length)
        label_id = label_map[example.label]
        # if ex_index < 5 and show_exp:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("tokens: %s" % " ".join(
        #         [str(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     logger.info(
        #         "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #     logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features

def convert_2examples_to_features(examples, label_list, max_seq_length, tokenizer, show_exp=True):
    '''Loads a data file into a list of `InputBatch`s.
    Args:
        examples      : [List] 输入样本，包括question, label, index
        label_list    : [List] 所有可能的类别，可以是int、str等，如['book', 'city', ...]
        max_seq_length: [int] 文本最大长度
        tokenizer     : [Method] 分词方法
    Returns:
        features:
            input_ids  : [ListOf] token的id，在chinese模式中就是每个分词的id，对应一个word vector
            input_mask : [ListOfInt] 真实字符对应1，补全字符对应0
            segment_ids: [ListOfInt] 句子标识符，第一句全为0，第二句全为1
            label_id   : [ListOfInt] 将Label_list转化为相应的id表示
    '''
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    features = []
    for (ex_index, example_pair) in enumerate(examples):
        example1, example2 = example_pair
        input_ids1, input_mask1, segment_ids1 = _convert_examples_to_features(example1, tokenizer, max_seq_length)
        input_ids2, input_mask2, segment_ids2 = _convert_examples_to_features(example2, tokenizer, max_seq_length)
        label_id1 = label_map[example1.label]
        label_id2 = label_map[example2.label]
        # if ex_index < 5 and show_exp:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("tokens: %s" % " ".join(
        #         [str(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     logger.info(
        #         "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #     logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append((
            InputFeatures(input_ids=input_ids1,
                          input_mask=input_mask1,
                          segment_ids=segment_ids1,
                          label_id=label_id1),
            InputFeatures(input_ids=input_ids2,
                          input_mask=input_mask2,
                          segment_ids=segment_ids2,
                          label_id=label_id2)),
        )
    return features


def get_dataloader(examples, args, batch_size, label_list, tokenizer):
    features = convert_examples_to_features(
        examples, label_list, args.max_seq_length, tokenizer, show_exp=False, map_label=False)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    sampler = SequentialSampler(data)
    return DataLoader(data, sampler=sampler, batch_size=batch_size)

