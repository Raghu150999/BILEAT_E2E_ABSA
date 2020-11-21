# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" BERT classification fine-tuning: utilities to work with GLUE tasks """

from __future__ import absolute_import, division, print_function

import csv
import logging
import os
import sys
from io import open

from seq_utils import *

logger = logging.getLogger(__name__)

SMALL_POSITIVE_CONST = 1e-4

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
    
    def copy(self):
        return InputExample(self.guid, self.text_a, self.text_b, self.label)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class SeqInputFeatures(object):
    """A single set of features of data for the ABSA task"""
    def __init__(self, input_ids, input_mask, segment_ids, label_ids, label_ids_1, label_ids_o, stm_lm_labels, evaluate_label_ids, label_sent):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.label_ids_1 = label_ids_1
        self.label_ids_o = label_ids_o
        self.stm_lm_labels = stm_lm_labels
        self.label_sent = label_sent
        # mapping between word index and head token index
        self.evaluate_label_ids = evaluate_label_ids

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
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(cell for cell in line)
                lines.append(line)
            return lines


class ABSAProcessor(DataProcessor):
    """Processor for the ABSA datasets"""
    def get_train_examples(self, data_dir, tagging_schema):
        return self._create_examples(data_dir=data_dir, set_type='train', tagging_schema=tagging_schema)

    def get_dev_examples(self, data_dir, tagging_schema):
        return self._create_examples(data_dir=data_dir, set_type='dev', tagging_schema=tagging_schema)

    def get_test_examples(self, data_dir, tagging_schema):
        return self._create_examples(data_dir=data_dir, set_type='test', tagging_schema=tagging_schema)

    def get_labels(self, tagging_schema):
        if tagging_schema == 'OT':
            return []
        elif tagging_schema == 'BIO':
            return ['O', 'B-POS', 'I-POS', 'B-NEG', 'I-NEG', 'B-NEU', 'I-NEU']
        elif tagging_schema == 'BIEOS':
            return ['O', 'B-POS', 'I-POS', 'E-POS', 'S-POS',
            'B-NEG', 'I-NEG', 'E-NEG', 'S-NEG',
            'B-NEU', 'I-NEU', 'E-NEU', 'S-NEU']
        else:
            raise Exception("Invalid tagging schema %s..." % tagging_schema)
        
    def get_normal_labels(self, tagging_schema):
        if tagging_schema == 'OT':
            return []
        elif tagging_schema == 'BIO':
            return ['O', 'B', 'I']
        elif tagging_schema == 'BIEOS':
            return ['O', 'B', 'I', 'E', 'S']
        else:
            raise Exception("Invalid tagging schema %s..." % tagging_schema)
    
    @staticmethod
    def get_sentiment_labels():
        return ['O', 'POS', 'NEG', 'NEU']

    def _create_examples(self, data_dir, set_type, tagging_schema):
        examples = []
        file = os.path.join(data_dir, "%s.txt" % set_type)
        class_count = np.zeros(3)
        with open(file, 'r', encoding='UTF-8') as fp:
            sample_id = 0
            for line in fp:
                sent_string, tag_string = line.strip().split('####')
                words = []
                tags = []
                for tag_item in tag_string.split(' '):
                    eles = tag_item.split('=')
                    if len(eles) == 1:
                        raise Exception("Invalid samples %s..." % tag_string)
                    elif len(eles) == 2:
                        word, tag = eles
                    else:
                        word = ''.join((len(eles) - 2) * ['='])
                        tag = eles[-1]
                    words.append(word)
                    tags.append(tag)
                # convert from ot to bieos
                if tagging_schema == 'BIEOS':
                    tags = ot2bieos_ts(tags)
                elif tagging_schema == 'BIO':
                    tags = ot2bio_ts(tags)
                else:
                    # original tags follow the OT tagging schema, do nothing
                    pass
                guid = "%s-%s" % (set_type, sample_id)
                text_a = ' '.join(words)
                #label = [absa_label_vocab[tag] for tag in tags]
                gold_ts = tag2ts(ts_tag_sequence=tags)
                for (b, e, s) in gold_ts:
                    if s == 'POS':
                        class_count[0] += 1
                    if s == 'NEG':
                        class_count[1] += 1
                    if s == 'NEU':
                        class_count[2] += 1
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=tags))
                sample_id += 1
        print("%s class count: %s" % (set_type, class_count))
        return examples


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def read_lexicon():
    """
    read sentiment lexicon from the disk
    :return:
    """
    path = 'mpqa_full.txt'
    sent_lexicon = {}
    with open(path) as fp:
        for line in fp:
            word, polarity = line.strip().split('\t')
            if word not in sent_lexicon:
                sent_lexicon[word] = polarity
    return sent_lexicon

def convert_examples_to_seq_features(examples, label_list, tokenizer,
                                     cls_token_at_end=False, pad_on_left=False, cls_token='[CLS]',
                                     sep_token='[SEP]', pad_token=0, sequence_a_segment_id=0,
                                     sequence_b_segment_id=1, cls_token_segment_id=1, pad_token_segment_id=0,
                                     mask_padding_with_zero=True, stm_win=3):
    # feature extraction for sequence labeling
    label_map = {label: i for i, label in enumerate(label_list[0])}
    label_map_1 = {label: i for i, label in enumerate(label_list[1])}
    sentiment_map = {label: i for i, label in enumerate(ABSAProcessor.get_sentiment_labels())}
    features = []
    max_seq_length = -1
    examples_tokenized = []
    stm_lex = read_lexicon()
    imp_words = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = []
        labels_a = []
        labels_a_1 = []
        labels_o = []
        stm_lm_labels = []
        evaluate_label_ids = []
        words = example.text_a.split(' ')
        wid, tid = 0, 0
        op_tags = []
        op_labels = []
        for j in range(len(words)):
            # left boundary of sentimental context
            stm_ctx_lb = j - stm_win
            if stm_ctx_lb < 0:
                stm_ctx_lb = 0
            stm_ctx_rb = j + stm_win + 1
            left_ctx = words[stm_ctx_lb:j]
            right_ctx = words[j+1:stm_ctx_rb]
            stm_ctx = left_ctx + right_ctx
            flag = 0
            for w in stm_ctx:
                if w in stm_lex:
                    flag = 1
                    break
            if words[j] in stm_lex:
                if j > 0 and op_labels[-1] != 'O':
                    op_labels.append('I')
                else:
                    op_labels.append('B')
            else:
                op_labels.append('O')
            op_tags.append(flag)
        positions = []
        for word, label, op_tag, op_label in zip(words, example.label, op_tags, op_labels):
            subwords = tokenizer.tokenize(word)
            tokens_a.extend(subwords)
            stm_lm_labels.extend([op_tag] * len(subwords))
            if label[0] == 'B':
                tmp = 'I' + label[1:]
                labels_a.extend([label] + [tmp] * (len(subwords) - 1))
                labels_a_1.extend([label[0]] + [tmp[0]] * (len(subwords) - 1))
            else:
                labels_a.extend([label] * (len(subwords)))
                labels_a_1.extend([label[0]] * (len(subwords)))
            if len(subwords) == 1 and (label[0] != 'O' or op_label != 'O'):
                positions.append((tid + 1, wid))
            if op_label == 'B':
                labels_o.extend([op_label] + ['I'] * (len(subwords) - 1))
            else:
                labels_o.extend([op_label] * len(subwords))
            evaluate_label_ids.append(tid)
            wid += 1
            # move the token pointer
            tid += len(subwords)
        imp_words.append(positions)
        assert tid == len(tokens_a)
        evaluate_label_ids = np.array(evaluate_label_ids, dtype=np.int32)
        examples_tokenized.append((tokens_a, labels_a, labels_a_1, stm_lm_labels, labels_o, evaluate_label_ids))
        if len(tokens_a) > max_seq_length:
            max_seq_length = len(tokens_a)
    # count on the [CLS] and [SEP]
    max_seq_length += 2
    # max_seq_length = 128
    for ex_index, (tokens_a, labels_a, labels_a_1, stm_lm_labels, labels_o, evaluate_label_ids) in enumerate(examples_tokenized):
        # Add sep token
        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)
        labels = labels_a + ['O']
        labels_1 = labels_a_1 + ['O']
        stm_lm_labels = stm_lm_labels + [0]
        labels_o = labels_o + ['O']
        if cls_token_at_end:
            # evaluate label ids not change
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
            labels = labels + ['O']
            labels_1 = labels_1 + ['O']
            labels_o = labels_o + ['O']
            stm_lm_labels = stm_lm_labels + [0]
        else:
            # right shift 1 for evaluate label ids
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids
            labels = ['O'] + labels
            labels_1 = ['O'] + labels_1
            labels_o = ['O'] + labels_o
            stm_lm_labels = [0] + stm_lm_labels
            evaluate_label_ids += 1
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        #print("Current labels:", labels)
        label_ids = [label_map[label] for label in labels]
        label_ids_1 = [label_map_1[label] for label in labels_1]
        label_ids_o = [label_map_1[label] for label in labels_o]
        label_sent = [sentiment_map[label[-3:]] for label in labels]

        # pad the input sequence and the mask sequence
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            # pad sequence tag 'O'
            label_ids = ([0] * padding_length) + label_ids
            label_ids_1 = ([0] * padding_length) + label_ids_1
            label_ids_o = ([0] * padding_length) + label_ids_o
            stm_lm_labels = ([0] * padding_length) + stm_lm_labels
            label_sent = ([0] * padding_length) + label_sent
            # right shift padding_length for evaluate_label_ids
            evaluate_label_ids += padding_length
        else:
            # evaluate ids not change
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)
            # pad sequence tag 'O'
            label_ids = label_ids + ([0] * padding_length)
            label_ids_1 = label_ids_1 + ([0] * padding_length)
            label_ids_o = label_ids_o + ([0] * padding_length)
            stm_lm_labels = stm_lm_labels + ([0] * padding_length)
            label_sent = label_sent + ([0] * padding_length)
        
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(label_ids_1) == max_seq_length
        assert len(label_ids_o) == max_seq_length
        assert len(stm_lm_labels) == max_seq_length
        assert len(label_sent) == max_seq_length

        # if ex_index < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("tokens: %s" % " ".join(
        #             [str(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #     logger.info("labels: %s " % ' '.join([str(x) for x in label_ids]))
        #     logger.info("evaluate label ids: %s" % evaluate_label_ids)
        #     print()
        features.append(
            SeqInputFeatures(input_ids=input_ids,
                             input_mask=input_mask,
                             segment_ids=segment_ids,
                             label_ids=label_ids,
                             label_ids_1=label_ids_1,
                             label_ids_o=label_ids_o,
                             stm_lm_labels=stm_lm_labels,
                             evaluate_label_ids=evaluate_label_ids,
                             label_sent=label_sent))
    # print("maximal sequence length is", max_seq_length)
    # print(imp_words[0])
    return features, imp_words


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False, pad_on_left=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=1, pad_token_segment_id=0,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features


def match_ts(gold_ts_sequence, pred_ts_sequence):
    """
    calculate the number of correctly predicted targeted sentiment
    :param gold_ts_sequence: gold standard targeted sentiment sequence
    :param pred_ts_sequence: predicted targeted sentiment sequence
    :return:
    """
    # positive, negative and neutral
    tag2tagid = {'POS': 0, 'NEG': 1, 'NEU': 2}
    hit_count, gold_count, pred_count = np.zeros(3), np.zeros(3), np.zeros(3)
    for t in gold_ts_sequence:
        #print(t)
        ts_tag = t[2]
        tid = tag2tagid[ts_tag]
        gold_count[tid] += 1
    for t in pred_ts_sequence:
        ts_tag = t[2]
        tid = tag2tagid[ts_tag]
        if t in gold_ts_sequence:
            hit_count[tid] += 1
        pred_count[tid] += 1
    return hit_count, gold_count, pred_count


def compute_metrics_absa(preds, labels, all_evaluate_label_ids, tagging_schema):
    if tagging_schema == 'BIEOS':
        absa_label_vocab = {'O': 0, 'B-POS': 1, 'I-POS': 2, 'E-POS': 3, 'S-POS': 4,
                        'B-NEG': 5, 'I-NEG': 6, 'E-NEG': 7, 'S-NEG': 8,
                        'B-NEU': 9, 'I-NEU': 10, 'E-NEU': 11, 'S-NEU': 12}
    elif tagging_schema == 'BIO':
        absa_label_vocab = {'O': 0, 'B-POS': 1, 'I-POS': 2,
        'B-NEG': 3, 'I-NEG': 4, 'B-NEU': 5, 'I-NEU': 6}
    elif tagging_schema == 'OT':
        absa_label_vocab = {'O': 0, 'T-POS': 1, 'T-NEG': 2, 'T-NEU': 3}
    else:
        raise Exception("Invalid tagging schema %s..." % tagging_schema)
    absa_id2tag = {}
    for k in absa_label_vocab:
        v = absa_label_vocab[k]
        absa_id2tag[v] = k
    # number of true postive, gold standard, predicted targeted sentiment
    n_tp_ts, n_gold_ts, n_pred_ts = np.zeros(3), np.zeros(3), np.zeros(3)
    # precision, recall and f1 for aspect-based sentiment analysis
    ts_precision, ts_recall, ts_f1 = np.zeros(3), np.zeros(3), np.zeros(3)
    n_samples = len(all_evaluate_label_ids)
    pred_y, gold_y = [], []
    class_count = np.zeros(3)
    tagging = []
    for i in range(n_samples):
        evaluate_label_ids = all_evaluate_label_ids[i]

        pred_labels = preds[i][evaluate_label_ids]
        gold_labels = labels[i][evaluate_label_ids]
        assert len(pred_labels) == len(gold_labels)
        # here, no EQ tag will be induced
        pred_tags = [absa_id2tag[label] for label in pred_labels]
        gold_tags = [absa_id2tag[label] for label in gold_labels]

        if tagging_schema == 'OT':
            gold_tags = ot2bieos_ts(gold_tags)
            pred_tags = ot2bieos_ts(pred_tags)
        elif tagging_schema == 'BIO':
            gold_tags = ot2bieos_ts(bio2ot_ts(gold_tags))
            pred_tags = ot2bieos_ts(bio2ot_ts(pred_tags))
        else:
            # current tagging schema is BIEOS, do nothing
            pass
        g_ts_sequence, p_ts_sequence = tag2ts(ts_tag_sequence=gold_tags), tag2ts(ts_tag_sequence=pred_tags)

        hit_ts_count, gold_ts_count, pred_ts_count = match_ts(gold_ts_sequence=g_ts_sequence,
                                                              pred_ts_sequence=p_ts_sequence)
        n_tp_ts += hit_ts_count
        n_gold_ts += gold_ts_count
        n_pred_ts += pred_ts_count
        for (b, e, s) in g_ts_sequence:
            if s == 'POS':
                class_count[0] += 1
            if s == 'NEG':
                class_count[1] += 1
            if s == 'NEU':
                class_count[2] += 1
        tagging.append((g_ts_sequence, p_ts_sequence))
    for i in range(3):
        n_ts = n_tp_ts[i]
        n_g_ts = n_gold_ts[i]
        n_p_ts = n_pred_ts[i]
        ts_precision[i] = float(n_ts) / float(n_p_ts + SMALL_POSITIVE_CONST)
        ts_recall[i] = float(n_ts) / float(n_g_ts + SMALL_POSITIVE_CONST)
        ts_f1[i] = 2 * ts_precision[i] * ts_recall[i] / (ts_precision[i] + ts_recall[i] + SMALL_POSITIVE_CONST)

    macro_f1 = ts_f1.mean()

    # calculate micro-average scores for ts task
    # TP
    n_tp_total = sum(n_tp_ts)
    # TP + FN
    n_g_total = sum(n_gold_ts)
    print("class_count:", class_count)

    # TP + FP
    n_p_total = sum(n_pred_ts)
    micro_p = float(n_tp_total) / (n_p_total + SMALL_POSITIVE_CONST)
    micro_r = float(n_tp_total) / (n_g_total + SMALL_POSITIVE_CONST)
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r + SMALL_POSITIVE_CONST)
    scores = {'macro-f1': macro_f1, 'precision': micro_p, "recall": micro_r, "micro-f1": micro_f1}
    return scores, tagging


processors = {
    "laptop14": ABSAProcessor,
    "rest_total": ABSAProcessor,
    "rest_total_revised": ABSAProcessor,
    "rest14": ABSAProcessor,
    "rest15": ABSAProcessor,
    "rest16": ABSAProcessor,
    "rest_total_adv": ABSAProcessor,
    "laptop14_adv": ABSAProcessor
}

output_modes = {
    "cola": "classification",
    "mnli": "classification",
    "mnli-mm": "classification",
    "mrpc": "classification",
    "sst-2": "classification",
    "sts-b": "regression",
    "qqp": "classification",
    "qnli": "classification",
    "rte": "classification",
    "wnli": "classification",
    "laptop14": "classification",
    "rest_total": "classification",
    "rest14": "classification",
    "rest15": "classification",
    "rest16": "classification",
    "rest_total_revised": "classification",
    "rest_total_adv": "classification",
    "laptop14_adv": "classification"
}
