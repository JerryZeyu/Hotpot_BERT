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
import warnings
import numpy as np
import pandas as pd
import random
import pickle
import os
from collections import OrderedDict
import sys
import json
from io import open

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

logger = logging.getLogger(__name__)

def pickle_dump_large_file(obj, filepath):
    max_bytes = 2**31 - 1
    bytes_out = pickle.dumps(obj)
    n_bytes = sys.getsizeof(bytes_out)
    with open(filepath, 'wb') as f_out:
        for idx in range(0, n_bytes, max_bytes):
            f_out.write(bytes_out[idx:idx + max_bytes])


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

    def __init__(self, input_ids, input_mask, segment_ids, label_id, gu_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.gu_id = gu_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        header = []
        uid = None

        df = pd.read_csv(input_file, sep='\t')

        for name in df.columns:
            if name.startswith('[SKIP]'):
                if 'UID' in name and not uid:
                    uid = name
            else:
                header.append(name)
        if not uid or len(df) == 0:
            warnings.warn('Possibly misformatted file: '+input_file)
            return []
        return df.apply(lambda r: (r[uid], ' '.join(str(s) for s in list(r[header]) if not pd.isnull(s))), 1).tolist()


class EprgProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "questions.tsv.train.tsv")))
        return self._create_examples(
            os.path.join(data_dir, "questions.tsv.train.tsv"),  os.path.join(data_dir, "tables"),  "train")

    def get_dev_examples(self, row):
        """See base class."""
        return self._create_dev_examples(row)

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, questions_file, explanation_tables_file, set_type):
        """Creates examples for the training and dev sets."""
        explanations = []
        for path, _, files in os.walk(explanation_tables_file):
            for file in files:
                explanations += self._read_tsv(os.path.join(path, file))
        if not explanations:
            warnings.warn('Empty explanations')
        dict_explanations = {}
        for item in explanations:
            dict_explanations[item[0]] = item[1]
        df_q = pd.read_csv(questions_file, sep='\t')
        df_q['Answer_flag'] = None
        df_q['row_flag'] = None
        df_q['explanation_lenth'] = None
        df_q['Answer_number'] = df_q['question'].map(lambda x: len(x.split('(')) - 1)
        # print(df_q['explanation'])
        df_q['explanation_lenth'] = df_q['explanation'].map(lambda y: len(list(OrderedDict.fromkeys(str(y).split(' ')).keys())))
        examples = []
        i_flag = 0
        count_not_in_tables=[]
        count_not_in_tables_questionid=[]
        max_question_lenth_list=[]
        max_row_lenth_list=[]
        for _, row in df_q.iterrows():
            if 'SUCCESS' not in str(row['flags']).split(' '):
                continue
            # if set_type == 'train':
            #     if row['explanation_lenth'] == 1:
            #         continue
            if row['AnswerKey'] == 'A' or row['AnswerKey'] == "1":
                ac = 0
            elif row['AnswerKey'] == 'B' or row['AnswerKey'] == "2":
                ac = 1
            elif row['AnswerKey'] == 'C' or row['AnswerKey'] == "3":
                ac = 2
            else:
                ac = 3
            #print('ac: ',ac)
            #print(row['QuestionID'])
            #print(row['AnswerKey'])
            #print('question: ',row['question'].split('(')[ac + 1])
            question_ac = row['question'].split('(')[0] +'[ANSWER]'+row['question'].split('(')[ac + 1].split(')')[1]
            question_ac = question_ac.replace("''", '" ').replace("``", '" ')
            #print(question_ac)
            #print('old: ',row['question'].split('(')[0] +row['question'].split('(')[ac + 1])
            text_a = question_ac
            max_question_lenth_list.append(len(text_a))
            explanations_id_list = []
            for single_row_id in list(OrderedDict.fromkeys(str(row['explanation']).split(' ')).keys()):
                explanations_id_list.append(single_row_id.split('|')[0])
            for filer_single in explanations_id_list:
                if filer_single not in dict_explanations.keys():
                    count_not_in_tables.append(filer_single)
                    count_not_in_tables_questionid.append(row['QuestionID'])
                    explanations_id_list.remove(filer_single)
            #assert len(explanations_id_list) == int(row['explanation_lenth'])
            non_explanations_list = []
            for each_row in dict_explanations.keys():
                if each_row not in explanations_id_list:
                    non_explanations_list.append(each_row)
            non_explanations_list = random.sample(non_explanations_list, 100)
            final_rows_list = explanations_id_list+non_explanations_list
            random.shuffle(final_rows_list)
            for each_row_id in final_rows_list:
                if each_row_id in explanations_id_list:
                    i_flag += 1
                    each_row_true = dict_explanations[each_row_id]
                    each_row_true = each_row_true.replace("''", '" ').replace("``", '" ')
                    text_b = each_row_true
                    max_row_lenth_list.append(len(text_b))
                    guid = i_flag
                    label = "1"
                    examples.append(
                        InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
                else:
                    i_flag += 1
                    each_row_false = dict_explanations[each_row_id]
                    each_row_false = each_row_false.replace("''", '" ').replace("``", '" ')
                    text_b = each_row_false
                    max_row_lenth_list.append(len(text_b))
                    guid = i_flag
                    label = "0"
                    examples.append(
                        InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        print('examples length: ', len(examples))
        print('count_not_in_tables_questions: ', len(count_not_in_tables_questionid))
        print('count_not_in_tables_rows: ', len(set(count_not_in_tables)))
        print('max question lenth: ', np.max(max_question_lenth_list))
        print('max row lenth: ', np.max(max_row_lenth_list))
        return examples

    def _create_hotpot_examples(self, questions_file,  set_type):
        """Creates examples for the training and dev sets."""
        data = json.load(open(questions_file, 'r'))
        examples = []
        i_flag = 0
        for article in data:
            question = article['question'].strip().replace("''", '" ').replace("``", '" ')
            answer = article['answer'].strip().replace("''", '" ').replace("``", '" ')
            text_a = question + ' [ANSWER] ' + answer
            #print(text_a)
            paragraphs = article['context']
            sp_set = set(list(map(tuple, article['supporting_facts'])))
            # print(article['_id'])
            # print('support number: ',len(sp_set))
            for para in paragraphs:
                negative_sentences = []
                cur_title, cur_para = para[0], para[1]
                for sent_id, sent in enumerate(cur_para):
                    if (cur_title, sent_id) in sp_set:
                        i_flag += 1
                        guid = i_flag
                        text_b = sent.strip().replace("''", '" ').replace("``", '" ')
                        label = "1"
                        examples.append(
                            InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
                    else:
                        negative_sentences.append(sent)
                #print(len(negative_sentences))
                if len(negative_sentences) < 2:
                    continue
                else:
                    final_negative_sentences = random.sample(negative_sentences, 2)
                for neg_sent in final_negative_sentences:
                    i_flag += 1
                    guid = i_flag
                    text_b = neg_sent.strip().replace("''", '" ').replace("``", '" ')
                    label = "0"
                    examples.append(
                        InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        random.shuffle(examples)
        print('train examples number: ', len(examples))
        return examples


    def _create_dev_examples(self,row):
        explanations = []
        for path, _, files in os.walk(os.path.join('./expl-tablestore-export-2019-09-10-165215','tables')):
            for file in files:
                explanations += self._read_tsv(os.path.join(path, file))
        if not explanations:
            warnings.warn('Empty explanations')
        dict_explanations = {}
        for item in explanations:
            dict_explanations[item[0]] = item[1]
        i_flag = 0
        examples = []
        debug_output_dict = {}
        if row['AnswerKey'] == 'A' or row['AnswerKey'] == "1":
            ac = 0
        elif row['AnswerKey'] == 'B' or row['AnswerKey'] == "2":
            ac = 1
        elif row['AnswerKey'] == 'C' or row['AnswerKey'] == "3":
            ac = 2
        else:
            ac = 3
        question_ac = row['question'].split('(')[0] + '[ANSWER]' + row['question'].split('(')[ac + 1].split(')')[1]
        question_ac = question_ac.replace("''", '" ').replace("``", '" ')
        # print(question_ac)
        # print('old: ',row['question'].split('(')[0] +row['question'].split('(')[ac + 1])
        text_a = question_ac
        explanations_id_list = []
        for single_row_id in list(OrderedDict.fromkeys(str(row['explanation']).split(' ')).keys()):
            explanations_id_list.append(single_row_id.split('|')[0])
        for filer_single in explanations_id_list:
            if filer_single not in dict_explanations.keys():
                explanations_id_list.remove(filer_single)
        # assert len(explanations_id_list) == int(row['explanation_lenth'])
        non_explanations_list = []
        for each_row in dict_explanations.keys():
            if each_row not in explanations_id_list:
                non_explanations_list.append(each_row)
        final_rows_list = explanations_id_list + non_explanations_list
        random.shuffle(final_rows_list)
        for each_row_id in final_rows_list:
            if each_row_id in explanations_id_list:
                i_flag += 1
                each_row_true = dict_explanations[each_row_id]
                each_row_true = each_row_true.replace("''", '" ').replace("``", '" ')
                text_b = each_row_true
                guid = i_flag
                label = "1"
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
                debug_output_dict[str(i_flag)] = {'text_a': text_a, 'text_b': text_b, 'label': label}
            else:
                i_flag += 1
                each_row_false = dict_explanations[each_row_id]
                each_row_false = each_row_false.replace("''", '" ').replace("``", '" ')
                text_b = each_row_false
                guid = i_flag
                label = "0"
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
                debug_output_dict[str(i_flag)] = {'text_a': text_a, 'text_b': text_b, 'label': label}
        return examples, debug_output_dict

def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]',
                                 cls_token_segment_id=1,
                                 sep_token='[SEP]',
                                 sep_token_extra=False,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
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
            # Account for [CLS], [SEP], [SEP] with "- 3". " -4" for RoBERTa.
            special_tokens_count = 4 if sep_token_extra else 3
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count)
        else:
            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            special_tokens_count = 3 if sep_token_extra else 2
            if len(tokens_a) > max_seq_length - special_tokens_count:
                tokens_a = tokens_a[:(max_seq_length - special_tokens_count)]

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
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
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
            #print('label id: ',label_id)
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            #logger.info("guid: %s" % (example.guid))
            logger.info("guid: %d" % (example.guid))
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
                              label_id=label_id,
                              gu_id=example.guid))
    return features


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


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task_name == "eprg":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)


processors = {
    "eprg": EprgProcessor
}

output_modes = {
    "eprg": "classification"
}

GLUE_TASKS_NUM_LABELS = {
    "eprg": 2
}
