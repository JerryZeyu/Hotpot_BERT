import json
import random
import numpy as np
from joblib import Parallel, delayed
from pytorch_transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForSequenceClassification, BertTokenizer,
                                  RobertaConfig,
                                  RobertaForSequenceClassification,
                                  RobertaTokenizer,
                                  XLMConfig, XLMForSequenceClassification,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForSequenceClassification,
                                  XLNetTokenizer)
config_class, model_class, tokenizer_class = (BertConfig, BertForSequenceClassification, BertTokenizer)

tokenizer = tokenizer_class.from_pretrained('bert-base-uncased', do_lower_case=True)

# print('paragraphs lenth: ',len(paragraphs))


data = json.load(open('hotpot_train_v1.1.json', 'r'))
##print(len(data)) ##90447
max_question_lenth_list=[]
max_row_lenth_list=[]
max_gold_row_lenth_list=[]
examples = []
i_flag = 0
pos_number = 0
neg_number = 0
for article in data[0:100]:
    question = article['question'].strip().replace("''", '" ').replace("``", '" ')
    answer = article['answer'].strip().replace("''", '" ').replace("``", '" ')
    text_a = question + ' [ANSWER] ' + answer
    max_question_lenth_list.append(len(tokenizer.tokenize(text_a)))
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
                pos_number += 1
                guid = i_flag
                text_b = sent.strip().replace("''", '" ').replace("``", '" ')
                print('gold text b: ',text_b)
                max_gold_row_lenth_list.append(len(tokenizer.tokenize(text_b)))
                label = 1
                examples.append({'guid':guid, 'text_a':text_a,'text_b':text_b, 'label':label})
            else:
                negative_sentences.append(sent)
        #print(len(negative_sentences))
        if len(negative_sentences) < 2:
            continue
        else:
            final_negative_sentences = random.sample(sorted(negative_sentences, key=lambda sent_temp: len(tokenizer.tokenize(sent_temp)))[:10], 2)
        for neg_sent in final_negative_sentences:
            neg_number += 1
            i_flag += 1
            guid = i_flag
            text_b = neg_sent.strip().replace("''", '" ').replace("``", '" ')
            max_row_lenth_list.append(len(tokenizer.tokenize(text_b)))
            print('text b: ', text_b)
            label = 0
            examples.append({'guid': guid, 'text_a': text_a, 'text_b': text_b, 'label': label})
random.shuffle(examples)
print('train examples number: ', len(examples))
print('max question lenth: ', np.max(max_question_lenth_list))
print(sorted(max_question_lenth_list,reverse=True)[0:100])
print('max row lenth: ', np.max(max_row_lenth_list))
print(sorted(max_row_lenth_list, reverse=True)[0:100])
print('max gold row lenth: ', np.max(max_gold_row_lenth_list))
print(sorted(max_gold_row_lenth_list, reverse=True)[0:100])
print('pos number: ',pos_number)
print('neg number: ', neg_number)




