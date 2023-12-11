import csv
import pickle
from tqdm import tqdm
import jsonlines
import pandas as pd
import json
import random
''' data config'''
dataset_path = '/data/xkliu/EL_datasets/'
num_candidates = 5
'''utils'''
def read_item_counts_dict(filename):
    id_counts_df = pd.read_csv(filename).set_index('page_id')
    item_counts_dict = id_counts_df.counts.to_dict()
    return item_counts_dict

def read_kg_json(filename):
    print('read KG')
    kg_dict = {}
    with open(filename) as input_f:
        for line in tqdm(input_f):
            line = json.loads(line.strip())
            kg_dict[line['wiki_id']] = {'name':line['src']['name'],
                                        'summary':line['src']['summary'],
                                        'text':line['src']['text'],
                                        'wiki_id':line['wiki_id']}

    return kg_dict

'''data loader'''
# alias = pickle.load(open(dataset_path + 'kg_src/alias_table.pickle', 'rb'), encoding='utf-8')
# title = pickle.load(open(dataset_path + 'kg_src/title_dict.pickle', 'rb'), encoding='utf-8')
# des_with_tokens = pickle.load(open(dataset_path + 'kg_src/descriptions_with_len_dict.pickle', 'rb'), encoding='utf-8')
title2id = pickle.load(open(dataset_path + 'kg_src/title2id_dict.pickle', 'rb'), encoding='utf-8')
# item_counts_dict = read_item_counts_dict(dataset_path + 'kg_src/item_counts_dict.csv')
# des_summary = read_kg_json(dataset_path + 'kg_src/recall_entity_summarize_13B_processed.json')

'''static'''
dataset_name = 'aida'
dataset_type = 'train'
def id_add(dataset_name):
    all_num = 0
    has_ans = 0
    hit_num = 0
    mention_count = {}
    mention_tokens = {}
    output_file = open(dataset_path + "datasets_id/{}_{}.jsonl".format(dataset_name, dataset_type), 'w')
    with jsonlines.open(dataset_path + "{}_{}.jsonl".format(dataset_name, dataset_type), 'r') as reader:
        for row in tqdm(reader):
            entity_name = row['output']
            entity_id = title2id.get(entity_name, -1)
            row['ans_id'] = entity_id
            output_file.write(json.dumps(row, ensure_ascii=False) + '\n')

    return all_num, has_ans, hit_num, mention_count, mention_tokens

def kg_static():
    static_dict = {}
    for k, v in tqdm(des_summary.items()):
        token_len = v['tokens']
        static_dict[token_len] = static_dict.get(token_len, 0) + 1
    return static_dict

all_num, has_ans, hit_num, mention_count, mention_tokens = id_add(dataset_name)
