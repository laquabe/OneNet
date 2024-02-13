
import csv
import pickle
from tqdm import tqdm
import jsonlines
import pandas as pd
import json
''' recall entity having mention'''
''' data config'''
dataset_path = '/data/xkliu/EL_datasets/'
num_candidates = 40
'''utils'''
def read_item_counts_dict(filename):
    id_counts_df = pd.read_csv(filename).set_index('page_id')
    item_counts_dict = id_counts_df.counts.to_dict()
    return item_counts_dict

'''data loader'''
alias = pickle.load(open(dataset_path + 'kg_src/alias_table.pickle', 'rb'), encoding='utf-8')
title = pickle.load(open(dataset_path + 'kg_src/title_dict.pickle', 'rb'), encoding='utf-8')
des_with_tokens = pickle.load(open(dataset_path + 'kg_src/descriptions_with_len_dict.pickle', 'rb'), encoding='utf-8')
title2id = pickle.load(open(dataset_path + 'kg_src/title2id_dict.pickle', 'rb'), encoding='utf-8')
# item_counts_dict = read_item_counts_dict(dataset_path + 'kg_src/item_counts_dict.csv')

'''static'''
recall_entity_list_all = set()
def candidate_gen(dataset_name):
    recall_entity_list = set()
    with jsonlines.open(dataset_path + "datasets_id/{}_test.jsonl".format(dataset_name), 'r') as reader:
        for row in tqdm(reader):
            if row['ans_id'] == -1 or row['ans_id'] == -2:
                continue

            candiates = alias.get(row['mention'].lower(), [])
            origin_id = title2id.get(row['mention'], -1)
            if origin_id != -1:
                candiates.append(origin_id)
            recall_entity_list = (set(candiates) | recall_entity_list)
    return recall_entity_list

dataset_list = ['aida']

for dataset in dataset_list:
    recall_entity_list_sole = candidate_gen(dataset)
    recall_entity_list_all = recall_entity_list_all | recall_entity_list_sole

output_f = open(dataset_path + 'kg_src/recall_entity_ace2004.txt', 'w')
for wiki_id in recall_entity_list_all:
    output_f.write(str(wiki_id) + '\n')

with open(dataset_path + 'kg_src/recall_entity_ace2004.txt', 'r') as input_f, \
    open(dataset_path + 'kg_src/recall_entity_ace2004.json', 'w') as output_f:

    for line in tqdm(input_f):
        line = line.strip()
        text = des_with_tokens.get(line, {'text':-1, 'tokens':-1})
        name = title.get(int(line), -1)
        text['name'] = name
        output_f.write(json.dumps({
            'wiki_id':line,
            'src':text
        }, ensure_ascii=False) + '\n')