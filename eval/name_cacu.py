import csv
import pickle
from tqdm import tqdm
import jsonlines
import pandas as pd
import json
# 统计数据集中，是否存在同一段落，menion完全一样但是所指实体不一样的
''' data config'''
dataset_path = '/data/xkliu/EL_datasets/'
num_candidates = 40
'''utils'''
def read_item_counts_dict(filename):
    id_counts_df = pd.read_csv(filename).set_index('page_id')
    item_counts_dict = id_counts_df.counts.to_dict()
    return item_counts_dict

def name_count(filename):
    # left_context,mention, right_context, output
    # to see mention -> output is 1to1?
    mentions_dict = {}
    diff_num = 0
    repeat_dict = {}
    with open(filename, 'r') as input_f:
        # 存在context覆盖的问题
        for line in tqdm(input_f):
            line = json.loads(line.strip())
            context = line['left_context'].split(',')[-1] + ' ' + line['mention'] + ' ' + line['right_context'].split(',')[0]
            context = context.strip()
            context = ' '.join(context.split())
            # print(context, line['mention'])
            context_dict = mentions_dict.get(line['mention'], {})
            in_flag = False
            for c, entity in context_dict.items():
                if c in context:
                    in_flag = True
                    if line['output'] != entity:
                        repeat_dict_m = repeat_dict.get(line['mention'],{})
                        repeat_dict_m['context'] = context
                        repeat_dict_list = repeat_dict_m.get('output', set())
                        repeat_dict_list.add(line['output'])
                        repeat_dict_list.add(entity)
                        repeat_dict_m['output'] = repeat_dict_list
                        repeat_dict[line['mention']] = repeat_dict_m
                        diff_num += 1
                elif context in c:
                    in_flag = True
                    if line['output'] != entity:
                        repeat_dict_m = repeat_dict.get(line['mention'],{})
                        repeat_dict_m['context'] = c
                        repeat_dict_list = repeat_dict_m.get('output', set())
                        repeat_dict_list.add(line['output'])
                        repeat_dict_list.add(entity)
                        repeat_dict_m['output'] = repeat_dict_list
                        repeat_dict[line['mention']] = repeat_dict_m
                        diff_num += 1
            if not in_flag:
                context_dict[context] = line['output']
                mentions_dict[line['mention']] = context_dict
    # for k,v in repeat_dict.items():
    #     print(k, v)
    # print(len(mentions_dict))
    return diff_num
'''data loader'''
# alias = pickle.load(open(dataset_path + 'kg_src/alias_table.pickle', 'rb'), encoding='utf-8')
# title = pickle.load(open(dataset_path + 'kg_src/title_dict.pickle', 'rb'), encoding='utf-8')
# des_with_tokens = pickle.load(open(dataset_path + 'kg_src/descriptions_with_len_dict.pickle', 'rb'), encoding='utf-8')
# title2id = pickle.load(open(dataset_path + 'kg_src/title2id_dict.pickle', 'rb'), encoding='utf-8')
# item_counts_dict = read_item_counts_dict(dataset_path + 'kg_src/item_counts_dict.csv')

'''static'''
datasets = ['ace2004', 'aida', 'aquaint', 'cweb', 'msnbc', 'wiki']
# datasets = ['aida']
diff_nums = []

for dataset in datasets:
    diff_num = name_count(dataset_path + '{}_test.jsonl'.format(dataset))
    diff_nums.append(diff_num)

for dataset, diff_num in zip(datasets, diff_nums):
    print('{}: diff num {}'.format(dataset, diff_num))