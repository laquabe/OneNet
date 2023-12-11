import csv
import pickle
from tqdm import tqdm
import jsonlines
import pandas as pd
import json
''' data config'''
dataset_path = '/data/xkliu/EL_datasets/'
num_candidates = 20
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
            kg_dict[line['wiki_id']] = {'summary':line['src']['summary'],
                                        'sum_tokens':line['sum_tokens']}

    return kg_dict

'''data loader'''
alias = pickle.load(open(dataset_path + 'kg_src/alias_table.pickle', 'rb'), encoding='utf-8')
# title = pickle.load(open(dataset_path + 'kg_src/title_dict.pickle', 'rb'), encoding='utf-8')
des_with_tokens = pickle.load(open(dataset_path + 'kg_src/descriptions_with_len_dict.pickle', 'rb'), encoding='utf-8')
title2id = pickle.load(open(dataset_path + 'kg_src/title2id_dict.pickle', 'rb'), encoding='utf-8')
item_counts_dict = read_item_counts_dict(dataset_path + 'kg_src/item_counts_dict.csv')
# des_summary = read_kg_json(dataset_path + 'kg_src/recall_entity_summarize_7B_with_tokens.json')

'''static'''
dataset_name = 'wiki'
def candidate_gen(dataset_name):
    all_num = 0
    has_ans = 0
    hit_num = 0
    mention_count = {}
    mention_tokens = {}
    with jsonlines.open(dataset_path + "datasets_id/{}_test.jsonl".format(dataset_name), 'r') as reader:
        for row in tqdm(reader):
            all_num += 1
            if row['ans_id'] == -1 or row['ans_id'] == -2:
                continue
            has_ans += 1

            candiates = alias.get(row['mention'].lower(), [])
            # origin_id = title2id.get(row['mention'], -1)
            # if origin_id != -1:
            #     candiates.append(origin_id)

            if len(candiates) > num_candidates:
                heat_dict = {}
                for cand in candiates:
                    heat_dict[cand] = item_counts_dict.get(cand, 0)
                heat_dict = sorted(heat_dict.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)
                candiates = [k for k, v in heat_dict[:num_candidates]]

            if row['ans_id'] in candiates:
                hit_num += 1
            else:
                print(row)
            
            mention_count[len(candiates)] = mention_count.get(len(candiates), 0) + 1
            cand_tokens = 0
            for cand in candiates:
                # print(des_summary.get(str(cand), {'summary':'','sum_tokens':0}))
                # cand_tokens += des_with_tokens.get(str(cand), {'summary':'','sum_tokens':0})['sum_tokens']
                cand_tokens += 1
            mention_tokens[cand_tokens] = mention_tokens.get(cand_tokens, 0) + 1

    return all_num, has_ans, hit_num, mention_count, mention_tokens

def kg_static():
    static_dict = {}
    for k, v in tqdm(des_summary.items()):
        token_len = v['tokens']
        static_dict[token_len] = static_dict.get(token_len, 0) + 1
    return static_dict

def bag_freq(len_dict:dict, tag):
    # Token: 0-256, 256-512, 512-1024, 1048-2048, 2048-4096, >4096
    # counts: 1, 2-3, 4-6, 6-10, 10-50, 50-100, >100
    bag_dict = {}
    if tag == 'tokens':
        bag_dict = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}
        for tokens, count in len_dict.items():
            if tokens == 0:
                bag_dict[0] = bag_dict.get(6, 0) + count
            if tokens <= 256:
                bag_dict[0] = bag_dict.get(0, 0) + count
            elif tokens <= 512:
                bag_dict[1] = bag_dict.get(1, 0) + count
            elif tokens <= 1024:
                bag_dict[2] = bag_dict.get(2, 0) + count
            elif tokens <= 2048:
                bag_dict[3] = bag_dict.get(3, 0) + count
            elif tokens <= 4096:
                bag_dict[4] = bag_dict.get(4, 0) + count
            else:
                bag_dict[5] = bag_dict.get(5, 0) + count
    elif tag == 'counts':
        for tokens, count in len_dict.items():
            if tokens <= 1:
                bag_dict[0] = bag_dict.get(0, 0) + count
            elif tokens <= 3:
                bag_dict[1] = bag_dict.get(1, 0) + count
            elif tokens <= 6:
                bag_dict[2] = bag_dict.get(2, 0) + count
            elif tokens <= 10:
                bag_dict[3] = bag_dict.get(3, 0) + count
            elif tokens <= 50:
                bag_dict[4] = bag_dict.get(4, 0) + count
            elif tokens <= 100:
                bag_dict[5] = bag_dict.get(5, 0) + count
            else:
                bag_dict[6] = bag_dict.get(6, 0) + count
    
        if bag_dict.get(6, 0) == 0:
            bag_dict[6] = 0
    return bag_dict

def draw_pie(data_dict:dict, save_path, title, labels):
    data_sorted = dict(sorted(data_dict.items(), key = lambda kv:(kv[0], kv[1])))
    import matplotlib.pyplot as plt
    plt.clf()
    plt.pie(data_sorted.values(), labels=labels, autopct='%1.1f%%')
    plt.title(title)
    plt.savefig(save_path)

all_num ,has_ans, hit_num, mention_count, mention_tokens =candidate_gen(dataset_name)
# mention_token_bag = bag_freq(mention_tokens, 'tokens')
# mention_count_bag = bag_freq(mention_count, 'counts')
# print(mention_count_bag)
# print(mention_token_bag)
print(has_ans, all_num, has_ans / all_num)
print(hit_num, all_num, hit_num / all_num)
# draw_pie(mention_count_bag, dataset_path + 'pic/{}_sum_counts.png'.format(dataset_name), 
#          '{}_recall_entities'.format(dataset_name), ['1', '2-3', '4-6', '6-10', '10-50', '50-100', '>100'])
# draw_pie(mention_token_bag, dataset_path + 'pic/{}_sum_tokens.png'.format(dataset_name),
#           '{}_recall_tokens'.format(dataset_name), ['1-256', '256-512', '512-1024', '1024-2048', '2048-4096', '>4096', '0'])

# kg_count = kg_static()
# kg_count_bag = bag_freq(kg_count, 'tokens')
# print(kg_count_bag)
# draw_pie(kg_count_bag, dataset_path + 'pic/kg_tokens.png',
#           'kg_tokens', ['0-256', '256-512', '512-1024', '1048-2048', '2048-4096', '>4096'])