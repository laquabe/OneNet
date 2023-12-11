import csv
import pickle
from tqdm import tqdm
import jsonlines
import pandas as pd
import json
import random
''' data config'''
dataset_path = '/data/xkliu/EL_datasets/'
num_candidates = 10
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
alias = pickle.load(open(dataset_path + 'kg_src/alias_table.pickle', 'rb'), encoding='utf-8')
# title = pickle.load(open(dataset_path + 'kg_src/title_dict.pickle', 'rb'), encoding='utf-8')
# des_with_tokens = pickle.load(open(dataset_path + 'kg_src/descriptions_with_len_dict.pickle', 'rb'), encoding='utf-8')
# title2id = pickle.load(open(dataset_path + 'kg_src/title2id_dict.pickle', 'rb'), encoding='utf-8')
item_counts_dict = read_item_counts_dict(dataset_path + 'kg_src/item_counts_dict.csv')
des_summary = read_kg_json(dataset_path + 'kg_src/recall_entity_summarize_13B_processed.json')

'''static'''
dataset_name = 'wiki'
def candidate_gen(dataset_name):
    all_num = 0
    has_ans = 0
    hit_num = 0
    mention_count = {}
    mention_tokens = {}
    output_file = open(dataset_path + "datasets_recall/{}_test_13B_top10g.jsonl".format(dataset_name), 'w')
    with jsonlines.open(dataset_path + "datasets_id/{}_test.jsonl".format(dataset_name), 'r') as reader:
        for row in tqdm(reader):
            hit_flag = False
            print_flag = False
            all_num += 1
            if row['ans_id'] == -1 or row['ans_id'] == -2:
                cand_src = []
                row['candidates'] = cand_src
                output_file.write(json.dumps(row, ensure_ascii=False) + '\n')
                continue

            has_ans += 1

            candiates = alias.get(row['mention'].lower(), [])
            # origin_id = title2id.get(row['mention'], -1)
            # if origin_id != -1:
            #     candiates.append(origin_id)

            if row['ans_id'] in candiates:
                hit_num += 1
                hit_flag = True
            else:
                print(row)
            
            if len(candiates) > num_candidates:
                heat_dict = {}
                for cand in candiates:
                    heat_dict[cand] = item_counts_dict.get(cand, 0)
                heat_dict = sorted(heat_dict.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)
                new_candiates = [k for k, v in heat_dict[:num_candidates]]
                if hit_flag:
                    if row['ans_id'] not in new_candiates:
                        # print_flag = True
                        new_candiates.pop(-1)
                        new_candiates.append(row['ans_id'])
                        random.shuffle(new_candiates)

                candiates = [i for i in new_candiates]

            mention_count[len(candiates)] = mention_count.get(len(candiates), 0) + 1
            cand_tokens = 0
            cand_src = []

            for cand in candiates:
                # print(des_summary.get(str(cand), {'summary':'','sum_tokens':0}))
                # cand_tokens += des_summary.get(str(cand), {'summary':'','sum_tokens':0})['sum_tokens']
                try:
                    cand_src.append(des_summary[str(cand)])
                except:
                    continue
            mention_tokens[cand_tokens] = mention_tokens.get(cand_tokens, 0) + 1

            row['candidates'] = cand_src
            output_file.write(json.dumps(row, ensure_ascii=False) + '\n')

    return all_num, has_ans, hit_num, mention_count, mention_tokens

def kg_static():
    static_dict = {}
    for k, v in tqdm(des_summary.items()):
        token_len = v['tokens']
        static_dict[token_len] = static_dict.get(token_len, 0) + 1
    return static_dict

all_num, has_ans, hit_num, mention_count, mention_tokens = candidate_gen(dataset_name)