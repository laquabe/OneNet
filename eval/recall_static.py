import json
from tqdm import tqdm

datasets_path = '/data/xkliu/EL_datasets/datasets_recall/cand_filter/'
dateset_name = 'wiki'
recall_file = '{}_test_prompt4_sum13B_13B.jsonl'.format(dateset_name)

all_num = 0
miss_num = 0
recall_num = 0
error_num = 0
negative_list = ['"no"', "'no'", 'not related', 'cannot establish a relationship']
with open(datasets_path + recall_file, 'r') as input_f:
    recall_dict = {}
    for line in input_f:
        line = json.loads(line.strip())
        if line['id'] not in recall_dict.keys():
            all_num += 1
            recall_dict[line['id']] = {
                'left_context':line['left_context'],
                'mention':line['mention'],
                'right_context':line['right_context'],
                'output':line['output'],
                'ans_id':line['ans_id']
                }
        if len(line['llm_answer']) == 0:
            error_num += 1
        hit_flag = True
        for neg_words in negative_list:
            if neg_words in line['llm_answer']:
                hit_flag = False
                if line['ans_id'] == int(line['cand_wiki_id']):
                    miss_num += 1
                break
        if hit_flag == True:
            # print(hit_flag, line['llm_answer'])
            recall_num += 1

print(all_num, all_num - miss_num, (all_num - miss_num) / all_num, recall_num, error_num)