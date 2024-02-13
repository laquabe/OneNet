import json
from tqdm import tqdm

dataset_name = 'msnbc'
filter_method = 'prompt0_sum13B_13B'
negative_list = ['"no"', "'no'", 'not related','cannot establish a relationship']
dataset_path = '/data/xkliu/EL_datasets/datasets_recall/'
with open(dataset_path + 'cand_filter/{}_test_{}.jsonl'.format(dataset_name, filter_method)) as input_f, \
    open(dataset_path + 'listwise_input/no_cand/{}_test_13B.jsonl'.format(dataset_name)) as no_cand_f, \
    open(dataset_path + 'listwise_input/meta/{}_test_{}.jsonl'.format(dataset_name, filter_method), 'w') as output_f:
    
    id_cand_dict = {}
    id_set = set()
    for line in tqdm(input_f):
        line = json.loads(line.strip())
        if line['id'] not in id_cand_dict.keys():
            id_set.add(line['id'])
            id_cand_dict[line['id']] = {
                'left_context':line['left_context'],
                "mention":line["mention"],
                "right_context":line["right_context"],
                "output":line["output"],
                "ans_id":line["ans_id"],
                "id":line["id"],
                "candidates":[],
                "meta_cands":[]
            }
        neg_flag = False
        meta_cands = id_cand_dict[line['id']]['meta_cands']
        meta_cands.append(line['cand_name'])
        for neg_words in negative_list:
            if neg_words in line['llm_answer']:
                neg_flag = True
        if neg_flag:
            continue
        cand_list = id_cand_dict[line['id']]['candidates']
        cand_list.append({
            'name':line["cand_name"],
            'text':line["cand_text"],
            'summary':line["cand_summary"],
            'wiki_id':line["cand_wiki_id"]
        })
        id_cand_dict[line['id']]['candidates'] = cand_list
    
    for k, v in id_cand_dict.items():
        output_f.write(json.dumps(v, ensure_ascii=False) + '\n')
    
    # for i in range(257):
    #     if i not in id_set:
    #         print(i)
    
    for line in tqdm(no_cand_f):
        output_f.write(line)

