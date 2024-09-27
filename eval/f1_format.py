import json
from tqdm import tqdm
import re

def result_decode(llm_pred:str, map_dict):
    llm_pred = llm_pred.lstrip('<|assistant|>\n')
    llm_pred = llm_pred.lower()
    '''llm result decode
        1. decode json like {'entity_name'}
        2. if fail, use the last entity name in sentence
    '''
    decode_flag = False
    pred_entity = None
    json_str = re.search(r'{.*}', llm_pred)
    if json_str:
        try:
            json_str = json.loads(json_str.group())
            if 'name of the entity' in json_str.keys():
                pred_entity = json_str['name of the entity']
            else:
                for v in json_str.values():
                    if not str(v).isdigit():
                        pred_entity = v
            if pred_entity != None:
                decode_flag = True
        except:
            pass
    if decode_flag:
        '''check name valid'''
        try:
            if pred_entity in map_dict.keys():
                return pred_entity
            else:
                decode_flag = False
        except:
            decode_flag = False

    if not decode_flag:
        index_dict = {}
        for name in map_dict.keys():
            if name in llm_pred:
                pos = llm_pred.find(name) - len(name)
                index_dict[name] = pos
    
        if len(index_dict) > 0:
            pred_entity, _ = sorted(index_dict.items(), key = lambda kv:(kv[1], len(kv[0]), kv[0]), reverse=False)[0]
        else:
            pred_entity = 'none'
    
    return pred_entity

def file_f1(file_name, test_field):
    all_num = 0
    hit_num = 0
    error_num = 0
    with open(file_name) as input_f:
        for line in input_f:
            all_num += 1
            line = json.loads(line)

            if len(line[test_field]) == 0:
                error_num += 1
                continue

            map_dict = {'none':4096}
            for cand in line['candidates']:
                map_dict[str(cand['name']).lower()] = int(cand['wiki_id'])

            pred_entity = result_decode(line[test_field], map_dict)
            pred_num = map_dict[pred_entity]

            if pred_num == line['ans_id']:
                hit_num += 1
            
    print(all_num, hit_num, hit_num / all_num ,error_num)

if __name__ == "__main__":
    datasets_path = '/data/xkliu/EL_datasets/result/zephyr/merge/final/'
    dateset_name = 'ace2004_test_noprompt_sum13B_13B_nocot_prompt1'
    test_file = '{}.jsonl'.format(dateset_name)
    test_field = 'llm_merge'
    file_f1(datasets_path + test_file, test_field)
