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

# bad_f = open('/data/xkliu/EL_datasets/result/zephyr/badcase/ace2004_test_noprompt_sum13B_13B_noprompt.jsonl', 'w')
def file_f1_merge(file_name, test_field):
    all_num = 0
    hit_num = 0
    error_num = 0
    with open(file_name) as input_f:
        for line in input_f:
            all_num += 1
            line = json.loads(line)

            test_error = []
            for test_key in test_field:
                if len(line[test_key]) == 0:
                    test_error.append(1)
                    continue

                map_dict = {'none':4096}
                for cand in line['candidates']:
                    map_dict[cand['name'].lower()] = int(cand['wiki_id'])

                pred_entity = result_decode(line[test_key], map_dict)
                pred_num = map_dict[pred_entity]

                if pred_num == line['ans_id']:
                    hit_num += 1
                    break
            
    print(all_num, hit_num, hit_num / all_num ,error_num)

def vote_result_1(res_dict:dict):
    if "llm_predict_prompt0" not in res_dict.keys() or "llm_predict_prompt1" not in res_dict.keys():
        return res_dict['llm_prior']
    if res_dict["llm_predict_prompt0"] == res_dict["llm_predict_prompt1"]:
        return res_dict["llm_predict_prompt0"]
    else:
        return res_dict['llm_prior']

def vote_result_2(res_dict:dict):
    res_list = list(res_dict.values())
    res_rank = {}
    for i in res_list:
        num = res_rank.get(i, 0)
        res_rank[i] = num + 1
    return sorted(res_rank.items(), key=lambda item: item[1], reverse=True)[0][0]

def file_f1_vote(file_name, test_field, vote_type=1):
    all_num = 0
    hit_num = 0
    error_num = 0
    with open(file_name) as input_f:
        for line in input_f:
            all_num += 1
            line = json.loads(line)

            test_dict = {'llm_prior':-1}
            for test_key in test_field:
                if len(line[test_key]) == 0:
                    continue

                map_dict = {'none':4096}
                for cand in line['candidates']:
                    map_dict[cand['name'].lower()] = int(cand['wiki_id'])

                pred_entity = result_decode(line[test_key], map_dict)
                pred_num = map_dict[pred_entity]
                test_dict[test_key] = pred_num
            if vote_type == 1:
                pred_num = vote_result_1(test_dict)
            else:
                pred_num = vote_result_1(test_dict)
            if pred_num == line['ans_id']:
                hit_num += 1
            
    print(all_num, hit_num, hit_num / all_num ,error_num)

def file_f1_judge(file_name, test_field):
    pass

if __name__ == "__main__":
    datasets_path = '/data/xkliu/EL_datasets/result/zephyr/merge/'
    dateset_name = 'cweb_test_prompt0'
    recall_file = '{}_sum13B_13B.jsonl'.format(dateset_name)
    test_field = ['llm_prior',"llm_predict_prompt0", "llm_predict_prompt1"]
    # file_f1_merge(datasets_path + recall_file, test_field)
    file_f1_vote(datasets_path + recall_file, test_field, 2)
