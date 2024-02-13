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

def judge_decode(llm_pred:str):
    llm_pred = llm_pred.lstrip('<|assistant|>\n')
    llm_pred = llm_pred.lower()
    '''llm judge decode
        "yes" "no"
        other:  yes
    '''
    pos_str_list = ['"yes"', "'yes'", 'be correct', 'is correct']
    neg_str_list = ['"no"', "'no'", 'incorrect', 'be not correct', 'is not correct']
    pos_index, neg_index = len(llm_pred), len(llm_pred)
    for pos_str in pos_str_list:
        if pos_str not in llm_pred:
            continue
        index = llm_pred.find(pos_str) - len(pos_str)
        if index < pos_index:
            pos_index = index

    for neg_str in neg_str_list:
        if neg_str not in llm_pred:
            continue
        index = llm_pred.find(neg_str) - len(neg_str)
        if index < neg_index:
            neg_index = index 

    if pos_index <= neg_index:
        return True
    else:
        return False

# bad_f = open('/data/xkliu/EL_datasets/result/zephyr/badcase/ace2004_test_noprompt_sum13B_13B_noprompt.jsonl', 'w')
def file_f1(file_name, test_field, judge_field):
    all_num = 0
    hit_num = 0
    error_num = 0
    tp, tn, fp, fn = 0,0,0,0
    with open(file_name) as input_f:
        for line in input_f:
            all_num += 1
            line = json.loads(line)

            if len(line[test_field]) == 0:
                error_num += 1
                continue

            map_dict = {'none':4096}
            for cand in line['candidates']:
                map_dict[cand['name'].lower()] = int(cand['wiki_id'])

            pred_entity = result_decode(line[test_field], map_dict)
            pred_num = map_dict[pred_entity]

            judge_res = judge_decode(line[judge_field])
            if pred_num == line['ans_id']:
                hit_num += 1
                if judge_res == True:
                    tp += 1
                else:
                    fp += 1
            else:
                if judge_res == True:
                    tn += 1
                else:
                    fn += 1
            

            # else:
            #     bad_f.write(json.dumps(line,ensure_ascii=False) + '\n')
            
    print(all_num, hit_num, hit_num / all_num ,error_num)
    print(tp, tn, fp, fn)

def merge_decode(llm_pred):
    llm_pred = llm_pred.lstrip('<|assistant|>\n')
    llm_pred = llm_pred.lower()
    '''llm judge decode
        "yes" "no"
        other:  yes
    '''
    pos_str_list = ['the first inference']
    neg_str_list = ['the second inference']
    pos_index, neg_index = len(llm_pred), len(llm_pred)
    for pos_str in pos_str_list:
        if pos_str not in llm_pred:
            continue
        index = llm_pred.find(pos_str) - len(pos_str)
        if index < pos_index:
            pos_index = index

    for neg_str in neg_str_list:
        if neg_str not in llm_pred:
            continue
        index = llm_pred.find(neg_str) - len(neg_str)
        if index < neg_index:
            neg_index = index 

    if pos_index < neg_index:
        return 'context'
    else:
        return 'prior'
    
def file_f1_merge(file_name, context_field, prior_field, judge_field):
    all_num = 0
    hit_num = 0
    error_num = 0
    context_num = 0
    prior_num = 0
    merge_error_num = 0
    with open(file_name) as input_f:
        for line in input_f:
            all_num += 1
            line = json.loads(line)

            if len(line[judge_field]) == 0:
                merge_error_num += 1
                judge_res == 'context'
            else:
                judge_res = merge_decode(line[judge_field])
                if judge_res == 'context':
                    context_num += 1
                elif judge_res == 'prior':
                    prior_num += 1
            
            if (len(line[context_field]) == 0) and (len(line[prior_field]) == 0):
                error_num += 1
                continue

            map_dict = {'none':4096}
            for cand in line['candidates']:
                map_dict[cand['name'].lower()] = int(cand['wiki_id'])

            if len(line[context_field]) == 0:
                context_pred_num = -1
            else:
                context_pred_entity = result_decode(line[context_field], map_dict)
                context_pred_num = map_dict[context_pred_entity]

            if len(line[prior_field]) == 0:
                prior_pred_num = -1
            else:
                prior_pred_entity = result_decode(line[prior_field], map_dict)
                prior_pred_num = map_dict[prior_pred_entity]
            
            if judge_res == 'context':
                pred_num = context_pred_num
            else:
                pred_num = prior_pred_num
            
            if pred_num == line['ans_id']:
                hit_num += 1
            
            # else:
            #     bad_f.write(json.dumps(line,ensure_ascii=False) + '\n')
            
    print(all_num, hit_num, hit_num / all_num ,error_num)
    print(all_num, merge_error_num, context_num, prior_num)

def cand_filter(cand_list:list, hit_set:set):
    new_cand = []
    for cand in cand_list:
        if int(cand['wiki_id']) in hit_set:
            new_cand.append(cand)
    return new_cand

def file_f1_merge(file_name, context_field, prior_field, judge_field, support_field, output_file_name):
    output_f = open(output_file_name, 'w')
    with open(file_name) as input_f:
        for line in tqdm(input_f):
            line = json.loads(line)
            
            if (len(line[context_field]) == 0) and (len(line[prior_field]) == 0):
                output_f.write(json.dumps(line, ensure_ascii=False) + '\n')
                continue

            map_dict = {'none':4096}
            for cand in line['candidates']:
                map_dict[cand['name'].lower()] = int(cand['wiki_id'])

            if len(line[context_field]) == 0:
                context_pred_num = -1
            else:
                context_pred_entity = result_decode(line[context_field], map_dict)
                context_pred_num = map_dict[context_pred_entity]

            if len(line[prior_field]) == 0:
                prior_pred_num = -1
            else:
                prior_pred_entity = result_decode(line[prior_field], map_dict)
                prior_pred_num = map_dict[prior_pred_entity]
            
            if (context_pred_num == -1) and (prior_pred_num == -1):
                output_f.write(json.dumps(line, ensure_ascii=False) + '\n')
                continue

            new_cand = cand_filter(line['candidates'], set([context_pred_num, prior_pred_num]))
            line['candidates'] = new_cand
            output_f.write(json.dumps(line, ensure_ascii=False) + '\n')
                


if __name__ == "__main__":
    datasets_path = '/data/xkliu/EL_datasets/result/zephyr/merge/'
    dataset_name = 'wiki_test_prompt0'
    recall_file = '{}_sum13B_13B.jsonl'.format(dataset_name)
    test_field = 'llm_predict_prompt0'
    prior_field = 'llm_prior'
    judge_field = 'llm_merge'
    support_field = 'llm_predict_prompt1'
    output_path = '/data/xkliu/EL_datasets/result/zephyr/merge/filter/'
    
    file_f1_merge(datasets_path + recall_file, test_field, prior_field, judge_field, support_field, 
                  output_path + '{}.jsonl'.format(dataset_name))
