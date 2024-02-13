import json
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

dataset = 'wiki_test_prompt0_sum13B_13B'
merge_dataset_path = '/data/xkliu/EL_datasets/result/zephyr/merge/final/'
meta_dataset_path = '/data/xkliu/EL_datasets/datasets_recall/listwise_input/meta/'
case_study_path = '/data/xkliu/EL_datasets/result/case/'

input_file_name = merge_dataset_path + '{}_nocot_prompt1.jsonl'.format(dataset)
meta_file_name = meta_dataset_path + '{}.jsonl'.format(dataset)
output_context_file_name =  case_study_path + '{}_context.jsonl'.format(dataset)
output_prior_file_name = case_study_path + '{}_prior.jsonl'.format(dataset)

meta_cand_dict = {}
with open(meta_file_name) as meta_f:
    for line in meta_f:
        line = json.loads(line.strip())
        in_cand = set()
        out_cand = []
        if len(line['candidates']) == 0:
            continue
        for cand in line['candidates']:
            in_cand.add(cand['name'])
        for cand in line['meta_cands']:
            if cand not in in_cand:
                out_cand.append(cand)
        meta_cand_dict[line['id']] = {'in_cands':list(in_cand), 'out_cands':out_cand} 

same_str = 'The prior and context are the same.'
with open(input_file_name) as input_f, \
    open(output_context_file_name, 'w') as context_f, \
    open(output_prior_file_name, 'w') as prior_f: 

    for line in input_f:
        line = json.loads(line.strip())
        if len(line['candidates']) == 0:
            continue
        if same_str in line['llm_merge']:
            continue
        
        map_dict = {'none':4096}
        for cand in line['candidates']:
            map_dict[str(cand['name']).lower()] = int(cand['wiki_id'])

        final_res = result_decode(line['llm_merge'], map_dict)
        final_pred = map_dict[final_res]
        line['out_cands'] = meta_cand_dict[line['id']]['out_cands']
        line['in_cands'] = meta_cand_dict[line['id']]['in_cands']
        line['context'] = line['left_context'] + '###' + line['mention'] + '###' + line['right_context']
        if final_pred == line['ans_id']:
            context_res = result_decode(line['llm_predict_prompt0'], map_dict)
            context_pred = map_dict[context_res]

            prior_res = result_decode(line['llm_prior'], map_dict)
            prior_pred = map_dict[prior_res]

            if context_pred == line['ans_id']:
                context_f.write(json.dumps(line) + '\n')
            if prior_pred == line['ans_id']:
                prior_f.write(json.dumps(line) + '\n')
