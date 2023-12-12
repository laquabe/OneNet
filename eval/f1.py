import json
from tqdm import tqdm

datasets_path = '/data/xkliu/EL_datasets/result/'
dateset_name = 'msnbc_test_prompt0'
recall_file = '{}_sum13B_13B_noprompt.jsonl'.format(dateset_name)

all_num = 0
hit_num = 0
error_num = 0
bad_case_file = open(datasets_path.replace('result', 'badcase') + recall_file, 'w')
with open(datasets_path + recall_file, 'r') as input_f:
    for line in tqdm(input_f):
        all_num += 1
        line = json.loads(line.strip())
        if len(line['llama_predict']) == 0:
            error_num += 1
            continue
        map_dict = {'none':-1}
        # i = 1
        cand_id_set = set()
        for cand in line['candidates']:
            map_dict[cand['name'].lower()] = int(cand['wiki_id'])
            cand_id_set.add(int(cand['wiki_id']))
        index_dict = {}
        for name in map_dict.keys():
            llm_ans = line['llama_predict'].lower()
            if name in llm_ans:
                pos = llm_ans.rfind(name) + len(name)
                index_dict[name] = pos
        if len(index_dict) > 0:
            pred_name, _ = sorted(index_dict.items(), key = lambda kv:(kv[1], len(kv[0]), kv[0]), reverse=True)[0]
            pred_num = map_dict[pred_name]
        else:
            pred_name = 'none'
            pred_num = -1
        
        if pred_num == line['ans_id']:
            hit_num += 1
        else:
            if line['ans_id'] in cand_id_set:
                line['decode_name'] = pred_name
                line['decode_id'] = pred_num
                bad_case_file.write(json.dumps(line, ensure_ascii=False) + '\n')

print(all_num, hit_num, hit_num / all_num ,error_num)