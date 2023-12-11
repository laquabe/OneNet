import json
from tqdm import tqdm

def pointwise_judge(gpt_ans:str):
    # find last word position in sentence
    yes_pos = gpt_ans.rfind('yes')
    no_pos = gpt_ans.rfind('no')
    if yes_pos > no_pos:
        return 'yes'
    else:
        return 'no'

def pairwise_judge(gpt_ans:str, first_mention, second_mention):
    #key word: the first, the second, (first_mention), (second_mention)
    position_dict = {}
    position_dict['the first'] = gpt_ans.rfind('the first')
    position_dict['the second'] = gpt_ans.rfind('the second')
    position_dict[first_mention] = gpt_ans.rfind(first_mention)
    position_dict[second_mention] = gpt_ans.rfind(second_mention)
    sort_list = sorted(position_dict.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)
    ans_pos = sort_list[0]
    if (ans_pos[0] == first_mention) or (ans_pos[0] == 'the first'):
        return 1
    elif (ans_pos[0] == second_mention) or (ans_pos[0] == 'the second'):
        return 2
    else:
        return 0
    
def listwise_judge(llm_ans:str, candidates:list):
    '''return el wiki_id'''
    llm_ans = llm_ans.lower()
    cand_id_dict = {'None':-1}
    for cand in candidates:
        cand_id_dict[cand['name']] = int(cand['wiki_id'])
    pos_dict = {}
    for name in cand_id_dict.keys():
        pos_dict[name] = llm_ans.rfind(name.lower()) + len(name)
    sort_list = sorted(pos_dict.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)
    return cand_id_dict[sort_list[0][0]]

point_all, point_hit, pair_all, pair_hit, neg_all, neg_hit = 0,0,0,0,0,0
list_all, list_hit = 0,0
dataset_name = 'aida_train_GPT4_prompt5_listwise'
with open('/data/xkliu/EL_datasets/COT_sample/{}.jsonl'.format(dataset_name)) as input_f, \
    open('final/{}.jsonl'.format(dataset_name), 'w') as output_f:
    
    for line in tqdm(input_f):
        line = json.loads(line.strip())
        gpt_ans = line['gpt_ans']
        gpt_ans = gpt_ans.lower()
        # determine COT type
        if line['COT_type'] == 'pointwise':
            point_all += 1
            gpt_ans = pointwise_judge(gpt_ans)
            if line['ans_id'] == int(line['cand_wiki_id']):
                if gpt_ans == 'no':
                    continue
            else:
                if gpt_ans == 'yes':
                    continue
            point_hit += 1
            output_f.write(json.dumps(line, ensure_ascii=False) + '\n')
        elif line['COT_type'] == 'pairwise':
            pair_all += 1
            gpt_ans = pairwise_judge(gpt_ans, line['pos_cand_name'].lower(), line['pos_cand_name'].lower())
            if gpt_ans != line['pos_position']:
                continue
            pair_hit += 1
            output_f.write(json.dumps(line, ensure_ascii=False) + '\n')
        elif line['COT_type'] == 'neg_pair':
            neg_all += 1
            if 'none' not in gpt_ans:
                continue
            neg_hit += 1
            output_f.write(json.dumps(line, ensure_ascii=False) + '\n')
        elif line['COT_type'] == 'listwise':
            list_all += 1
            gpt_ans = listwise_judge(gpt_ans, line['candidates'])
            if gpt_ans == -1:
                if line['pos_position'] == -1:
                    list_hit += 1
                    output_f.write(json.dumps(line, ensure_ascii=False) + '\n')
            else:
                if gpt_ans == line['ans_id']:
                    list_hit += 1
                    output_f.write(json.dumps(line, ensure_ascii=False) + '\n')

# print(point_hit / point_all, pair_hit / pair_all, neg_hit / neg_all,
#       (point_hit + pair_hit + neg_hit) / (point_all + pair_all + neg_all))
print(list_all, list_hit, list_hit / list_all)