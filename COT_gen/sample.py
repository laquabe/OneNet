import json
from tqdm import tqdm
import random
category_list = ['General reference','Culture and the arts','Geography and places','Health and fitness', 'History and events', 'Human activities', 'Mathematics and logic', 'Natural and physical sciences', 'People and self', 'Philosophy and thinking', 'Religion and belief systems', 'Society and social sciences', 'Technology and applied sciences']
print(len(category_list))
category_num = {}
candidate_len = {'1':0, '>1':0}
with open('aida_train_sample_13B_GPTjudge.jsonl') as input_f:
    for line in tqdm(input_f):
        line = json.loads(line.strip())
        llm_ans = line["llm_answer"]
        llm_ans = llm_ans.lower()
        hit_cat_dict = {}
        for category in category_list:
            category = category.lower()
            if category in llm_ans:
                pos = llm_ans.find(category)
                hit_cat_dict[category] = pos
        if len(hit_cat_dict) > 0:
            category, _ = sorted(hit_cat_dict.items(), key = lambda kv:(kv[1], kv[0]))[0]
            category_num[category] = category_num.get(category, 0) + 1
        if len(line['candidates']) == 1:
            candidate_len['1'] += 1
        else:
            candidate_len['>1'] += 1
    for k,v in category_num.items():
        print('{}:{}'.format(k, v))
    print(candidate_len)
exit()
min_num = 4
sample_category_num = {}
with open('aida_train_13Bcategory.jsonl') as input_f, \
    open('aida_train_sample_13Bcategory.jsonl', 'w') as output_f:

    for line in tqdm(input_f):
        tag_flag = False
        line = json.loads(line.strip())
        if len(line['candidates']) < 2: # have pos
            continue
        llm_ans = line["llm_answer"]
        llm_ans = llm_ans.lower()
        hit_cat_dict = {}
        for category in category_list:
            category = category.lower()
            if category in llm_ans:
                pos = llm_ans.find(category)
                hit_cat_dict[category] = pos
        if len(hit_cat_dict) > 0:
            category, _ = sorted(hit_cat_dict.items(), key = lambda kv:(kv[1], kv[0]))[0]
            if sample_category_num.get(category, 0) > 9:
                continue
            else:
                rand_num = random.randint(0, category_num[category])
                if rand_num < 5 * min_num:
                    output_f.write(json.dumps(line, ensure_ascii=False) + '\n')
                    sample_category_num[category] = sample_category_num.get(category, 0) + 1

    
    for k,v in sample_category_num.items():
        print('{}:{}'.format(k, v))

