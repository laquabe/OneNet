import json
from tqdm import tqdm

category_list = ['General reference','Culture and the arts','Geography and places','Health and fitness', 'History and events', 'Human activities', 'Mathematics and logic', 'Natural and physical sciences', 'People and self', 'Philosophy and thinking', 'Religion and belief systems', 'Society and social sciences', 'Technology and applied sciences']
def classifer(llm_ans:str):
    llm_ans = llm_ans.lower()
    hit_cat_dict = {}
    for category in category_list:
        category = category.lower()
        if category in llm_ans:
            pos = llm_ans.find(category)
            hit_cat_dict[category] = pos
    if len(hit_cat_dict) > 0:
        category, _ = sorted(hit_cat_dict.items(), key = lambda kv:(kv[1], kv[0]))[0]
    else:
        category = 'Any'
    return category

dataset = 'wiki_test_prompt0_sum13B_13B'
with open('/data/xkliu/EL_datasets/datasets_recall/listwise_input/llm_category/{}.jsonl'.format(dataset)) as input_f, \
    open('/data/xkliu/EL_datasets/datasets_recall/listwise_input/{}_with_c.jsonl'.format(dataset), 'w') as output_f:

    for line in tqdm(input_f):
        line = json.loads(line.strip())
        llm_ans = line["llm_category"]
        line_category = classifer(llm_ans)
        line["llm_category"] = line_category
        # del line['llm_answer']
        output_f.write(json.dumps(line, ensure_ascii=False) + '\n')