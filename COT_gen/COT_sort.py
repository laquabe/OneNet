import numpy as np
import json
from tqdm import tqdm

src_file_name = 'wiki_test_prompt0'
cot_embed_arr = np.load('/data/xkliu/EL_datasets/embedding/aida_train_merge.npy')
src_embed_list = np.load('/data/xkliu/EL_datasets/embedding/{}.npy'.format(src_file_name))
map_file = open('/data/xkliu/EL_datasets/embedding/{}_cot_catonly.jsonl'.format(src_file_name), 'w')
cot_file = open('/data/xkliu/EL_datasets/COT_sample/final/aida_train_merge_listwise_repeated.jsonl')
src_file = open('/data/xkliu/EL_datasets/result/{}_sum13B_13B_noprompt.jsonl'.format(src_file_name))

category_dict = {}
category_list = ['General reference','Culture and the arts','Geography and places','Health and fitness', 'History and events', 'Human activities', 'Mathematics and logic', 'Natural and physical sciences', 'People and self', 'Philosophy and thinking', 'Religion and belief systems', 'Society and social sciences', 'Technology and applied sciences']

for cate in category_list:
    category_dict[cate.lower()] = []

for line in cot_file:
    line = json.loads(line.strip())
    line_category = line['llm_category']
    for cate in category_list:
        if line_category.lower() == cate.lower():
            cate_list = category_dict[cate.lower()]
            cate_list.append(1)
            category_dict[cate.lower()] = cate_list
        else:
            cate_list = category_dict[cate.lower()]
            cate_list.append(0)
            category_dict[cate.lower()] = cate_list

category_dict['any'] = [0]*65
category_embed_dict = {}
for k, v in category_dict.items():
    category_embed_dict[k] = np.array(v)

src_embed_list = src_embed_list.tolist()
src_file_list = src_file.readlines()
alpha = 0.0

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

for src_embed, line in tqdm(zip(src_embed_list, src_file_list)):
    line = json.loads(line.strip())
    line_category = line['llm_category']
    cate_embed = category_embed_dict[line_category.lower()]
    src_embed = np.array(src_embed)     # d
    src_embed = src_embed[:,np.newaxis]
    score_embed = np.dot(cot_embed_arr, src_embed)
    score_embed = np.squeeze(score_embed)
    score_embed = normalization(score_embed)
    score_embed = alpha * score_embed + (1 - alpha) * cate_embed
    score_embed = np.argsort(-score_embed)
    # print(score_embed[:10])
    score_list = score_embed.tolist()
    map_file.write(json.dumps({'cand_index':score_list}) + '\n')

