import json
from matplotlib import pyplot as plt
dataset = 'wiki_test_prompt0_sum13B_13B'
input_file_name = '/data/xkliu/EL_datasets/COT_sample/final/aida_train_merge_listwise_repeated.jsonl'
pic_path = '/data/xkliu/EL_datasets/pic/{}_category.png'.format(dataset) 

def draw_pie(static_statistic:dict):
    plt.title('{}_category'.format(dataset))
    plt.pie(static_statistic.values(), labels=static_statistic.keys(), autopct='%0.1f%%')
    plt.savefig(pic_path)

category_dict = {}
with open(input_file_name) as input_f:
    for line in input_f:
        line = json.loads(line.strip())
        cate = line['llm_category']
        category_dict[cate] = category_dict.get(cate, 0) + 1

print(category_dict)
# draw_pie(category_dict)
