import json
from tqdm import tqdm

dataset_path = '/data/xkliu/EL_datasets/result/zephyr/'
dataset = 'wiki_test_prompt0'
prompt_list = [0, 1]
context_file_list = []

for prompt_name in prompt_list:
    context_file_list.append(dataset_path + '{}_sum13B_13B_prompt{}_5top1.jsonl'.format(dataset, prompt_name))

result_dict = {}
prior_file_name = dataset_path + 'prior/{}_sum13B_13B.jsonl'.format(dataset)
output_file_name = dataset_path + 'merge/{}_sum13B_13B.jsonl'.format(dataset)

for i in range(len(prompt_list)):
    file_name = context_file_list[i]
    prompt_name = prompt_list[i]
    with open(file_name) as input_f:
        for line in tqdm(input_f):
            line = json.loads(line)
            l_id = line['id']
            l_ans_dict = result_dict.get(l_id, {})
            l_ans_dict['llm_predict_prompt{}'.format(prompt_name)] = line['llama_predict']
            result_dict[l_id] = l_ans_dict

output_f = open(output_file_name, 'w')
with open(prior_file_name) as prior_file:
    for line in tqdm(prior_file):
        line = json.loads(line)
        l_id = line['id']
        l_ans_dict = result_dict[l_id]
        for k, v in l_ans_dict.items():
            line[k] = v
        output_f.write(json.dumps(line, ensure_ascii=False) + '\n')

