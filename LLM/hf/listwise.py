from transformers import AutoTokenizer
import transformers
import torch
import os
import json
from tqdm import tqdm
import random

os.environ['CUDA_VISIBLE_DEVICES']='1,3'
model = "/data/xkliu/LLMs/models/llama2-13b-chat-hf"
dataset_path = '/data/xkliu/EL_datasets/'
category_list = ['General reference','Culture and the arts','Geography and places','Health and fitness', 'History and events', 'Human activities', 'Mathematics and logic', 'Natural and physical sciences', 'People and self', 'Philosophy and thinking', 'Religion and belief systems', 'Society and social sciences', 'Technology and applied sciences']

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

def read_json(file_name):
    data_list = []
    with open(file_name) as src_f:
        for line in src_f:
            line = json.loads(line.strip())
            data_list.append(line)
    return data_list

def read_prompt(file_name):
    prompt_dict = {}
    with open(file_name) as prompt_f:
        for line in prompt_f:
            line = json.loads(line.strip())
            prompt_dict[line['id']] = line
    
    return prompt_dict

def list_prompt_formula(src_dict:dict):
    content = 'mention:{}\n'.format(src_dict['mention'])

    context = src_dict['left_context'] + ' ###' + src_dict['mention'] + '### ' + src_dict['right_context']
    context = context.strip()
    context = ' '.join(context.split())
    content += 'context:{}\n'.format(context)

    candidates = random.sample(src_dict['candidates'], len(src_dict['candidates']))
    i = 1
    for cand in candidates:
        cand_entity = '{}.{}'.format(cand['name'], cand['summary'])
        content += 'entity {}:{}\n'.format(i, cand_entity)
        i += 1
    content += '\n'
    return content

def read_cot(cot_file_name):
    cot_dict = {}
    cot_global_list = []
    with open(cot_file_name) as input_f:
        for line in tqdm(input_f):
            line = json.loads(line.strip())
            prompt = list_prompt_formula(line)
            prompt += 'Answer: {}\n\n'.format(line['gpt_ans'].replace('\n',''))
            cot_list = cot_dict.get(line['llm_category'], [])
            cot_list.append(prompt)
            cot_dict[line['llm_category']] = cot_list
            cot_global_list.append(prompt)
    return cot_dict, cot_list

def prompt_formula(src_dict, cot_dict, cot_global_list):
    # content = "You're an entity disambiguator. I'll give you some tips on entity disambiguation, you should pay attention to these textual features:\n\n"
    # content = "You're an entity disambiguator. I'll give you the description of entity disambiguation and some tips on entity disambiguation, you should pay attention to these textual features:\n\n"
    content = "<<SYS>>You're an entity disambiguator.<</SYS>> "
    # content += instruction_dict[0]['prompt']
    # content += '\n\n'
    cot_case_list = cot_dict.get(src_dict['llm_category'], [])
    if len(cot_case_list) > 0:
        cot_case = random.sample(cot_case_list, 1)[0]
    else:
        cot_case = random.sample(cot_global_list, 1)[0]
    content += 'The following example will help you understand the task:\n\n'
    content += cot_case

    content += "Now, I'll give you a mention, a context, and a list of candidates entities, the mention will be highlighted with '###' in context.\n\n"
    content += 'mention:{}\n'.format(src_dict['mention'])

    context = src_dict['left_context'] + ' ###' + src_dict['mention'] + '### ' + src_dict['right_context']
    context = context.strip()
    context = ' '.join(context.split())
    content += 'context:{}\n'.format(context)

    candidates = random.sample(src_dict['candidates'], len(src_dict['candidates']))
    i = 1
    for cand in candidates:
        cand_entity = '{}.{}'.format(cand['name'], cand['summary'])
        content += 'entity {}:{}\n'.format(i, cand_entity)
        i += 1
    content += '\n'

    # content += """You need to determine which candidate entity is more likely to be the mention. Please refer to the above tips and examples, give your reasons, and finally answer serial number of the entity and the name of the entity. If all candidate entities are not appropriate, you can answer '-1.None'."""
    content += """You need to determine which candidate entity is more likely to be the mention. Please refer to the above example, give your reasons, and finally answer serial number of the entity and the name of the entity. 
    If all candidate entities are not appropriate, you can answer '-1.None'."""

    return content

dataset_name = 'msnbc_test_prompt0_sum13B_13B'
dataset = read_json(dataset_path + 'datasets_recall/listwise_input/{}_with_c.jsonl'.format(dataset_name))
output_f = open(dataset_path + 'result/{}_noprompt.jsonl'.format(dataset_name), 'w')
instruction_dict = read_prompt('/data/xkliu/EL_code/MyEL/utils/prompt.jsonl')
cot_dict, cot_global_list = read_cot('/data/xkliu/EL_datasets/COT_sample/final/aida_train_merge_listwise_with_c.jsonl')

for src_dict in tqdm(dataset):
    content = prompt_formula(src_dict, cot_dict,  cot_global_list)
    content = ('<s>[INST]' + content + '[/INST]')
    # print(content)
    try:
        sequences = pipeline(
            content,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=4096,
        )
        for seq in sequences:
            gen_text = seq['generated_text']
            gen_text = gen_text[len(content):]
            src_dict['llama_predict'] = gen_text
            output_f.write(json.dumps(src_dict, ensure_ascii=False) + '\n')
    except:
        src_dict['llama_predict'] = ''
        output_f.write(json.dumps(src_dict, ensure_ascii=False) + '\n')
        

