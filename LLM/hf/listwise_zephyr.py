from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import os
import json
from tqdm import tqdm
import random
from transformers import pipeline

# os.environ['CUDA_VISIBLE_DEVICES']='0,1'
# os.environ["WORLD_SIZE"] = "1"
model = "/data/xkliu/LLMs/models/zephyr-7b-beta"
dataset_path = '/data/xkliu/EL_datasets/'
category_list = ['General reference','Culture and the arts','Geography and places','Health and fitness', 'History and events', 'Human activities', 'Mathematics and logic', 'Natural and physical sciences', 'People and self', 'Philosophy and thinking', 'Religion and belief systems', 'Society and social sciences', 'Technology and applied sciences']

# tokenizer = AutoTokenizer.from_pretrained(model)
# model = AutoModelForCausalLM.from_pretrained(model)
# model = model.to(device)
# pipe = pipeline("text-generation", model=model, torch_dtype=torch.bfloat16, device_map="auto")

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
    cot_index_dict = {}
    i = 0
    with open(cot_file_name) as input_f:
        for line in tqdm(input_f):
            line = json.loads(line.strip())
            prompt = list_prompt_formula(line)
            prompt += 'Answer: {}\n\n'.format(line['gpt_ans'].replace('\n',''))
            cot_list = cot_dict.get(line['llm_category'], [])
            cot_list.append(prompt)
            cot_dict[line['llm_category']] = cot_list
            
            cot_index_dict[i] = prompt
            i += 1
    return cot_dict, cot_index_dict

def prompt_formula_judge(src_dict, judge_key):
    system_content = "You're an entity disambiguator."

    content = "I'll give you a mention, a context, and a list of candidates entities, the mention will be highlighted with '###' in context.\n\n"
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
    content += """You need to determine which candidate entity is more likely to be the mention. Please refer to the above example, give your reasons, and finally answer serial number of the entity and the name of the entity. If all candidate entities are not appropriate, you can answer '-1.None'."""

    check_content = "Now please double-check the reasoning process above. "
    check_content += 'let’s think step by step, analyze its correctness, and finally answer "yes" or "no".'
    # content += 'You should answer in the following json format {"Double-check result":}'
    
    messages = [
    {
        "role": "system",
        "content": system_content,
    },
    {"role": "user", "content": content},
    {"role": "assistant", "content": src_dict[judge_key].strip('<|assistant|>\n')},
    {"role": "user", "content": check_content}
    ]

    return messages

def prompt_formula_prior(src_dict):
    system_content = "You're an entity disambiguator."
    content = "I'll provide you a mention and its candidates below.\n\n"
    content += 'mention:{}\n'.format(src_dict['mention'])

    candidates = random.sample(src_dict['candidates'], len(src_dict['candidates']))
    i = 1
    for cand in candidates:
        cand_entity = '{}.{}'.format(cand['name'], cand['summary'])
        content += 'entity {}:{}\n'.format(i, cand_entity)
        i += 1
    content += '\n'

    content += 'Based on your knowledge, please determine which is the most likely entity when people refer to mention "{}", '.format(src_dict['mention'])
    content += "and finally answer the name of the entity."
    # content += 'You should answer in the following json format {"Name of the entity":}'

    messages = [
    {
        "role": "system",
        "content": system_content,
    },
    {"role": "user", "content": content},
    ]

    return messages

def read_cot_cand(file_name):
    cand_list = []
    with open(file_name) as input_f:
        for line in input_f:
            line = json.loads(line.strip())
            cand_list.append(line['cand_index'])
    
    return cand_list

def prompt_formula(src_dict, cot_dict, cot_index_dict, cot_cand, topk):
    system_content = "You're an entity disambiguator. I'll give you some tips on entity disambiguation, you should pay attention to these textual features:\n\n"
    # system_content = "You're an entity disambiguator. I'll give you the description of entity disambiguation and some tips on entity disambiguation, you should pay attention to these textual features:\n\n"
    # system_content = "You're an entity disambiguator."
    system_content += instruction_dict[4]['prompt']
    # content += '\n\n'

    '''only category'''
    # cot_case_list = cot_dict.get(src_dict['llm_category'], [])
    # if len(cot_case_list) > 0:
    #     cot_case = random.sample(cot_case_list, 1)[0]
    # else:
    #     cot_case = random.sample(cot_global_list, 1)[0]
    '''category and sentence sim'''
    cot_index_list = cot_cand[:topk]
    cot_index = random.sample(cot_index_list, 1)[0]
    # print(cot_index)
    cot_case = cot_index_dict[cot_index]
    
    content = 'The following example will help you understand the task:\n\n'
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
    # content += """You need to determine which candidate entity is more likely to be the mention. Please refer to the above example, give your reasons, and finally answer serial number of the entity and the name of the entity. 
    # If all candidate entities are not appropriate, you can answer '-1.None'.You should answer in the following json format {"Serial number":, "Name of the entity":}"""
    content += """You need to determine which candidate entity is more likely to be the mention. Please refer to the above example, give your reasons, and finally answer serial number of the entity and the name of the entity. If all candidate entities are not appropriate, you can answer '-1.None'."""

    messages = [
    {
        "role": "system",
        "content": system_content,
    },
    {"role": "user", "content": content},
    ]

    return messages

dataset_name = 'wiki_test_prompt0'
'''listwise or prior'''
# dataset = read_json(dataset_path + 'datasets_recall/listwise_input/{}_sum13B_13B_with_c.jsonl'.format(dataset_name))
'''judge'''
dataset = read_json(dataset_path + 'result/zephyr/{}_sum13B_13B_prompt1_5top1.jsonl'.format(dataset_name))

output_f = open(dataset_path + 'result/zephyr/judge/{}_sum13B_13B_prompt1_5top1.jsonl'.format(dataset_name), 'w')
output_key = 'llm_judge'
max_new_token = 1024
instruction_dict = read_prompt('/data/xkliu/EL_code/LLM4EL/prompt/prompt.jsonl')
cot_dict, cot_index_dict = read_cot('/data/xkliu/EL_datasets/COT_sample/final/aida_train_merge_listwise_repeated.jsonl')
dataset_cand_list = read_cot_cand('/data/xkliu/EL_datasets/embedding/{}_cot.jsonl'.format(dataset_name))
topk = 1

for src_dict, cot_cand in tqdm(zip(dataset, dataset_cand_list)):
    # messages = prompt_formula(src_dict, cot_dict, cot_index_dict, cot_cand, topk)  # listwise
    messages = prompt_formula_judge(src_dict, 'llama_predict')                                     # judge
    # messages = prompt_formula_prior(src_dict)                                        # prior
    
    print(json.dumps(messages))
    exit()
    try:
        prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = pipe(prompt, max_new_tokens=max_new_token, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        gen_text = outputs[0]["generated_text"]
        gen_start_pos = gen_text.find('<|assistant|>')
        gen_text = gen_text[gen_start_pos:]
        # print(gen_text)
        src_dict[output_key] = gen_text
        # exit()
        output_f.write(json.dumps(src_dict, ensure_ascii=False) + '\n')
    except:
        src_dict[output_key] = ''
        output_f.write(json.dumps(src_dict, ensure_ascii=False) + '\n')
    

