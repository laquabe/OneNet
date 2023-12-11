# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Optional
import json
import fire
import pickle
from tqdm import tqdm
from llama import Llama
import os
import random
import torch.distributed as dist
import datetime

os.environ['CUDA_VISIBLE_DEVICES']='2,3'
# os.environ['OMP_NUM_THREADS']='1'
dataset_path = '/data/xkliu/EL_datasets/'
# des_with_tokens = pickle.load(open(dataset_path + 'kg_src/descriptions_with_len_dict.pickle', 'rb'), encoding='utf-8')
category_list = ['General reference','Culture and the arts','Geography and places','Health and fitness', 'History and events', 'Human activities', 'Mathematics and logic', 'Natural and physical sciences', 'People and self', 'Philosophy and thinking', 'Religion and belief systems', 'Society and social sciences', 'Technology and applied sciences']

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
    content = "You're an entity disambiguator. "
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
    content += """You need to determine which candidate entity is more likely to be the mention. Please refer to the above example, give your reasons, and finally answer serial number of the entity and the name of the entity. If all candidate entities are not appropriate, you can answer '-1.None'."""

    return content

dataset_name = 'ace2004_test_noprompt_sum13B_13B'
# descriptions = read_json(dataset_path + 'kg_src/recall_entity_aida.json')
# dataset = read_json(dataset_path + 'datasets_recall/listwise_input/{}_with_c.jsonl'.format(dataset_name))
dataset = read_json(dataset_path + 'datasets_recall/listwise_input/split/{}_stage1.jsonl'.format(dataset_name))
output_f = open(dataset_path + 'result/{}_noprompt_stage1.jsonl'.format(dataset_name), 'w')
instruction_dict = read_prompt('/data/xkliu/EL_code/MyEL/utils/prompt.jsonl')
cot_dict, cot_global_list = read_cot('/data/xkliu/EL_datasets/COT_sample/final/aida_train_merge_listwise_with_c.jsonl')

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
):
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600))
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    i = 0
    dialogs = []
    src_list = []
    for src_dict in tqdm(dataset):
        src_list.append(src_dict)
        content = prompt_formula(src_dict, cot_dict, cot_global_list)
        dialogs.append(
            [{'role':'user', 'content': content}]
        )
        i += 1
        if i < max_batch_size:
            continue
        try:
            results = generator.chat_completion(
                dialogs,  # type: ignore
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )
            results_batch = []
            for dialog, result in zip(dialogs, results):
                results_batch.append(result['generation']['content'])
            
        except:
            results_batch = []
            for _ in dialogs:
                results_batch.append('')
        for s_dict, res in zip(src_list, results_batch):
            s_dict['llm_answer'] = res
            output_f.write(json.dumps(s_dict,  ensure_ascii=False) + '\n')

        src_list = []
        dialogs = []
        i = 0
        
    
    if len(src_list) > 0:
        try:
            results = generator.chat_completion(
                dialogs,  # type: ignore
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )
            results_batch = []
            for dialog, result in zip(dialogs, results):
                results_batch.append(result['generation']['content'])
        except:
            results_batch = []
            for _ in dialogs:
                results_batch.append('')
        for s_dict, res in zip(src_list, results_batch):
            s_dict['llm_answer'] = res
            output_f.write(json.dumps(s_dict,  ensure_ascii=False) + '\n')
        

if __name__ == "__main__":
    fire.Fire(main)
    # for src_dict in tqdm(dataset):
    #     content = prompt_formula(src_dict, cot_dict,  cot_global_list)
    #     print(content)
    #     exit()