# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Optional
import json
import fire
import pickle
from tqdm import tqdm
from llama import Llama
import os
os.environ['CUDA_VISIBLE_DEVICES']='3,4'
# os.environ['OMP_NUM_THREADS']='1'
dataset_path = '/data/xkliu/EL_datasets/'
# des_with_tokens = pickle.load(open(dataset_path + 'kg_src/descriptions_with_len_dict.pickle', 'rb'), encoding='utf-8')
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
def prompt_formula(src_dict):
    # content = "You're an entity disambiguator. I'll give you some tips on entity disambiguation, and you need to pay attention to these textual features:\n\n"
    content = "You're an entity disambiguator. I'll give you the description of entity disambiguation and some tips on entity disambiguation, and you need to pay attention to these textual features:\n\n"
    # content = "You're an entity disambiguator. "
    content += instruction_dict[0]['prompt']
    content += '\n\n'
    content += "Now, I'll give you a mention, a context, and a candidate entity, and the mention will be highlighted with '###'.\n\n"
    content += 'mention:{}\n'.format(src_dict['mention'])

    context = src_dict['left_context'] + ' ###' + src_dict['mention'] + '### ' + src_dict['right_context']
    context = context.strip()
    context = ' '.join(context.split())
    content += 'context:{}\n'.format(context)

    cand_entity = '{}.{}'.format(src_dict['cand_name'], src_dict['cand_text'])
    content += 'candidate entity:{}\n\n'.format(cand_entity)

    content += """You need to determine if the mention and the candidate entity are related. Please refer to the above tips and give your reasons, and finally answer 'yes' or 'no'. Answer 'yes' when you think the information is insufficient or uncertain."""
    # content += "You need to determine if the mention and the candidate entity are related. Please refer to the above tips and **only** answer 'yes' or 'no'. Answer 'yes' when you think the information is insufficient or uncertain."
    # content += """You need to determine if the mention and the candidate entity are related. Please give your reasons, and answer 'yes' or 'no'. Answer 'yes' when you think the information is insufficient or uncertain."""

    return content

dataset_name = 'wiki'
# descriptions = read_json(dataset_path + 'kg_src/recall_entity_aida.json')
dataset = read_json(dataset_path + 'datasets_recall/flatten/{}_test_13B_top10g.jsonl'.format(dataset_name))
output_f = open(dataset_path + 'datasets_recall/cand_filter/{}_test_prompt0_text_13B.jsonl'.format(dataset_name), 'w')
instruction_dict = read_prompt('/data/xkliu/EL_code/MyEL/utils/prompt.jsonl')

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
):
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
        content = prompt_formula(src_dict)
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
    #     content = prompt_formula(src_dict)
    #     print(content)
    #     exit()