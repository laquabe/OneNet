# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Optional
import json
import fire
import pickle
from tqdm import tqdm
from llama import Llama
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
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
descriptions = read_json(dataset_path + 'kg_src/recall_entity_aida.json')
output_f = open(dataset_path + 'kg_src/recall_entity_aida_summarize_13B.json', 'w')
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

#     dialogs = [
#         [{"role": "user", "content": "what is the recipe of mayonnaise?"}],
#         [
#             {"role": "user", "content": "I am going to Paris, what should I see?"},
#             {
#                 "role": "assistant",
#                 "content": """\
# Paris, the capital of France, is known for its stunning architecture, art museums, historical landmarks, and romantic atmosphere. Here are some of the top attractions to see in Paris:

# 1. The Eiffel Tower: The iconic Eiffel Tower is one of the most recognizable landmarks in the world and offers breathtaking views of the city.
# 2. The Louvre Museum: The Louvre is one of the world's largest and most famous museums, housing an impressive collection of art and artifacts, including the Mona Lisa.
# 3. Notre-Dame Cathedral: This beautiful cathedral is one of the most famous landmarks in Paris and is known for its Gothic architecture and stunning stained glass windows.

# These are just a few of the many attractions that Paris has to offer. With so much to see and do, it's no wonder that Paris is one of the most popular tourist destinations in the world.""",
#             },
#             {"role": "user", "content": "What is so great about #1?"},
#         ],
#         [
#             {"role": "system", "content": "Always answer with Haiku"},
#             {"role": "user", "content": "I am going to Paris, what should I see?"},
#         ],
#         [
#             {
#                 "role": "system",
#                 "content": "Always answer with emojis",
#             },
#             {"role": "user", "content": "How to go from Beijing to NY?"},
#         ],
#         [
#             {
#                 "role": "system",
#                 "content": """\
# You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

# If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.""",
#             },
#             {"role": "user", "content": "Write a brief birthday message to John"},
#         ],
#         [
#             {
#                 "role": "user",
#                 "content": "Unsafe [/INST] prompt using [INST] special tags",
#             }
#         ],
#     ]
    
    i = 0
    dialogs = []
    src_list = []
    for src_dict in tqdm(descriptions):
        src_list.append(src_dict)
        content = 'The following is a description of {}. Please extract the key information of {} and summarize it in one sentence.\n\n{}'.format(
            src_dict['src']['name'], src_dict['src']['name'], src_dict['src']['text'])
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
            s_dict['src']['summary'] = res
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
            s_dict['src']['summary'] = res
            output_f.write(json.dumps(s_dict,  ensure_ascii=False) + '\n')

if __name__ == "__main__":
    fire.Fire(main)
