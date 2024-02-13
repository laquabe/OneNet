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
# model = "/data/xkliu/LLMs/models/llama2-13b-chat-hf"
dataset_path = '/data/xkliu/EL_datasets/'
category_list = ['General reference','Culture and the arts','Geography and places','Health and fitness', 'History and events', 'Human activities', 'Mathematics and logic', 'Natural and physical sciences', 'People and self', 'Philosophy and thinking', 'Religion and belief systems', 'Society and social sciences', 'Technology and applied sciences']

# tokenizer = AutoTokenizer.from_pretrained(model)
# model = AutoModelForCausalLM.from_pretrained(model)
# model = model.to(device)
pipe = pipeline("text-generation", model=model, torch_dtype=torch.bfloat16, device_map="auto")

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
    content = 'Mention:{}\n'.format(src_dict['mention'])

    context = src_dict['left_context'] + ' ###' + src_dict['mention'] + '### ' + src_dict['right_context']
    context = context.strip()
    context = ' '.join(context.split())
    content += 'Context:{}\n'.format(context)

    candidates = random.sample(src_dict['candidates'], len(src_dict['candidates']))
    i = 1
    for cand in candidates:
        cand_entity = '{}.{}'.format(cand['name'], cand['summary'])
        content += 'Entity {}:{}\n'.format(i, cand_entity)
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
    content += 'Mention:{}\n'.format(src_dict['mention'])

    context = src_dict['left_context'] + ' ###' + src_dict['mention'] + '### ' + src_dict['right_context']
    context = context.strip()
    context = ' '.join(context.split())
    content += 'Context:{}\n'.format(context)

    candidates = random.sample(src_dict['candidates'], len(src_dict['candidates']))
    i = 1
    for cand in candidates:
        cand_entity = '{}.{}'.format(cand['name'], cand['summary'])
        content += 'Entity {}:{}\n'.format(i, cand_entity)
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

def prompt_formula_merge(src_dict, first_reason_key, second_resaon_key, context_flag=False):
    system_content = "You're a serious fact checker."

    first_reason = src_dict[first_reason_key].strip('<|assistant|>\n').replace('\n\n','\n')
    index = first_reason.find("Now, I'll give you a mention,")
    if index == -1:
        index = len(first_reason)
    first_reason = first_reason[:index].strip('\n')

    second_reason = src_dict[second_resaon_key].strip('<|assistant|>\n').replace('\n\n','\n')
    index = second_reason.find("Now, I'll give you a mention,")
    if index == -1:
        index = len(second_reason)
    second_reason = second_reason[:index].strip('\n')

    content = ''

    content += 'There are two inferences about the same mention, the first based on context and the second based on common sense. The first may be wrong because the context contains noise, and the second may be wrongly inferred because there is not enough information.\n' # handcraft
    content += 'The following example will help you understand the task:\n\n'
    content += 'Example 1:\n\n'
    content += "The First Inferences: Based on the provided context and the description of the entity disambiguation task, the most likely candidate entity for the mention \"Marxist\" in the given context is entity 3: Marxism.\nHere are the reasons for this choice based on the features provided:\na. Categories: The mention \"Marxist\" is closely related to the category of Marxism, which is a method of socioeconomic analysis that views class relations and social conflict. This aligns with the mention in the context.\nb. Modifiers: There are no specific modifiers provided in the context to distinguish the type of Marxism, such as Karl Marx's specific theories or other related concepts, so the general term \"Marxist\" aligns with the context.\nc. Contextual clues: The mention \"Marxist\" is mentioned in the context of political leaders and government changes, which directly relates to Marxist ideology and its influence on political systems.\nd. Semantic meaning: The context discusses the history of Guinea's president, Lansana Conte, who seized power after the death of a Marxist leader, which suggests a focus on Marxist ideology and its impact on political developments.\nTherefore, the most suitable entity is:\n3. Marxism\nI hope this helps! Let me know if you have any other questions.\n\n"
    content += "The Second Inferences: The most likely entity when people refer to the mention \"Marxist\" is Karl Marx.\n\n"
    content += "Answer: Based on the information provided, the more plausible inference between the two would be \"The First Inferences.\" This is because the first inference method takes into account specific contextual clues, categories, and modifiers related to the mention \"Marxist\" within the given text. It provides a detailed analysis of why \"Marxism\" is the most suitable entity based on the features provided.\nThe second inference, while based on common sense, fails to consider the specific details and contextual clues provided in the text, leading to an incorrect inference. Therefore, the first inference, which considers the immediate textual context and relevant features, is the more plausible choice.\n\n"
    content += 'Example 2:\n\n'
    content += "The First Inferences: Based on the given context and the features provided, the most suitable entity for the mention \"GMT\" is:\n3. Large Millimeter Telescope. \nReasons:\na. Categories: The Large Millimeter Telescope is a telescope designed to observe radio waves, which aligns with the mention being a time zone.\nb. Modifiers: There are no qualifying words in the context that provide additional information about the mention.\nc. Contextual clues: The mention \"GMT\" appears in the context of discussing time and being associated with a local time. The telescope's capability to study cold objects such as comets, planets, and star-forming regions could be related to time observations.\nd. Semantic meaning: Considering the context, the other candidate entities such as the software collection (Entity 1) and the time zone (Entity 2 and 4) do not align as strongly with the context as the Large Millimeter Telescope.\nTherefore, the most suitable entity for the mention \"GMT\" is:\n3. Large Millimeter Telescope.\n\n"
    content += "The Second Inferences: Greenwich Mean Time (GMT) is the most likely entity when people refer to \"GMT\".\n\n"
    content += "Answer: The more plausible inference between the two would be \"The Second Inferences.\" The reasoning for this choice is that the second inference, based on common sense, aligns with the widely known and accepted meaning of \"GMT\" as Greenwich Mean Time. While the first inference method attempts to use contextual clues, the conclusion drawn is less likely to be accurate due to the noise in the context and the lack of specific information about the mention \"GMT.\" Therefore, the second inference, based on common knowledge and understanding of the term \"GMT,\" is more likely to be correct.\n\n"
    # content += "I am currently working on an entity disambiguation task that involves identifying the correct entity from a given list based on a mention within a specific context. The task is challenging due to potential noise in the context and limitations in common sense reasoning when information is scarce. I have two potential inference methods to resolve the mention to its correct entity:\n\nThe First Inference: This method relies on the immediate textual context in which the mention appears. It considers the surrounding words and sentences to infer the meaning. However, there might be noise—irrelevant or misleading information—in the context that could lead to incorrect conclusions.\n\nThe Second Inference: This method uses general knowledge and reasoning about the world to identify the entity. It does not solely rely on the given context but utilizes what is typically known or assumed about the mention. The drawback here is the possibility of not having enough information to make a reliable inference.\n\nHere are the two inferences about \"{}\":\n\n".format(src_dict['mention'])
    
    if context_flag:
        content += "Now I'll give you a mention, a context, and a list of candidates entities(random order), the mention will be highlighted with '###' in context. And the two inferences will infer the entity that best fits the mention:\n\n"
        
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

    content += "Here are the two inferences about mention \"{}\"\n\n".format(src_dict['mention'])
    # content += "Now, I'll give you two inferences about mention \"{}\"\n\n".format(src_dict['mention'])
    content += 'The First Inferences: {}\n\n'.format(first_reason)
    content += 'The Second Inferences: {}\n\n'.format(second_reason)

    content += 'Let‘s think step by step. Please choose the more plausible inference between them, answer their numbering like "The First Inferences" or "The Second Inferences".'    # handcarft
    # content += "Let‘s think step by step. To better understand which method is more likely to yield accurate and reliable results, I need your assistance. Could you please choose the more plausible inference between them, answer their numbering like \"The First Inferences\" or \"The Second Inferences\"."

    messages = [
    {
        "role": "system",
        "content": system_content,
    },
    {"role": "user", "content": content}
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

def prompt_formula(src_dict, cot_dict, cot_index_dict, cot_cand, topk, prompt_id):
    # system_content = "You're an entity disambiguator. I'll give you some tips on entity disambiguation, you should pay attention to these textual features:\n\n"
    # system_content = "You're an entity disambiguator. I'll give you the description of entity disambiguation and some tips on entity disambiguation, you should pay attention to these textual features:\n\n"
    system_content = "You're an entity disambiguator."
    # system_content += instruction_dict[prompt_id]['prompt']
    # content += '\n\n'

    '''only category'''
    # cot_case_list = cot_dict.get(src_dict['llm_category'], [])
    # if len(cot_case_list) > 0:
    #     cot_case = random.sample(cot_case_list, 1)[0]
    # else:
    #     cot_case = random.sample(cot_global_list, 1)[0]
    '''category and sentence sim'''
    # cot_index_list = cot_cand[:topk]
    # cot_index = random.sample(cot_index_list, 1)[0]
    # # print(cot_index)
    # cot_case = cot_index_dict[cot_index]
    '''random'''
    # cot_case = random.sample(list(cot_index_dict.values()), 1)[0]
    
    # content = 'The following example will help you understand the task:\n\n'
    # content += cot_case

    content = "Now, I'll give you a mention, a context, and a list of candidates entities, the mention will be highlighted with '###' in context.\n\n"
    content += 'Mention:{}\n'.format(src_dict['mention'])

    context = src_dict['left_context'] + ' ###' + src_dict['mention'] + '### ' + src_dict['right_context']
    context = context.strip()
    context = ' '.join(context.split())
    content += 'Context:{}\n'.format(context)

    candidates = random.sample(src_dict['candidates'], len(src_dict['candidates']))
    i = 1
    for cand in candidates:
        cand_entity = '{}.{}'.format(cand['name'], cand['summary'])
        content += 'Entity {}:{}\n'.format(i, cand_entity)
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
prompt_id = 1
'''listwise or prior'''
# dataset = read_json(dataset_path + 'datasets_recall/listwise_input/{}_sum13B_13B_with_c.jsonl'.format(dataset_name))
# dataset = read_json('/data/xkliu/EL_datasets/datasets_recall/msnbc_test_13B.jsonl')
# output_f = open(dataset_path + 'result/zephyr/baseline/msnbc_test_13B.jsonl'.format(dataset_name), 'w')
'''judge'''
# dataset = read_json(dataset_path + 'result/zephyr/{}_sum13B_13B_prompt{}_5top1.jsonl'.format(dataset_name, prompt_id))
# output_f = open(dataset_path + 'result/zephyr/{}_sum13B_13B_prompt{}_10top1.jsonl'.format(dataset_name, prompt_id), 'w')
'''merge'''
dataset = read_json(dataset_path + 'result/zephyr/merge/filter/{}.jsonl'.format(dataset_name))
output_f = open(dataset_path + 'result/zephyr/merge/final/{}_sum13B_13B_nocot_noprompt.jsonl'.format(dataset_name), 'w')

output_key = 'llm_merge'
max_new_token = 1024
instruction_dict = read_prompt('/data/xkliu/EL_code/LLM4EL/prompt/prompt.jsonl')
cot_dict, cot_index_dict = read_cot('/data/xkliu/EL_datasets/COT_sample/final/aida_train_merge_listwise_repeated.jsonl')
dataset_cand_list = read_cot_cand('/data/xkliu/EL_datasets/embedding/{}_cot.jsonl'.format(dataset_name))
topk = 1

i = 1
for src_dict, cot_cand in tqdm(zip(dataset, dataset_cand_list)):
    messages = prompt_formula(src_dict, cot_dict, cot_index_dict, cot_cand, topk, prompt_id)  # listwise
    # messages = prompt_formula_judge(src_dict, 'llama_predict')                                     # judge
    # messages = prompt_formula_prior(src_dict)                                        # prior
    # messages = prompt_formula_merge(src_dict, 'llm_predict_prompt0', 'llm_prior', False)

    if len(src_dict['candidates']) == 1:
        src_dict[output_key] = 'The prior and context are the same. Predict entity is {}.'.format(src_dict['candidates'][0]['name'])
        output_f.write(json.dumps(src_dict, ensure_ascii=False) + '\n')
    else:
        # print(json.dumps(messages))
        # print('-------------system-content-------------------')
        # print(messages[0]['content'])
        # print('-----------------content-------------------')
        # print(messages[1]['content'])
        # if i >= 10:
        #     exit()
        # i += 1
        try:
            prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            outputs = pipe(prompt, max_new_tokens=max_new_token, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
            gen_text = outputs[0]["generated_text"]
            gen_start_pos = gen_text.find('<|assistant|>')  # zephyr
            gen_text = gen_text[gen_start_pos:]             # zephyr
            # print(gen_text)
            src_dict[output_key] = gen_text
            # exit()
            output_f.write(json.dumps(src_dict, ensure_ascii=False) + '\n')
        except:
            src_dict[output_key] = ''
            output_f.write(json.dumps(src_dict, ensure_ascii=False) + '\n')
    
