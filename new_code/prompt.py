from LLM_calls import load_llm, llm_call
import json
from tqdm import tqdm
import random

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
    cot_index_dict = {}
    i = 0
    with open(cot_file_name) as input_f:
        for line in tqdm(input_f):
            line = json.loads(line.strip())
            prompt = list_prompt_formula(line)
            prompt += 'Answer: {}\n\n'.format(line['gpt_ans'].replace('\n',''))
            
            cot_index_dict[i] = prompt
            i += 1

    return cot_index_dict


def summary_prompt(src_dict):
    content = 'The following is a description of {}. Please extract the key information of {} and summarize it in one sentence.\n\n{}'.format(
    src_dict['title'], src_dict['title'], src_dict['text'])

    return content

def category_prompt(src_dict):
    category_list = ['General reference','Culture and the arts','Geography and places','Health and fitness', 'History and events', 'Human activities', 'Mathematics and logic', 'Natural and physical sciences', 'People and self', 'Philosophy and thinking', 'Religion and belief systems', 'Society and social sciences', 'Technology and applied sciences']
    content = 'You are a mention classifier. Wikipedia categorizes entity into the following categories:'
    for category in category_list:
        content += ' {},'.format(category)
    content = (content.rstrip(',') + '.\n')
    content += "Now, I will give you a mention and its context, the mention will be highlighted with '###'.\n\n"""

    content += 'mention:{}\n'.format(src_dict['mention'])
    if 'cut_left_context' in src_dict.keys():
        context = src_dict['cut_left_context'] + ' ###' + src_dict['mention'] + '### ' + src_dict['cut_right_context']
    else:
        context = src_dict['left_context'] + ' ###' + src_dict['mention'] + '### ' + src_dict['right_context']
    context = context.strip()
    context = ' '.join(context.split())
    content += 'context:{}\n\n'.format(context)

    content += "please determine which of the above categories the mention {} belongs to?".format(src_dict['mention'])
    return content

def point_wise_el_prompt(src_dict, cut=False):
    content = "You're an entity disambiguator. I'll give you the description of entity disambiguation and some tips on entity disambiguation, and you need to pay attention to these textual features:\n\n"
    content += instruction_dict[0]['prompt']
    content += '\n\n'
    content += "Now, I'll give you a mention, a context, and a candidate entity, and the mention will be highlighted with '###'.\n\n"
    content += 'Mention:{}\n'.format(src_dict['mention'])

    if cut:
        context = src_dict['cut_left_context'] + ' ###' + src_dict['mention'] + '### ' + src_dict['cut_right_context']
    else:
        context = src_dict['left_context'] + ' ###' + src_dict['mention'] + '### ' + src_dict['right_context']
    context = context.strip()
    context = ' '.join(context.split())
    content += 'Context:{}\n'.format(context)

    cand_entity = '{}.{}'.format(src_dict['candidates']['title'], src_dict['candidates']['summary'])
    content += 'Candidate Entity:{}\n\n'.format(cand_entity)

    content += """You need to determine if the mention and the candidate entity are related. Please refer to the above tips and give your reasons, and finally answer 'yes' or 'no'. Answer 'yes' when you think the information is insufficient or uncertain."""

    return content

def prior_prompt(src_dict, have_id=True):
    system_content = "You're an entity disambiguator."
    content = "I'll provide you a mention and its candidates below.\n\n"
    content += 'Mention:{}\n'.format(src_dict['mention'])

    candidates = random.sample(src_dict['candidates'], len(src_dict['candidates']))
    i = 1
    for cand in candidates:
        cand_entity = '{}.{}'.format(cand['title'], cand['summary'])
        if have_id:
            content += 'Entity {}:{}\n'.format(cand['document_id'], cand_entity)
        else:
            content += 'Entity {}:{}\n'.format(i, cand_entity)
        i += 1
    content += '\n'

    content += 'Based on your knowledge, please determine which is the most likely entity when people refer to mention "{}", '.format(src_dict['mention'])
    content += "and finally answer the id and name of the entity."

    return system_content, content

def context_prompt(src_dict, cot_index_dict, instruction_dict, prompt_id=0, have_id=True, ent_des='summary'):
    system_content = "You're an entity disambiguator. I'll give you the description of entity disambiguation and some tips on entity disambiguation, you should pay attention to these textual features:\n\n"
    system_content += instruction_dict[prompt_id]['prompt']

    '''category and sentence sim'''
    cot_index = src_dict['cot_index']
    cot_case = cot_index_dict[cot_index]
    
    content = 'The following example will help you understand the task:\n\n'
    content += cot_case

    content += "Now, I'll give you a mention, a context, and a list of candidates entities, the mention will be highlighted with '###' in context.\n\n"
    content += 'Mention:{}\n'.format(src_dict['mention'])

    if 'cut_left_context' in src_dict.keys():
        context = src_dict['cut_left_context'] + ' ###' + src_dict['mention'] + '### ' + src_dict['cut_right_context']
    else:
        context = src_dict['left_context'] + ' ###' + src_dict['mention'] + '### ' + src_dict['right_context']
    context = context.strip()
    context = ' '.join(context.split())
    content += 'Context:{}\n'.format(context)

    candidates = random.sample(src_dict['candidates'], len(src_dict['candidates']))
    i = 1
    for cand in candidates:
        cand_entity = '{}.{}'.format(cand['title'], cand[ent_des])
        if have_id:
            content += 'Entity {}:{}\n'.format(cand['document_id'], cand_entity)
        else:
            content += 'Entity {}:{}\n'.format(i, cand_entity)
        i += 1
    content += '\n'

    content += """You need to determine which candidate entity is more likely to be the mention. Please refer to the above example, give your reasons, and finally answer id of the entity and the name of the entity. If all candidate entities are not appropriate, you can answer '-1.None'."""

    return system_content, content

def merge_prompt(src_dict, instruction_dict, prompt_id=1, have_id=True):
    if len(src_dict['candidates']) == 1:
        content = 'The prior and context are the same. Predict entity is {}.'.format(src_dict['candidates'][0]['title'])
        return None, content
    
    system_content = "You're an entity disambiguator. I'll give you the description of entity disambiguation and some tips on entity disambiguation, you should pay attention to these textual features:\n\n"
    system_content += instruction_dict[prompt_id]['prompt']

    content = "Now, I'll give you a mention, a context, and a list of candidates entities, the mention will be highlighted with '###' in context.\n\n"
    content += 'Mention:{}\n'.format(src_dict['mention'])

    if 'cut_left_context' in src_dict.keys():
        context = src_dict['cut_left_context'] + ' ###' + src_dict['mention'] + '### ' + src_dict['cut_right_context']
    else:
        context = src_dict['left_context'] + ' ###' + src_dict['mention'] + '### ' + src_dict['right_context']
    context = context.strip()
    context = ' '.join(context.split())
    content += 'Context:{}\n'.format(context)

    candidates = random.sample(src_dict['candidates'], len(src_dict['candidates']))
    i = 1
    for cand in candidates:
        cand_entity = '{}.{}'.format(cand['title'], cand['summary'])
        if have_id:
            content += 'Entity {}:{}\n'.format(cand['document_id'], cand_entity)
        else:
            content += 'Entity {}:{}\n'.format(i, cand_entity)
        i += 1
    content += '\n'

    content += """You need to determine which candidate entity is more likely to be the mention. Please refer to the above example, give your reasons, and finally answer serial number of the entity and the name of the entity. If all candidate entities are not appropriate, you can answer '-1.None'."""

    return system_content, content

if __name__ == '__main__':
    model_name = 'Llama'
    pipeline = load_llm(model_name, '/data/xkliu/LLMs/models/Meta-Llama-3-8B-Instruct')
    # model_name = 'Zephyr'
    # pipeline = load_llm(model_name, '/data/xkliu/LLMs/models/zephyr-7b-beta')

    func_name = 'context'
    input_file_name = '/data/xkliu/EL_datasets/zeshel/mentions/split/listwise_raw_cot_3.json'
    output_file_name = '/data/xkliu/EL_datasets/zeshel/LLM_process/raw/summary/listwise_raw_cot_3.json'
    output_key = 'llm_response'
    full_flag = True
    
    instruction_dict = read_prompt('/data/xkliu/EL_code/LLM4EL/prompt/prompt.jsonl')
    cot_index_dict = read_cot('/data/xkliu/EL_datasets/COT_sample/final/aida_train_merge_listwise_repeated.jsonl')

    if not full_flag:
        with open(input_file_name) as input_f, \
            open(output_file_name, 'w') as output_f:
            for i, line in tqdm(enumerate(input_f)):
                line = json.loads(line.strip())
                if func_name == 'point_wise':
                    prompt = point_wise_el_prompt(line, cut=True)
                elif func_name == 'category':
                    prompt = category_prompt(line)
                elif func_name == 'context':
                    system_prompt, prompt = context_prompt(line, cot_index_dict=cot_index_dict, instruction_dict=instruction_dict, ent_des='summary')
                elif func_name == 'prior':
                    system_prompt, prompt = prior_prompt(line)
                elif func_name == 'merge':
                    system_prompt, prompt = merge_prompt(line, instruction_dict=instruction_dict)
                    if system_prompt == None:
                        response = prompt
                        line[output_key] = response
                        output_f.write(json.dumps(line, ensure_ascii=False) + '\n')
                        continue

                print('-'*50 + 'Prompt' + '-'*50)
                print(prompt)
                messages = [
                    {"role": "system", "content":system_prompt},
                    {"role": "user", "content": prompt},
                ]
                response = llm_call(messages, model_name, pipeline=pipeline)
                print('-'*50 + 'Response' + '-'*50)
                print(response)
                line[output_key] = response
                output_f.write(json.dumps(line, ensure_ascii=False) + '\n')
                if i >= 0:
                    break
    else:
        with open(input_file_name) as input_f, \
            open(output_file_name, 'w') as output_f:
            for line in input_f:
                line = json.loads(line.strip())
                if func_name == 'point_wise':
                    prompt = point_wise_el_prompt(line, cut=True)
                elif func_name == 'category':
                    prompt = category_prompt(line)
                elif func_name == 'context':
                    system_prompt, prompt = context_prompt(line, cot_index_dict=cot_index_dict, instruction_dict=instruction_dict)
                elif func_name == 'prior':
                    system_prompt, prompt = prior_prompt(line)
                elif func_name == 'merge':
                    system_prompt, prompt = merge_prompt(line, instruction_dict=instruction_dict)
                    if system_prompt == None:
                        response = prompt
                        line[output_key] = response
                        output_f.write(json.dumps(line, ensure_ascii=False) + '\n')
                        continue
                messages = [
                    {"role": "system", "content":system_prompt},
                    {"role": "user", "content": prompt},
                ]
                try:
                    response = llm_call(messages, model_name, pipeline=pipeline)
                except:
                    response = ''
                line[output_key] = response
                output_f.write(json.dumps(line, ensure_ascii=False) + '\n')
