from transformers import LlamaForCausalLM, LlamaTokenizer
import pickle
from tqdm import tqdm
import json

dataset_path = '/data/xkliu/EL_datasets/'
tokenizer = LlamaTokenizer.from_pretrained("/data/xkliu/LLMs/models/llama2-7b-chat-hf")
# model = LlamaForCausalLM.from_pretrained("/output/path")

# print(len(tokenizer('hello')['input_ids']))
# des_dict = pickle.load(open(dataset_path + 'kg_src/descriptions_dict.pickle', 'rb'), encoding='utf-8')
des_with_len_dict = {}
# for wiki_id, des in tqdm(des_dict.items()):
#     try:
#         des_with_len_dict[wiki_id] = {
#             'text':des,
#             'tokens':len(tokenizer(des)['input_ids'])
#         }
#     except:
#         des_with_len_dict[wiki_id] = {
#             'text':des,
#             'tokens':-1
#         }
def bag_freq(len_dict:dict, tag):
    # Token: 0-256, 256-512, 512-1024, 1048-2048, 2048-4096, >4096
    # counts: 1, 2-3, 4-6, 6-10, 10-50, 50-100, >100
    if tag == 'tokens':
        bag_dict = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}
        for tokens, count in len_dict.items():
            if tokens == 0:
                bag_dict[0] = bag_dict.get(6, 0) + count
            if tokens <= 16:
                bag_dict[0] = bag_dict.get(0, 0) + count
            elif tokens <= 64:
                bag_dict[1] = bag_dict.get(1, 0) + count
            elif tokens <= 128:
                bag_dict[2] = bag_dict.get(2, 0) + count
            elif tokens <= 256:
                bag_dict[3] = bag_dict.get(3, 0) + count
            elif tokens <= 512:
                bag_dict[4] = bag_dict.get(4, 0) + count
            else:
                bag_dict[5] = bag_dict.get(5, 0) + count
    elif tag == 'counts':
        bag_dict = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}
        for tokens, count in len_dict.items():
            if tokens <= 1:
                bag_dict[0] = bag_dict.get(0, 0) + count
            elif tokens <= 3:
                bag_dict[1] = bag_dict.get(1, 0) + count
            elif tokens <= 6:
                bag_dict[2] = bag_dict.get(2, 0) + count
            elif tokens <= 10:
                bag_dict[3] = bag_dict.get(3, 0) + count
            elif tokens <= 50:
                bag_dict[4] = bag_dict.get(4, 0) + count
            elif tokens <= 100:
                bag_dict[5] = bag_dict.get(5, 0) + count
            else:
                bag_dict[6] = bag_dict.get(6, 0) + count
    
        if bag_dict.get(6, 0) == 0:
            bag_dict[6] = 0
    return bag_dict

def draw_pie(data_dict:dict, save_path, title, labels):
    data_sorted = dict(sorted(data_dict.items(), key = lambda kv:(kv[0], kv[1])))
    import matplotlib.pyplot as plt
    plt.clf()
    plt.pie(data_sorted.values(), labels=labels, autopct='%1.1f%%')
    plt.title(title)
    plt.savefig(save_path)

output_f = open(dataset_path + 'kg_src/recall_entity_summarize_7B_with_tokens.json', 'w')

def dataset_token(filename):
    token_dict = {}
    with open(filename) as input_f:
        for line in tqdm(input_f):
            line = json.loads(line.strip())
            # context = line['left_context'] + ' ###' + line['mention'] + '### ' + line['right_context']
            # context = context.strip()
            # context = ' '.join(context.split())
            context = line['src']['summary']
            tokens = len(tokenizer(context)['input_ids'])
            line['sum_tokens'] = tokens
            output_f.write(json.dumps(line, ensure_ascii=False) + '\n')
            token_dict[tokens] = token_dict.get(tokens, 0) + 1
        
    return token_dict

# datasets = ['ace2004', 'aida', 'aquaint', 'cweb', 'msnbc', 'wiki']
# for dataset in datasets:
#     token_dict = dataset_token(dataset_path + '{}_test.jsonl'.format(dataset))
#     kg_count_bag = bag_freq(token_dict, 'tokens')
#     print(kg_count_bag)
#     draw_pie(kg_count_bag, dataset_path + 'pic/{}_context_tokens.png'.format(dataset),
#             '{}_context_tokens'.format(dataset), ['0-256', '256-512', '512-1024', '1048-2048', '2048-4096', '>4096'])

token_dict = dataset_token(dataset_path + 'kg_src/recall_entity_summarize_7B.json')
# kg_count_bag = bag_freq(token_dict, 'tokens')
# print(kg_count_bag)
# draw_pie(kg_count_bag, dataset_path + 'pic/kg_origin_context_tokens.png',
#         'kg_origin_context_tokens', ['1-16', '16-64', '64-128', '128-256', '256-512', '>512', '0'])

# for k, v in des_with_len_dict.items():
#     print(k ,v)

# with open(dataset_path + 'kg_src/descriptions_with_len_dict.pickle', 'wb') as output_f:
#     pickle.dump(des_with_len_dict, output_f)
