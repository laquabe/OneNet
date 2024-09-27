import json
from tqdm import tqdm
import re
import argparse

def pointwise_eval(eval_file_name):
    cands_len = []
    hit_num = 0
    with open(eval_file_name) as input_f:
        for line in tqdm(input_f):
            line = json.loads(line.strip())
            label_id = line['label_document_id']
            cand_set = set()
            for cand in line['candidates']:
                cand_set.add(cand['document_id'])
            if label_id in cand_set:
                hit_num += 1
            cands_len.append(len(line['candidates']))

        print('Recall: {:.4f}'.format(hit_num / 10000))
        print('Avg Cands: {:.4f}'.format(sum(cands_len) / len(cands_len)))

def listwise_eval(eval_file_name, eval_key, normalize=True, normal_set=None):
    '''eval for ZeShEL'''
    all_num ,hit_num, pred_num = 0, 0, 0
    all_num_dict, hit_num_dict = {}, {}
    with open(eval_file_name) as input_f:
        for line in tqdm(input_f):
            line = json.loads(line.strip())
            label_id = line['label_document_id']
            pred_id = line[eval_key]
            if (len(line['candidates']) != 0) and (pred_id != -1):
                pred_num += 1
            if label_id == pred_id:
                hit_num += 1
                hit_num_dict[line['corpus']] = hit_num_dict.get(line['corpus'], 0) + 1
            if normalize:
                if line['mention_id'] in normal_set:
                    all_num += 1
                    all_num_dict[line['corpus']] = all_num_dict.get(line['corpus'], 0) + 1
            else:
                all_num += 1
                all_num_dict[line['corpus']] = all_num_dict.get(line['corpus'], 0) + 1
    

    for k, v in hit_num_dict.items():
        if normalize:
            print('{} N.Acc: {}'.format(k, v / all_num_dict[k]))
        else:
            print('{} U.Acc: {}'.format(k, v / all_num_dict[k]))

def original_recall(src_file_name, cand_file_name):
    hit_set = set()
    with open(src_file_name) as src_f, \
        open(cand_file_name) as cand_f:
        hit_num = 0
        id2label = {}
        for line in tqdm(src_f):
            line = json.loads(line.strip())
            id2label[line['mention_id']] = line['label_document_id']

        for line in tqdm(cand_f):
            line = json.loads(line.strip())
            label_id = id2label[line['mention_id']]
            if label_id in line['tfidf_candidates']:
                hit_num += 1
                hit_set.add(line['mention_id'])
    
    return hit_set

def result_decode(llm_pred:str, map_dict):
    llm_pred = llm_pred.lstrip('<|assistant|>\n')
    llm_pred = llm_pred.lower()
    '''llm result decode
        1. decode json like {'entity_name'}
        2. if fail, use the last entity name in sentence
    '''
    decode_flag = False
    pred_entity = None
    json_str = re.search(r'{.*}', llm_pred)
    if json_str:
        try:
            json_str = json.loads(json_str.group())
            if 'name of the entity' in json_str.keys():
                pred_entity = json_str['name of the entity']
            else:
                for v in json_str.values():
                    if not str(v).isdigit():
                        pred_entity = v
            if pred_entity != None:
                decode_flag = True
        except:
            pass
    if decode_flag:
        '''check name valid'''
        try:
            if pred_entity in map_dict.keys():
                return pred_entity
            else:
                decode_flag = False
        except:
            decode_flag = False

    if not decode_flag:
        index_dict = {}
        for name in map_dict.keys():
            if name in llm_pred:
                pos = llm_pred.find(name) - len(name)
                index_dict[name] = pos
    
        if len(index_dict) > 0:
            pred_entity, _ = sorted(index_dict.items(), key = lambda kv:(kv[1], len(kv[0]), kv[0]), reverse=False)[0]
        else:
            pred_entity = 'none'
    
    return pred_entity

def file_f1(file_name, test_field):
    all_num = 0
    hit_num = 0
    error_num = 0
    with open(file_name) as input_f:
        for line in input_f:
            all_num += 1
            line = json.loads(line)

            if len(line[test_field]) == 0:
                error_num += 1
                continue

            map_dict = {'none':4096}
            for cand in line['candidates']:
                map_dict[str(cand['name']).lower()] = int(cand['wiki_id'])

            pred_entity = result_decode(line[test_field], map_dict)
            pred_num = map_dict[pred_entity]

            if pred_num == line['ans_id']:
                hit_num += 1
            
    print(all_num, hit_num, hit_num / all_num ,error_num)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DocFixQA args')
    parser.add_argument('--dataset_name', '-d', type=str, required=True, help="Dataset Name for Test")
    parser.add_argument('--file_path','-f',type=str, required=True, help="Path to Eval File")
    parser.add_argument('--answer_key','-k',type=str, required=True, help="Answer Key in File")
    parser.add_argument('--normalize', action='store_true', help="if ZeShEL report normalization result", default=False)

    args = parser.parse_args()
    dataset_name = args.dataset_name
    dataset_name = dataset_name.lower()
    file_path = args.file_path  # zeshel: final_pred_num; other: llm_merge
    answer_key = args.answer_key
    normalize = args.normalize

    # pointwise_eval('tfidf_candidates/test.json')
    if dataset_name == 'zeshel' and normalize:
        hit_set = original_recall('datasets/zeshel/mentions/test.json', 'datasets/zeshel/tfidf_candidates/test.json')
    else:
        hit_set = None
    
    if dataset_name == 'zeshel':
        listwise_eval(file_path, answer_key, normal_set=hit_set, normalize=normalize)
    else:
        file_f1(file_path, answer_key)