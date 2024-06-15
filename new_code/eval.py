import json
from tqdm import tqdm
import os
import copy
import re

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
    all_num = 0
    hit_num = 0
    pred_num = 0
    with open(eval_file_name) as input_f:
        for line in tqdm(input_f):
            line = json.loads(line.strip())
            label_id = line['label_document_id']
            pred_id = line[eval_key]
            if (len(line['candidates']) != 0) and (pred_id != -1):
                pred_num += 1
            if label_id == pred_id:
                hit_num += 1
            if normalize:
                if line['mention_id'] in normal_set:
                    all_num += 1
            else:
                all_num += 1
    
    print(all_num, hit_num, hit_num / all_num)
    print(pred_num, hit_num, hit_num / pred_num)
        

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

def summary_loss(input_file_name):
    loss_num = 0
    all_num = 0
    with open(input_file_name) as input_f:
        for line in tqdm(input_f):
            line = json.loads(line.strip())
            label_id = line['label_document_id']
            cands = line['candidates']
            for cand in cands:
                if cand['document_id'] == label_id:
                    if len(cand['summary']) == 0:
                        loss_num += 1
                    all_num += 1
    print(loss_num, all_num)

if __name__ == '__main__':
    # pointwise_eval('tfidf_candidates/test.json')
    # summary_loss('/data/xkliu/EL_datasets/zeshel/mentions/listwise_merge.json')
    hit_set = original_recall('mentions/test.json', 'tfidf_candidates/test.json')
    listwise_eval('process_data/result_llama3_raw_summary.json', 'context_pred_num', normal_set=hit_set)