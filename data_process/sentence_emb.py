import os
os.environ['CUDA_VISIBLE_DEVICES']='4'
import torch
torch.set_num_threads(10)
import numpy as np
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('/data/xkliu/hf_models/all-mpnet-base-v2')
sentence_file_name = 'aida_train_merge'
file_path = '/data/xkliu/EL_datasets/COT_sample/final/{}_listwise_repeated.jsonl'.format(sentence_file_name)

emb_list = []
with open(file_path) as input_f:
    for line in tqdm(input_f):
        src_dict = json.loads(line.strip())
        context = src_dict['left_context'] + src_dict['mention'] + src_dict['right_context']
        context = context.strip()
        context = ' '.join(context.split())
        emb = model.encode([context])[0]
        emb_list.append(emb)

emb_list = np.array(emb_list)
print(emb_list.shape)
np.save('/data/xkliu/EL_datasets/embedding/{}.npy'.format(sentence_file_name), emb_list)

