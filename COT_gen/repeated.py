import json

input_f = open('/data/xkliu/EL_datasets/COT_sample/final/aida_train_merge_listwise_with_c.jsonl')
output_f = open('/data/xkliu/EL_datasets/COT_sample/final/aida_train_merge_listwise_repeated.jsonl', 'w')

mention_set = set()

for line in input_f:
    j_line = json.loads(line.strip())
    mention = j_line['mention']
    if mention in mention_set:
        continue
    mention_set.add(mention)
    output_f.write(line)
    

