import json
from tqdm import tqdm
dataset_name = 'wiki'
sample_tag = True
if sample_tag:
    filename = '{}_test_13B_top10g.jsonl'.format(dataset_name)
else:
    filename = '{}_test_13B.jsonl'.format(dataset_name)

with open(filename) as input_f, \
    open('listwise_input/no_cand/{}_test_13B.jsonl'.format(dataset_name), 'w') as output_f:
    i = 0
    for line in tqdm(input_f):
        line = json.loads(line.strip())
        if len(line['candidates']) > 0:
            i += 1
            continue
        else:
            line['id'] = i
            i += 1
            output_f.write(json.dumps(line, ensure_ascii=False) + '\n')