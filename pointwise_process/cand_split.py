import json
from tqdm import tqdm
import copy
dataset = 'ace2004_test_noprompt_sum13B_13B'

max_cand_len = 5
def write_line(line, output_f):
    candidates = line['candidates']
    if len(candidates) > max_cand_len:
        split_point = int(len(candidates) / 2)
        cand_list_head = candidates[:split_point]
        cand_list_tail = candidates[split_point:]

        head_line = copy.deepcopy(line)
        head_line['candidates'] = cand_list_head
        write_line(head_line, output_f)

        tail_line = copy.deepcopy(line)
        tail_line['candidates'] = cand_list_tail
        write_line(tail_line, output_f)
    else:
        output_f.write(json.dumps(line, ensure_ascii=False) + '\n')



with open('{}_with_c.jsonl'.format(dataset)) as input_f ,\
    open('split/{}_stage1.jsonl'.format(dataset), 'w') as output_f:

    for line in tqdm(input_f):
        line = json.loads(line.strip())
        write_line(line, output_f)
