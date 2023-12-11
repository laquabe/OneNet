import json
from tqdm import tqdm
dataset_name = 'cweb'
with open('{}_test_13B_top10g.jsonl'.format(dataset_name)) as input_f, \
    open('flatten/{}_test_13B_top10g.jsonl'.format(dataset_name), 'w') as output_f:
    i = 0
    for line in tqdm(input_f):
        line = json.loads(line.strip())
        if len(line['candidates']) > 10:
            print(line)
        for can in line['candidates']:
            new_line = {}
            new_line['left_context'] = line['left_context']
            new_line['mention'] = line['mention']
            new_line['right_context'] = line['right_context']
            new_line['output'] = line['output']
            new_line['ans_id'] = line['ans_id']
            new_line['id'] = i
            new_line['cand_name'] = can['name']
            new_line['cand_summary'] = can['summary']
            new_line['cand_text'] = can['text']
            new_line['cand_wiki_id'] = can['wiki_id']
            output_f.write(json.dumps(new_line, ensure_ascii=False) + '\n')
        i += 1