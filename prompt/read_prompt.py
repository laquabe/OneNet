import json

with open('prompt.jsonl') as prompt_f:
    for line in prompt_f:
        line = json.loads(line.strip())
        print('-------------prompt-{}------------'.format(line['id']))
        print(line['prompt'])
        
