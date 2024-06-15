from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import torch
# device = "cuda" the device to load the model onto


def load_llm(model_name, model_path):
    if model_name == 'Mistral':
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto')
        tokenizer = AutoTokenizer.from_pretrained(model_path, device_map='auto')
        return model, tokenizer
    elif model_name == 'Llama':
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_path,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map='auto',
            )
        return pipeline
    elif model_name == 'Zephyr':
        pipe = transformers.pipeline("text-generation", model=model_path, torch_dtype=torch.bfloat16, device_map="auto")
        return pipe
    else:
        print('Error! No support models')

def llm_call(messages, model_name, model=None, tokenizer=None, pipeline=None, do_sample=False):
    if model_name == 'Mistral':
        model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")

        generated_ids = model.generate(model_inputs, max_new_tokens=1024, do_sample=do_sample)
        decoded = tokenizer.batch_decode(generated_ids)
        
        return decoded[0]   # include input, need extra process
    elif model_name == 'Llama':
        prompt = pipeline.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        terminators = [
            pipeline.tokenizer.eos_token_id,
            pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        if do_sample:
            outputs = pipeline(
                prompt,
                max_new_tokens=1024,
                eos_token_id=terminators,
                do_sample=do_sample,
                temperature=0.6,
                top_p=0.9,
            )
        else:
            outputs = pipeline(
                prompt,
                max_new_tokens=1024,
                eos_token_id=terminators,
                do_sample=do_sample,
            )

        return outputs[0]["generated_text"][len(prompt):]
    elif model_name == 'Zephyr':
        prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if do_sample:
            outputs = pipeline(prompt, max_new_tokens=1024, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        else:
            outputs = pipeline(prompt, max_new_tokens=1024, do_sample=True)

        gen_text = outputs[0]["generated_text"]
        gen_start_pos = gen_text.rfind('<|assistant|>')  # zephyr
        gen_text = gen_text[gen_start_pos:]
        gen_text = gen_text.lstrip('<|assistant|>').strip()
        
        return gen_text
    else:
        print('Error! No models use')

if __name__ == "__main__":
    messages = [
        {"role": "user", "content": "What is your favourite condiment?"},
        {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
        {"role": "user", "content": "Do you have mayonnaise recipes?"}
    ]
    model_name = 'Zephyr'
    '''Model & Tokenizer'''
    # model, tokenizer = load_llm(model_name, '')
    # response = llm_call(messages, model_name, model=model, tokenizer=tokenizer)
    # print(response)
    '''Pipeline'''
    pipeline = load_llm(model_name, '/data/xkliu/LLMs/models/zephyr-7b-beta')
    response = llm_call(messages, model_name, pipeline=pipeline)
    print(response)
