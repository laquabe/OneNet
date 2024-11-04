export CUDA_VISIBLE_DEVICES=6
python new_code/prompt.py \
    --dataset_name ace2004 \
    --dataset_path datasets/ace2004/listwise_merge.jsonl \
    --model_name Llama \
    --model_path \
    --func_name merge \
    --exp_name  test \
    --test
