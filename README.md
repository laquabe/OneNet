# OneNet
Repository for [OneNet: A Fine-Tuning Free Framework for Few-Shot Entity Linking via Large Language Model Prompting.](https://openreview.net/forum?id=GCxvf6qxeW) @ EMNLP 2024, main. [[arxiv](https://arxiv.org/abs/2410.07549)]

## Set Up

 Any environment with a reasonable Huggingface Transformers installation should be fine. You can also refer to the environments we provided in `requirements.txt`.

## Datasets
For Wiki-based data, please visit [NER4EL](https://github.com/Babelscape/ner4el/tree/master)

For ZeShEL, please visit [ZeShEL](https://github.com/lajanugen/zeshel)

Since OneNet is a multi-step pipeline, some steps may be time-consuming. Therefore, we also provide some intermediate results (e.g. summaries, filtered entities), please visit this [link](https://drive.google.com/file/d/1rHX7HgWdkwnSPf6p68FHf8_s_6OBrGRK/view?usp=sharing).

## Usage

To start, you can just run the `run.sh`
```
bash run.sh
```

`run.sh` is the script that calls the LLM inference in `new_code/prompt`, you just need to change *func_name*, you can run the inference of different steps in OneNet, the *func_name* description is as follows

| func_name | file format | Description |
|------------|------|-------------|
|summary | summary.json| Get summary of enitities|
|point_wise | pointwise_raw.json | Filter the irrelevant entities |
|category | listwise_filter.json | get the category of mention|
|context | listwise_filter.json | entity linking based on context |
|prior | listwise_filter.json | entity linking based on prior |
|merge | listwise_merge.json | merge EL result from context and prior |

We have organized most of the core code into `new_code\`. Specifically, `prompt.py` contains the prompts for all modules, while `data_process.py` contains most of the data processing code. Other details can also be found in other folders

## Evaluation

Please run `eval.sh`. 
```
bash eval.sh
```
We also provide the result file in the datasets, please visit [link](https://drive.google.com/file/d/1rHX7HgWdkwnSPf6p68FHf8_s_6OBrGRK/view?usp=sharing).


## Citation

If you find our work interesting/helpful, please consider citing OneNet
