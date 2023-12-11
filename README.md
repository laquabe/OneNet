# LLM EL 流程
本仓库是将EL用LLM处理的代码仓库
## 数据处理
    目的：一些数据处理，主要在data_process中，方便后续处理
    流程：
        参考data_process中，加id等操作
## 描述化简
    目的：用LLM对实体描述进行总结，本质上是用LLM做summary，以减少后面实体的长度。
    流程：
        0. data_process/recall_mention.py 筛选有召回的实体，减少运行量（大概从500W -> 20W）
        1. 运行 LLM/llama/batch_sum.py 用LLM进行总结。
## mention分类
    目的：为mention进行类别
    流程：
        1. 运行LLM/llama/category.py。用LLM进行分类
        2. 运行pointwise_process/category_decode.py。对LLM的输出进行解析
## pointwise EL
    目的：先用pointwise EL，对候选实体进行粗筛。
    流程：
        0. 对于部分候选实体过多的数据集，先用热度进行筛选，保证在可控时间内运行。具体而言，运行data_process/retriver_dataset.py。目前部分数据集取的top10
        1. 运行pointwise_process/flatten.py，将listwise数据集转为pointwise数据集。
        2. 运行LLM/llama/pointwise_el.py，进行pointwise EL粗筛选
        （在listwise 中发现有部分推荐太长，这里应该要加一步后处理）
## COT生成
    目的：在COT_gen中，这部分通过GPT3.5和GPT4进行生成。
## listwise EL
    目的：最后的listwise EL，输出最后的结果
    流程：
        0. 运行pointwise_process/no_cand.py，筛出原本没有candidate的数据，方便数据对齐
        1. 运行pointwise_process/listwise_cand.py，将pointwise数据集转化为listwise数据集，并和nocand部分合并
        2. 运行LLM/hf/listwise.py，完成最后的预测