# 代码说明
此部分是一些数据分析代码，包括数据集的数据特征、中间结果测评、最终结果测评等。

**llama_tokenizer.py** 对输入的文本token数目进行估计.
**name_cacu.py** 统计数据集中，是否存在同一段落，menion完全一样但是所指实体不一样的
**retriver_pic.py** 数据集的一些数据特征统计。
**recall_static.py** 做pointwise以后，召回剩下的比例，是最后listwise的上限。
