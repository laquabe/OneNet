# 代码说明
数据处理的代码

**add_ans_id.py** 为entity填上wikiid，方便做处理
**recall_mention.py**  筛选被mention召回的entity，这里是对基础的KGbase进行处理，否则数量太大，无法跑起来。
**retriver_dataset.py** 对candidate进行处理，这里包含一些截断处理，截断的逻辑