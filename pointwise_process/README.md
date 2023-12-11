# 代码说明
此部分是对pointwise的处理

**cand_trunc.py** 处理部分token超过推理长度的list，做法是将候选实体进行对半分。目前还没有决定是否采用这种方法。
**category_decode.py** 对LLM生成的类别进行类别解析，解析的方法是出现给定类型出现的最后的位置作为类别。
**flatten.py** 将listwise的数据集进行拉平处理，作为pointwise的处理。
**listwise_cand.py** 将pointwise的结果处理，过滤掉被LLM判断不是的结果，最后转成listwise input。注意需要no_cand的输入，从而保证和原始数据集的条数一样
**no_cand.py** 从原始listwise的数据集中，过滤出所有不包含candidates的数据条目。