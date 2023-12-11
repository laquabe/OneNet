# 代码说明
此部分是利用llama2进行推理的代码，主要是prompt书写问题

**batch_sum.py** 对entity的描述进行总结，主要是为了后续的token长度问题。
**category.py** 对mention进行分类，类别来自wiki的最大类别。
**listwise_el.py** listwise el，但token取4096时有bug，暂且不用这份推理代码
**pointwise_el.py** pointwise el, 对单独entity去做粗筛
