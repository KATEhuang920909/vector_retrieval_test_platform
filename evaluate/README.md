# 评价指标
采用MRR来评价模型的检索效果

![](./README/8b5c04a3c63a4eae80aa3a8d9207c171.png)


其中Q代表所有测试集，rank_i代表第i条测试query对应的相关doc在搜索系统返回中的位置。

如：对于第一条query的相关doc在检索系统中排在第一位，该测试query的MRR值为1；排在第二位，则MRR值为0.5，最终指标为全部测试query MRR值的平均数。


可采用MRR@K作为最终评测指标，即如果测试query相关doc不在top k，则MRR值为0,否则MRR值为1。


