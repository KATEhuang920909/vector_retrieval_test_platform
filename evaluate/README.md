# 评价指标
采用MRR来评价模型的检索效果

![[公式]](https://www.zhihu.com/equation?tex=%7B%5Crm+MRR%7D+%3D+%5Cfrac%7B1%7D%7BS%7D%5Csum_%7Bi%3D1%7D%5E%7BS%7D%5Cfrac%7B1%7D%7Bp_i%7D)


其中Q代表所有测试集，rank_i代表第i条测试query对应的相关doc在搜索系统返回中的位置。

如：对于第一条query的相关doc在检索系统中排在第一位，该测试query的MRR值为1；排在第二位，则MRR值为0.5，最终指标为全部测试query MRR值的平均数。


可采用MRR@K作为最终评测指标，即如果测试query相关doc不在top k，则MRR值为0,否则MRR值为1。

```text
recall@1：召回的top1为正确结果
recall@50：召回的top50为正确结果
recall@all：召回的topk为正确结果
```
