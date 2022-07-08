# faiss

## 1.Flat

暴力检索

* faiss无法计算余弦相似度，实现方法如下：

```python
# 训练阶段
faiss.normalize_L2(train_embedding)
index.train(train_embedding)
# search阶段
faiss.normalize_L2(embedding_for_querying)
d, i = faiss_index.search(embedding_for_querying, ann_count)
```

