# -*- coding: utf-8 -*-
# @Time    : 2022/3/20 2:26
# @Author  : huangkai
# @File    : data_preprocess.py
import random

import jieba
import re
from tqdm import tqdm_notebook
import pandas as pd

filters = "，。/；‘【】、.·~！……（）——{}|：“”《》？"
corpus = pd.read_csv(r'../data/corpus.tsv', sep='\t', header=None)
qrels = pd.read_csv(r'compet_data/qrels.train.tsv', sep='\t', header=None)
with open(r'compet_data/train.query.txt') as f:
    train = f.readlines()
    train = [k.strip().split("\t") for k in train]
    train = pd.DataFrame(train)

train[3] = train[3].apply(lambda x: '' if pd.isna(x) else x)
train[2] = train[2].apply(lambda x: '' if pd.isna(x) else x)
train[0] = train[0].apply(lambda x: int(x))
train[1] = train[1] + train[2] + train[3]
train = train[[0, 1]]
corpus.columns = [1, "document"]
train.columns = [0, "query"]

# pos data
data_pos = pd.merge(pd.merge(qrels, train, on=0, how="left"), corpus)

# neg_data
corpus_match_index = data_pos[1].values
data_document_neg = corpus[corpus[1].apply(lambda x: True if x not in corpus_match_index else False)]

data_pos_query_token = data_pos["query"].apply(lambda x: jieba.lcut(re.sub(filters, '', x))).reset_index(drop=True)
data_neg_document_token = data_document_neg["document"].apply(lambda x: jieba.lcut(re.sub(filters, '', x))).reset_index(
    drop=True)

# create neg data
count = 0
data_neg = pd.DataFrame([])
for i in tqdm_notebook(range(len(data_pos_query_token))):
    for j in range(random.randint(0, len(data_neg_document_token)), len(data_neg_document_token)):
        if not set(data_pos_query_token[i]) <= set(data_neg_document_token[j]):
            df = pd.DataFrame([data_pos["query"].iloc[i], data_document_neg["document"].iloc[j]]).T
            data_neg = data_neg.append(df)
            if count == 1:
                count = 0
                break
            count += 1

data_neg["label"] = 0
data_neg = data_neg[[0, 1, "label"]]
data_neg.columns = ["query", "document", "label"]
data_pos = data_pos[["query", "document"]]
data_pos["label"] = 1

data=pd.concat([data_pos,data_neg]).sample(frac=1)
train = data[:int(0.9*(data.shape[0]))].reset_index(drop=True)
dev = data[int(0.9*(data.shape[0])):].reset_index(drop=True)
train.to_csv(r'compet_data/train.csv')
dev.to_csv(r'compet_data/dev.csv')
