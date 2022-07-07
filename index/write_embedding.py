# -*- coding: utf-8 -*-
# @Time    : 2022/4/17 17:19
# @Author  : huangkai
# @File    : write_embedding.py
import pandas as pd
from utils.get_embedding import PTM_Embedding
from hnsw_model import HNSW
import pickle
import numpy as np
import pprint
import sys

sys.path.append("../")


def load_data(path):
    with open(path, encoding="utf8") as f:
        train = f.readlines()
        train = [k.strip().split("\t") for k in train]
        train = pd.DataFrame(train)
    return train


def fit_vector(corpus, batch_size=256, ):
    temp_res = np.zeros((0, 256))
    i = 0
    loop_n = len(corpus) // batch_size
    while True:
        if i == loop_n:
            break
        corpus_vec = ptm_embedding.get_w2v_embedding(corpus[i * batch_size:(i + 1) * batch_size])
        print(corpus_vec.shape, temp_res.shape)
        temp_res = np.append(temp_res, corpus_vec, axis=0)
        i += 1
    if len(corpus) % batch_size > 0:
        corpus_vec = ptm_embedding.get_w2v_embedding(corpus[i * batch_size:(i + 1) * batch_size])
        temp_res = np.append(temp_res, corpus_vec, axis=0)

    return temp_res

# 数据集
ptm_embedding = PTM_Embedding(model_type="word2vec", pre_train_path=r"../model/ptm/word2vec.model")
corpus = pd.read_csv(r'../data/ecom/corpus.tsv', sep='\t', header=None).sample(n=10000)
train = load_data(r'../data/ecom/train.query.txt')
dev = load_data(r'../data/ecom/dev.query.txt')
corpus = corpus[1].values

# =====================================================================
# index
hnsw = HNSW('cosine')
vector = fit_vector(corpus)
for index, i in enumerate(vector):
    hnsw.add(i)
# 保存
with open('../model/index_model/word2vec-cosine-256.ind', 'wb') as f:
    picklestring = pickle.dump(hnsw, f, pickle.HIGHEST_PROTOCOL)
# ============================================================================
# search
# load index model
hnsw = pickle.load(open('../model/index_model/word2vec-cosine-256.ind', 'rb'))

query_vec = ptm_embedding.get_w2v_embedding("隔夜衣架落地落地立式挂家用网红衣帽架")
idx = hnsw.search(query_vec, k=10)
idx = [(corpus[k[0]], k[1]) for k in idx]
pprint.pprint(idx)
