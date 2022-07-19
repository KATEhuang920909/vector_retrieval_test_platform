# -*- coding: utf-8 -*-
"""
 Time : 2022/4/13 1:16
 Author : huangkai
 File : search.py
 mail:18707125049@163.com
 paper:www.
"""

from faiss import normalize_L2, METRIC_INNER_PRODUCT, index_factory
import numpy as np
import pickle
from utils.get_embedding import PTM_Embedding


class ANNSearch:
    data = []

    def __init__(self, texts, vector, dim):
        self.dim = dim
        # for counter, key in enumerate(model.vocab.keys()):
        #     self.data.append(model[key])
        #     self.word2idx[key] = counter
        #     self.idx2word[counter] = key

        # leaf_size is a hyperparameter

        self.data = vector.astype("float32")
        self.textsidx = dict(zip(texts, np.arange(len(texts))))
        self.idx2texts = dict(zip(np.arange(len(texts)), texts))

        normalize_L2(self.data)
        dim, measure = self.dim, METRIC_INNER_PRODUCT
        param = "Flat"
        self.ForceIndex = index_factory(dim, param, measure)
        self.ForceIndex.train(self.data)
        self.ForceIndex.add(self.data)

    def search_by_fais(self, query, k=10):
        if type(query) == str:
            query = self.data[self.textsidx[query]]
        else:
            print(query.shape[0], query.shape[1])
            normalize_L2(query)
        dists, inds = self.ForceIndex.search(query.reshape(-1, self.dim), k)

        return zip([(idx, self.idx2texts[idx]) for idx in inds[0]], dists[0])


if __name__ == "__main__":
    # time_test()
    # import matplotlib.pyplot as plt
    from utils.get_embedding import PTM_Embedding

    # 数据集
    ptm_embedding = PTM_Embedding(model_type="word2vec", pre_train_path=r"../model/ptm/word2vec.model")

    # search
    # load index model
    search_model = pickle.load(open('../model/index_model/word2vec-cosine-256.ind', 'rb'))

    query_vec = ptm_embedding.get_w2v_embedding("实木挂墙电视柜悬空樱桃木墙壁柜黑胡桃木餐边储物柜日式藤编吊柜")
    result = search_model.search_by_fais(query_vec, k=10)
    print("faiss_force:", list(result))
