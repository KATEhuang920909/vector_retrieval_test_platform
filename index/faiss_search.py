# -*- coding: utf-8 -*-
# @Time    : 2022/7/7 18:53
# @Author  : huangkai
# @File    : faiss_search.py
# -*- coding: utf-8 -*-
# @Time    : 2022/1/11 19:57
# @Author  : huangkai
# @File    : faiss_experiment.py
import time, random
import numpy as np
from sklearn import preprocessing
import pandas as pd

import faiss
import re
import pickle
import gensim
import warnings
import jieba

warnings.filterwarnings("ignore")



class ANNSearch:
    data = []

    def __init__(self, texts, vector):
        # for counter, key in enumerate(model.vocab.keys()):
        #     self.data.append(model[key])
        #     self.word2idx[key] = counter
        #     self.idx2word[counter] = key

        # leaf_size is a hyperparameter

        # 这里加L2正则化，使得余弦相似度就是跟欧式距离等价
        # self.data=preprocessing.normalize(np.array(self.data), norm='l2')

        self.data = vector.astype("float32")
        self.textsidx = dict(zip(texts, np.arange(len(texts))))
        self.idx2texts = dict(zip(np.arange(len(texts)), texts))

        # self.faiss_index = faiss.IndexFlatIP(200)
        # self.faiss_index.train(self.data)
        # self.faiss_index.add(self.data)
        dim, measure = 100, faiss.METRIC_L2
        param = "Flat"
        self.ForceIndex = faiss.index_factory(dim, param, measure)
        self.ForceIndex.train(self.data)
        self.ForceIndex.add(self.data)

        # param = "IVF100,Flat"
        # self.IVFIndex = faiss.index_factory(dim, param, measure)
        # self.IVFIndex.train(self.data)
        # self.IVFIndex.add(self.data)


        # param = "HNSW64"
        # self.HNSW64Index = faiss.index_factory(dim, param, measure)
        # self.HNSW64Index.train(self.data)
        # self.HNSW64Index.add(self.data)



    def search_by_fais(self, query, k=10):
        vector = self.data[self.textsidx[query]]
        dists, inds = self.ForceIndex.search(vector.reshape(-1, 200), k)

        return zip([self.idx2texts[idx] for idx in inds[0][1:]], dists[0][1:])


    # def search_by_fais_V4(self, query, k=10):
    #     vector = self.data[self.textsidx[query]]
    #     dists, inds = self.IVFIndex.search(vector.reshape(-1, 200), k)
    #
    #     return zip([self.idx2texts[idx] for idx in inds[0][1:]], dists[0][1:])


    # def search_by_fais_V6(self, query, k=10):
    #     vector = self.data[self.textsidx[query]]
    #     dists, inds = self.HNSW64Index.search(vector.reshape(-1, 200), k)
    #
    #     return zip([self.idx2texts[idx] for idx in inds[0][1:]], dists[0][1:])

    # def search_by_annoy(self, query, annoy_model, k=10):
    #     vector = self.data[self.textsidx[query]]
    #     result = annoy_model.get_nns_by_vector(vector, k)
    #     text_result = [self.idx2texts[idx] for idx in result[1:]]
    #     return text_result


def time_test(texts, vector):
    # Linear Search
    res = []
    search_model = ANNSearch(texts, vector)
    text = "以前是朋友。"
    text_pcs = jieba.lcut(re.sub(filters, "", str(text)))

    # faiss搜索
    start = time.time()
    for _ in range(1000):
        search_model.search_by_fais(text, k=10)
    stop = time.time()
    print("time/query by faiss_force Search = %.2f s" % (float(stop - start)))
    res.append(float(stop - start))

    # start = time.time()
    # for _ in range(1000):
    #     search_model.search_by_fais_V4(text, k=10)
    # stop = time.time()
    # print("time/query by faiss_ivf_force Search = %.2f s" % (float(stop - start)))
    # res.append(float(stop - start))
    #
    #
    # start = time.time()
    # for _ in range(1000):
    #     search_model.search_by_fais_V6(text, k=10)
    # stop = time.time()
    # print("time/query by faiss_hnsw Search = %.2f s" % (float(stop - start)))
    # res.append(float(stop - start))

    return res


def result_test(texts, vector):
    text = "我跟他只是认识而已他咋了，欠钱吗？"
    search_model = ANNSearch(texts, vector)
    # bm25 检索
    text_pcs = jieba.lcut(re.sub(filters, "", str(text)))


    print("faiss_force:", list(search_model.search_by_fais(text, k=6)))
    # print("faiss_ivp:", list(search_model.search_by_fais_V4(text, k=6)))
    # print("faiss_hnsw:", list(search_model.search_by_fais_V6(text, k=6)))


if __name__ == "__main__":
    # time_test()
    import matplotlib.pyplot as plt

    filters = "[^a-zA-Z\u4e00-\u9fd5]"
    data = pd.read_csv(r"xx.csv")
    data = data["query"].unique()
    result = []
    with open("vector.pkl", "rb") as f:
        vector = pickle.load(f)

    # for i in range(len(data) // 1000):
    #     print((i + 1) * 1000)
    #     res = time_test(data[0:(i + 1) * 1000], vector[0:(i + 1) * 1000])
    #     result.append(res)
    # fs_f = [k[0] for k in result]
    # fs_pq = [k[1] for k in result]
    # fs_lsh = [k[2] for k in result]
    # fs_hnsw = [k[3] for k in result]
    # kt = [k[4] for k in result]
    # bt = [k[5] for k in result]
    # ann = [k[6] for k in result]
    # bm25_ = [k[7] for k in result]
    # plt.plot(fs_f, label="faiss_force")
    # plt.plot(fs_pq, label="faiss_pq")
    # plt.plot(fs_lsh, label="faiss_lsh")
    # plt.plot(fs_hnsw, label="faiss_hnsw")
    # plt.plot(kt, label="kd_tree")
    # plt.plot(bt, label="ball_tree")
    # plt.plot(ann, label="annoy")
    # plt.plot(bm25_, label="bm25")
    # plt.legend()
    # plt.show()
