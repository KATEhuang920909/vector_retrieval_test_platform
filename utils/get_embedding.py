# -*- coding: utf-8 -*-
# @Time    : 2022/4/17 1:24
# @Author  : huangkai
# @File    : get_embedding.py
import jieba
from gensim.models import KeyedVectors
import numpy as np


# 获取embedding
# 写入到index

class PTM_Embedding:
    def __init__(self, model_type, pre_train_path):
        if model_type == "word2vec":
            self.model = KeyedVectors.load(pre_train_path, mmap='r')

    def get_w2v_embedding(self, contents):
        if type(contents) in [str, int, float]:
            content_seg = jieba.lcut(contents)
            vec = np.mean([self.model[k] for k in content_seg], axis=0).reshape(1, -1)

        else:
            content_seg = [jieba.lcut(k) for k in contents]
            vec = np.array([np.mean([self.model[k] for k in ss], axis=0) for ss in content_seg])
        return vec


if __name__ == '__main__':
    def cosine_distance(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * (np.linalg.norm(b)))


    ptm_embedding = PTM_Embedding("word2vec", "../model/ptm/word2vec.model")
    vec = ptm_embedding.get_w2v_embedding(["狼牙棒狼牙棒", "狼牙棒"])
    vector2 = ptm_embedding.get_w2v_embedding("狼牙棒")
    print(cosine_distance(vec[0], vector2))
