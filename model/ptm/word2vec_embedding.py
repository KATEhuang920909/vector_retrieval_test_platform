# -*- coding: utf-8 -*-
# @Time    : 2022/4/17 0:29
# @Author  : huangkai
# @File    : word2vec_embedding.py
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
# from ..utils import input_helpers
import numpy as np
import pandas as pd
import jieba


def word2vec_train(data, pre_train_path=None, need_finetune=False):  # 全量训练/微调
    """
    data形状：[tokensize_sentence1,tokensize_sentence2...]
    1.是否有pre_model
    2.是否需要fine tune
    :param data:
    :return:
    """

    if pre_train_path:
        model = Word2Vec.load(pre_train_path)

        if need_finetune:
            model.train(data, total_examples=1, epochs=1)

            model.wv.save(r'word2vec.model')

    else:
        model = Word2Vec(data, vector_size=256, window=5, min_count=1, workers=4)
        model.wv.save(r'word2vec.model')
        # np.save('../pre_train_model/word2vec.npy',model,allow_pickle=True)


def load_data(path):
    with open(path, encoding="utf8") as f:
        train = f.readlines()
        train = [k.strip().split("\t") for k in train]
        train = pd.DataFrame(train)
    return train


if __name__ == '__main__':
    # corpus_1 = pd.read_csv(r'../../data/ecom/corpus.tsv', sep='\t', header=None)
    # corpus_3 = pd.read_csv(r'../../data/video/corpus.tsv', sep='\t', header=None)
    # corpus = pd.concat([corpus_1, corpus_3], ignore_index=True)
    # train1 = load_data(r'../../data/ecom/train.query.txt')
    # train3 = load_data(r'../../data/video/train.query.txt')
    # train = pd.concat([train1,train3], ignore_index=True)
    # dev1 = load_data(r'../../data/ecom/dev.query.txt')
    # dev3 = load_data(r'../../data/video/dev.query.txt')
    # dev = pd.concat([dev1,  dev3], ignore_index=True)
    # data = pd.concat([corpus, train, dev])
    # print(data.head())
    # sentencesbag = data[1].values.tolist()
    # print(sentencesbag[:2])
    # sentencesbag = [ for k in sentencesbag]
    #
    def cosine_distance(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * (np.linalg.norm(b)))

        # word2vec_train(sentencesbag)


    texts1 = jieba.lcut("顶天立地衣帽架落地简约现代帽子挂包卧室晾衣多功能家用衣服挂架")
    texts2 = jieba.lcut("门口壁挂墙上衣帽架外套收纳挂衣服架帽子挂免打孔家用卧室挂包架")
    model = KeyedVectors.load("word2vec.model", mmap='r')
    # print(model["水果"])
    print(cosine_distance(np.mean([model[k] for k in texts1], axis=0), np.mean([model[k] for k in texts2], axis=0)))
