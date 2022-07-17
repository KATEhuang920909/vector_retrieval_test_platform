# -*- coding:utf-8 -*-
import os

import requests
from flask_paginate import Pagination, get_page_parameter
from Logger.logger import log_v
from elasticsearchClass import Index
from forms import SearchForm
from flask import Blueprint


import pickle
from flask import request, jsonify, render_template, redirect
from utils.get_embedding import PTM_Embedding
# from retrieval.search import ANNSearch

# baike_es = Index(index_type="", index_name="baike")
home = Blueprint("home", __name__)
# 加载分类模型
goods_es = Index(index_type="goods_data", index_name="goods")
ptm_embedding = PTM_Embedding(model_type="word2vec", pre_train_path=r"../model/ptm/word2vec.model")

# search
# load index model
search_model = pickle.load(open('../model/index_model/word2vec-cosine-256.ind', 'rb'))


@home.route("/")
def index():
    searchForm = SearchForm()
    return render_template('index.html', searchForm=searchForm)


@home.route("/search", methods=['GET', 'POST'])
def search():
    search_key = request.values.get("search_key", default=None)
    search_engine = request.values.get("search_engine", default="elastic_search")
    if search_key:
        searchForm = SearchForm()
        log_v.error("[+] Search Keyword: " + search_key)
        if search_engine == "elastic_search":
            match_data = goods_es.search(search_key, count=30)["hits"]["hits"]
        else:# 向量检索
            query_vec = ptm_embedding.get_w2v_embedding(search_key)
            match_data = search_model.search_by_fais(query_vec, k=10)
        # 翻页
        PER_PAGE = 10
        page = request.args.get(get_page_parameter(), type=int, default=1)
        start = (page - 1) * PER_PAGE
        end = start + PER_PAGE
        total = 30
        print("最大数据总量:", total)
        pagination = Pagination(page=page, start=start, end=end, total=total)
        context = {
            'match_data': match_data[start:end],
            'pagination': pagination,
            'uid_link': "/goods/"
        }
        print(context)
        if request.method == "GET":
            return render_template('data.html', q=search_key, searchForm=searchForm, **context)
    return redirect('home.index')


@home.route("/add", methods=['GET', 'POST'])
def refresh_posts():
    return jsonify({"success": 1, "message": "All Indexes Refreshed", "data": {}})


@home.errorhandler(400)
def page_not_found(error):
    return "400错误", 400


@home.errorhandler(500)
def page_not_found(error):
    return "500错误", 500


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
