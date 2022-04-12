from elasticsearch import Elasticsearch
# import os
# import sys

class elasticSearch():

    def __init__(self, index_type: str, index_name: str, ip="127.0.0.1"):

        # self.es = Elasticsearch([ip], http_auth=('elastic', 'password'), port=9200)
        self.es = Elasticsearch("localhost:9200")
        self.index_type = index_type
        self.index_name = index_name

    def create_index(self):
        if self.es.indices.exists(index=self.index_name) is True:
            self.es.indices.delete(index=self.index_name)
        self.es.indices.create(index=self.index_name, ignore=400)

    def delete_index(self):
        try:
            self.es.indices.delete(index=self.index_name)
        except:
            pass

    def get_doc(self, uid):
        return self.es.get(index=self.index_name, id=uid)

    def insert_one(self, doc: dict):
        self.es.index(index=self.index_name, doc_type=self.index_type, body=doc)

    def insert_array(self, docs: list):
        for doc in docs:
            self.es.index(index=self.index_name, doc_type=self.index_type, body=doc)

    def search(self, query, count: int = 30):
        dsl = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["title", "content", "link"]
                }
            },
            "highlight": {
                "fields": {
                    "title": {}
                }
            }
        }
        match_data = self.es.search(index=self.index_name, body=dsl, size=count)
        return match_data

    def __search(self, query: dict, count: int = 20):
        results = []
        params = {
            'size': count
        }
        match_data = self.es.search(index=self.index_name, body=query, params=params)
        for hit in match_data['hits']['hits']:
            results.append(hit['_source'])

        return results





# def endpcess():
#     nb = os.popen("jps|grep Elasticsearch").read()
#     oldid = nb.split(' ')[0]
#     # print 'oldid'+oldid
#     nb = os.system("kill " + oldid)
#     return 'Elasticsearch is end'
#
#
# def startprcess():
#     os.system("su - es -c '/home/es/es/es-2.4.4/bin/elasticsearch -d'")
#     # nb= os.popen("jps|grep Elasticsearch").read()
#     # print 'newid'+nb.split(' ')[0]
#     return 'Elasticsearch is start'
#
#
# def restartprcess():
#     endpcess()
#     startprcess()
#     return 'Elasticsearch is restart'
#
#
# if __name__ == '__main__':
#     try:
#         name = sys.argv[1]
#         if name == '-e':
#             info = endpcess()
#             print(info)
#         elif name == '-s':
#             info = startprcess()
#             print(info)
#         elif name == '-r':
#             info = restartprcess()
#             print(info)
#         elif name == '-h':
#             print('cmd:python es.py -e  ----stop Elasticsearch service')
#             print('cmd:python es.py -s  ----start Elasticsearch service')
#             print('cmd:python es.py -r  ----restart Elasticsearch service')
#             print('cmd:python es.py -h  ----cmd help')
#     except:
#         print('cmd:python es.py -e  ----stop Elasticsearch service')
#         print('cmd:python es.py -s  ----start Elasticsearch service')
#         print('cmd:python es.py -r  ----restart Elasticsearch service')
#         print('cmd:python es.py -h  ----cmd help')