from elasticsearch import Elasticsearch
from elasticsearch import helpers
from app.Logger.logger import log_v


class Index():

    def __init__(self, index_type: str, index_name: str, ip="127.0.0.1"):

        # self.es = Elasticsearch([ip], http_auth=('elastic', 'password'), port=9200)
        self.es = Elasticsearch("http://127.0.0.1:9200/")
        self.index_type = index_type
        self.index_name = index_name

    @staticmethod
    def load_data():
        path = "data/corpus.tsv"
        with open(path, encoding='utf8') as f:
            data = [k.strip().split("\t")[1] for k in f.readlines()]
        return data

    def data_convert(self, ):
        log_v.info("convert sql data into single doc")

        questions = {}

        data = self.load_data()
        # df = pd.read_csv('faq_sub_question.csv', sep=',', error_bad_lines=False,encoding='utf-8')
        for key, value in enumerate(data):
            if not (key or value):
                continue
            questions[key] = {'doc_id': int(key), 'document': value}

        return questions

    # @staticmethod
    def create_index(self):
        request_body = {
            "mappings": {

                "properties": {
                    "doc_id": {
                        "type": "long",
                        "index": "false"
                    },
                    "document": {
                        "type": "text",
                        "analyzer": "index_ansj",
                        "search_analyzer": "query_ansj",
                        "index": "true"
                    }
                }
            }
        }
        try:
            # 若存在index，先删除index
            self.es.indices.delete(index=self.index_name, ignore=[400, 404])
            res = self.es.indices.create(index=self.index_name, body=request_body)

            log_v.info(res)
            log_v.info("Indices are created successfully")
        except Exception as e:
            log_v.warning(e)
        # if self.es.indices.exists(index=self.index_name) is True:
        #     self.es.indices.delete(index=self.index_name, ignore=[400, 404])
        # self.es.indices.create(index=self.index_name, ignore=400)

    # @staticmethod
    def bulk_index(self, questions, bulk_size):
        log_v.info("Bulk index for question")
        count = 1
        actions = []
        for question_index, question in questions.items():
            action = {
                "_index": self.index_name,
                "_id": question_index,
                "_source": question
            }
            actions.append(action)
            count += 1

            if len(actions) % bulk_size == 0:
                helpers.bulk(self.es, actions)
                actions = []

        if len(actions) > 0:
            helpers.bulk(self.es, actions)
            log_v.info("Bulk index: %s" % str(count))

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
                    "fields": ["document"],  # document
                    "type": "most_fields",
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


if __name__ == '__main__':
    # config = Config(FLAGS.env)
    index = Index(index_type="goods_data", index_name="goods")
    # questions = index.data_convert()
    # print(questions[0])
    # index.create_index()
    # index.bulk_index(questions, bulk_size=10000, )
    print(index.search("真空水杯"))
