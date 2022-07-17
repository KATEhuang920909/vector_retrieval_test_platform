# coding:utf8
# from .app import app
import sys
sys.path.append("../")
from flask_script import Manager, Server
from retrieval.search import ANNSearch
from app import app

manage = Manager(app)

# 启动命令
manage.add_command("runserver", Server(use_debugger=True))


if __name__ == "__main__":
    manage.run()
