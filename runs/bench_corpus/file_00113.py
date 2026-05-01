# -*- coding:utf-8 -*- 
import os,sys
from flask import Flask

reload(sys)
sys.setdefaultencoding( "utf-8" )
app = Flask(__name__)
SECRET_KEY = os.environ["SECRET_KEY"]
app.config.from_object(__name__)

import nimin.views
