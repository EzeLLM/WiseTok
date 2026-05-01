#! /bin/python3

"""
This is a flask program to tell a map what to display

"""

import flask
from flask import render_template
from flask import request
from flask import url_for
from flask import jsonify # For AJAX transactions

import json
import logging

# Date handling 
import arrow # Replacement for datetime, based on moment.js
import datetime # But we still need time
from dateutil import tz  # For interpreting local times

# Our own module
#import acp_times

###
# Globals
###
app = flask.Flask(__name__)
import CONFIG

app = flask.Flask(__name__)
import CONFIG
app.secret_key = CONFIG.secret_key  # Should allow using session variables

import secrets

accessToken = secrets.secret()

###
# Pages
###

page = CONFIG.POI
dest = []
lat = {}
long = {}


with open(page) as f: 
    
    data = [line.strip().split(",") for line in f.readlines()]
    dest = [d[0] for d in data]
    lat = {d[0]: d[1] for d in data} 
    long = {d[0]: d[2] for d in data}

@app.route("/")
@app.route("/index")
def index():
  flask.g.locations = CONFIG.number_of_bars
  app.logger.debug("Main page entry")
  flask.session['lat'] = lat
  flask.session['long'] = long
  flask.session['dest'] = dest
  flask.session['secret'] = str(accessToken)
  return flask.render_template('map.html')


@app.errorhandler(404)
def page_not_found(error):
    app.logger.debug("Page not found")
    flask.session['linkback'] =  flask.url_for("/")
    return flask.render_template('page_not_found.html'), 404


###############
#
# AJAX request handlers 
#   These return JSON, rather than rendering pages. 
#
###############



###################
if __name__ == "__main__":
    # Standalone. 
    app.debug = True
    app.logger.setLevel(logging.DEBUG)
    print("Opening for global access on port {}".format(CONFIG.PORT))
    app.run(port=CONFIG.PORT, host="0.0.0.0")
else:
    # Running from cgi-bin or from gunicorn WSGI server, 
    # which makes the call to app.run.  Gunicorn may invoke more than
    # one instance for concurrent service.
    #FIXME:  Debug cgi interface 
    app.debug=False

