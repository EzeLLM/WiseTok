import flask, flask.views
from flask import url_for, request, session, redirect, jsonify
from flask.ext.sqlalchemy import SQLAlchemy
from sqlalchemy import Boolean
from flask import render_template
from random import randint
from flask.ext import admin
from flask.ext.admin.contrib import sqla
from flask.ext.admin.contrib.sqla import ModelView
from flask.ext.admin import Admin, BaseView, expose
from progressbar import ProgressBar
import datetime
import threading
from threading import Timer
import requests
import time
from time import sleep
import json
import uuid
import sys
import os


app = flask.Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///ravenadmin.db'
db = SQLAlchemy(app)
app.secret_key = '3746ryfhtu7869tufjr634575yu096w'
# os.environ['DATABASE_URL']
# 'sqlite:///local.db'


class School(db.Model):
    id = db.Column(db.String(32),primary_key=True)
    api_key = db.Column(db.String(32))
    password = db.Column(db.String(20))
    name = db.Column(db.String(50))
    url = db.Column(db.String(50))
    address = db.Column(db.String(120))
    city = db.Column(db.String(30))
    email = db.Column(db.String(60))
    tel = db.Column(db.String(15))

class Report(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    kiosk_id = db.Column(db.String(32))
    date = db.Column(db.String(20))
    school_id = db.Column(db.String(32))
    school_name = db.Column(db.String(50))
    student_entry_log_count = db.Column(db.Integer())
    student_exit_log_count = db.Column(db.Integer())
    faculty_entry_log_count = db.Column(db.Integer())
    faculty_exit_log_count = db.Column(db.Integer())
    unsynced_log_count = db.Column(db.Integer())
    unsent_notification_count = db.Column(db.Integer())

class IngAdmin(sqla.ModelView):
    column_display_pk = True
admin = Admin(app, name='raven')
admin.add_view(IngAdmin(Report, db.session))
admin.add_view(IngAdmin(School, db.session))


@app.route('/', methods=['POST','GET'])
def render_index():
    reports = Report.query.all()
    return flask.render_template('index.html',reports=reports)


@app.route('/report/status/new', methods=['POST'])
def save_status_report():
    data = flask.request.form.to_dict()
    school = School.query.filter_by(api_key=data['api_key']).first()

    if school == None:
        return jsonify(status='Failed',message='School not found'),404

    today = Report.query.filter_by(kiosk_id=data['kiosk_id'],date=data['date']).first()

    if today:
        today.student_entry_log_count = data['student_entry_log_count']
        today.student_exit_log_count = data['student_exit_log_count']
        today.faculty_entry_log_count = data['faculty_entry_log_count']
        today.faculty_exit_log_count = data['faculty_exit_log_count']
        today.unsynced_log_count = data['unsynced_log_count']
        today.unsent_notification_count = data['unsent_notification_count']
        db.session.commit()
        return jsonify(status="Success"),201

    report = Report(
        date = data['date'],
        school_id = data['school_id'],
        kiosk_id = data['kiosk_id'],
        school_name = school.name,
        student_entry_log_count = data['student_entry_log_count'],
        student_exit_log_count = data['student_exit_log_count'],
        faculty_entry_log_count = data['faculty_entry_log_count'],
        faculty_exit_log_count = data['faculty_exit_log_count'],
        unsynced_log_count = data['unsynced_log_count'],
        unsent_notification_count = data['unsent_notification_count']
        )

    db.session.add(report)
    db.session.commit()
    return jsonify(status="Success"),201


@app.route('/db/rebuild', methods=['POST','GET'])
def rebuild_database():
    db.drop_all()
    db.create_all()

    school = School(
        id='123456789',
        api_key='ecc67d28db284a2fb351d58fe18965f9',
        password='test',
        name="Scuola Gesu Bambino",
        url='scuolagesubambino',
        address="10, Brgy Isabang",
        city="Lucena City",
        email="sgb.edu@gmail.com",
        tel="555-8898"
        )
    db.session.add(school)
    db.session.commit()

    return jsonify(status='Success')


if __name__ == '__main__':
    app.debug = True
    app.run(port=7000,threaded=True)

    # port=int(os.environ['PORT']), host='0.0.0.0'