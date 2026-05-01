from app import app, mail
from flask_mail import Message
from flask import render_template

def send_mail_flask(to, template, **kwargs):
    title = kwargs['title'] if 'title' in kwargs else 'SO YUMMY Notification'
    with app.app_context():
        msg = Message(title, sender = "hailiang.adam.wang@outlook.com", recipients=[to])
        msg.body = kwargs['ascii'] if 'ascii' in kwargs else "your email client doesn't support html email."
        msg.html = render_template(template+'.html', to=to, domain=app.config['FRONT_END_URL'], **kwargs)
    mail.send(msg)
