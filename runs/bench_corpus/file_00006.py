from flask import render_template, Blueprint, jsonify, \
    redirect, url_for, request
from flask_login import login_user, login_required, logout_user
from catalog_app.models import Item, Category, User
from flask_dance.contrib.google import google

core = Blueprint('core', __name__)


# ########## OAUTH SETUP ############################
# Google Oauth is pretty strict where login request can come,
# therefore all protected endpoints first redirected to 'login' and
# from here it is redirected to 'google.login'
@core.route('/login')
def login():
    return redirect(url_for('google.login'))


# Upon successful authentication with Google Oauth, login user locally
@core.route("/finish_login")
def finish_login():
    email = google.get("/oauth2/v2/userinfo").json()['email']
    user = User(email)
    login_user(user)
    if request.args.get("next"):
        return redirect(request.args.get("next"))
    return redirect(url_for('core.index'))


# Logout user, you must be logged in to logout
@core.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for('core.index'))
# ############## OAUTH END ############################


# This is the main page
@core.route('/')
def index():

    # get latest 5 products
    items = Item.query.order_by(Item.id.desc()).limit(5).all()
    categories = Category.query.all()
    return render_template('index.html', items=items, categories=categories)


# Return all items in all categories as a JSON API
@core.route("/catalog.json")
def json_all():

    categories = Category.query.all()
    return jsonify(categories=[category.serialize for category in categories])
