from flask import Blueprint, render_template

bp = Blueprint('main', __name__)

@bp.route('/')
def index():
    """Главная страница"""
    return render_template('index.html')

@bp.route('/about')
def about():
    """О проекте"""
    return render_template('about.html') 