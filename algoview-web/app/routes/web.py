from flask import Blueprint, render_template

bp = Blueprint('web', __name__)

@bp.route('/')
def index():
    """Главная страница"""
    return render_template('index.html')

@bp.route('/about')
def about():
    """О проекте"""
    return render_template('about.html')

@bp.route('/algorithms')
def algorithms():
    """Страница с алгоритмами"""
    return render_template('algorithms/index.html') 