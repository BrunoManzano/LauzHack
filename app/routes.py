from .models.Preprocessing import preprocessing
from flask import Blueprint, render_template

main = Blueprint('main', __name__)

@main.route('/')
def index():
    return render_template('index.html')

# @main.route('/model')
# def model():
#     return render_template('model.html')

# @main.route('/explainability')
# def explainability():
#     return render_template('explainability.html')
