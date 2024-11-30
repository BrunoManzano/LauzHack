from flask import Blueprint, render_template

main = Blueprint('main', __name__)

@main.route('/')
def static_study():
    # Render the static study template
    return render_template('static_study.html')

@main.route('/forecast')
def forecast():
    # Render the forecasting template
    return render_template('forecast.html')
