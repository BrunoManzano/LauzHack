from .models.Preprocessing import preprocessing
from .models.chatbot import get_gpt_response
from flask import Blueprint, render_template, request

main = Blueprint('main', __name__)

@main.route('/')
def index():
    return render_template('index.html')

# Initialize conversation history as a global variable
conversation_history = []

@main.route('/forecast', methods=['GET', 'POST'])
def model():
    forecast = None
    user_message = None
    chatbot_response = None

    if request.method == 'POST':
        # Handle file upload
        if 'datafile' in request.files:
            datafile = request.files['datafile']
            # Add code to process the CSV and generate forecast here
            forecast = "Your forecast data goes here"  # Example of forecast result

        # Handle user input message
        if 'user_input' in request.form:
            user_message = request.form['user_input']
            # Call get_gpt_response to get the chatbot's response
            chatbot_response = get_gpt_response(user_message)
    
    return render_template('forecast.html', forecast=forecast, user_message=user_message, chatbot_response=chatbot_response)