import pickle
import pandas as pd
import shap
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from flask import Blueprint, render_template, request, session
from .models.chatbot import get_gpt_response
from markdown import markdown

main = Blueprint('main', __name__)

@main.route('/')
def index():
    return render_template('index.html')

# Initialize conversation history as a global variable
conversation_history = []

def calculate_shap_values(data, model, forecast_steps=10, num_background_samples=50):
    """
    Calculate SHAP values for a VAR model's predictions.

    Parameters:
    - data (pd.DataFrame): Input time series data (used for training and calculating SHAP values).
    - model (VARResults): Trained VAR model.
    - forecast_steps (int): Number of steps to forecast.
    - num_background_samples (int): Number of background samples for SHAP (typically a small random sample from the data).

    Returns:
    - shap_values (list): SHAP values for each feature in the input data.
    """
    
    # Ensure the number of background samples does not exceed the dataset size
    num_background_samples = min(num_background_samples, len(data))
    
    # Define the prediction function
    def predict_var(input_data):
        # Forecasting using the VAR model
        forecast = model.forecast(input_data[-model.k_ar:], steps=len(input_data))
        return forecast
    
    # Generate background data (sample from the original data)
    background_data = data.sample(n=num_background_samples, random_state=42, replace=False).values
    
    # Initialize SHAP KernelExplainer
    explainer = shap.KernelExplainer(predict_var, background_data)
    
    # Calculate SHAP values for the full input data
    shap_values = explainer.shap_values(data.values)
    
    return shap_values


# Function to do inference
def inference(dataset):
    """
    Perform forecasting on the given dataset.

    Parameters:
    - dataset: DataFrame containing the time series data.

    Returns:
    - forecast: The forecasted values.
    """
    model = pickle.load(open('app\models\model.pkl', 'rb'))
    train_size = int(len(dataset) * 0.8)  # 80% for training, 20% for testing
    train_data = dataset[:train_size]
    test_data = dataset[train_size:]
    maxlags = 5

    # Forecasting
    forecast_steps = len(test_data)
    forecast = model.forecast(train_data.values[-maxlags:], forecast_steps)

    # Plot the forecast
    plot_forecast(test_data.values, forecast, train_size)
    print("Forecast:", forecast)
    # Plot explainability with shap values
    shap_values = calculate_shap_values(dataset, model, forecast_steps=forecast_steps)
    # shap.summary_plot(shap_values, train_data, plot_type='bar')
    return forecast

def plot_forecast(test_data, forecast, train_size):
    """
    Plots the actual test data and forecasted data, and prints the MSE.

    Parameters:
    - test_data: The actual test data (NumPy array).
    - forecast: The forecasted values (NumPy array).
    - train_size: The number of training data points.
    """
    # Calculate Mean Squared Error
    mse = mean_squared_error(test_data, forecast)
    print(f"Mean Squared Error (MSE): {mse}")

    # Create the figure and plot
    plt.figure(figsize=(20, 5))

    # Extract one specific series for demonstration (e.g., first column)
    real_values = test_data[:, 0] if len(test_data.shape) > 1 else test_data
    forecast_values = forecast[:, 0] if len(forecast.shape) > 1 else forecast

    # Print the mse next to the title
    plt.figtext(0.5, 0.95, f'Mean Squared Error: {mse:.2f}', ha='center', va='center', fontsize=12, color='red')

    # Plot the actual test data and forecasted data
    plt.plot(range(train_size, train_size + len(real_values)), real_values, label='Actual Test Data', color='blue')
    plt.plot(range(train_size, train_size + len(forecast_values)), forecast_values, label='Forecasted Data', color='orange')

    # Add title and labels
    plt.title('Actual vs Forecasted Data')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    # Display the plot
    plt.show()

@main.route('/forecast', methods=['GET', 'POST'])
def model():
    """
    Handles requests to the /forecast endpoint for uploading a dataset and generating a forecast.
    """
    forecast = None
    plot_path = None
    user_message = None
    chatbot_response = None

    if 'conversation_history' not in session:
        session['conversation_history'] = []

    if request.method == 'GET':
        # Load the dataset and perform inference when the page loads for the first time
        try:
            df_pivoted = pd.read_csv('notebooks/inference_dataset.csv')
            forecast = inference(df_pivoted)  # Perform inference and plot the result
            print("Forecast:", forecast)
            # Compute the shap values from the forecast
        except Exception as e:
            print(f"Error loading dataset: {e}")
            forecast = "Error loading dataset."

    if request.method == 'POST':
        # Handle user input message (chatbot logic)
        if 'user_input' in request.form:
            user_message = request.form['user_input']
            # Call get_gpt_response to get the chatbot's response
            chatbot_response = markdown(get_gpt_response(user_message))

            session['conversation_history'].append({'user_message': user_message, 'chatbot_response': chatbot_response})

    conversation_history = session['conversation_history']
    
    return render_template('forecast.html', forecast=forecast, plot_path=plot_path, conversation_history=conversation_history)
