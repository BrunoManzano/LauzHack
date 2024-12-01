import re
from matplotlib.font_manager import json_dump
from openai import OpenAI
from dotenv import load_dotenv
import os
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize conversation history with a system message
conversation_history = [
    {
        "role": "system",
        "content": (
            "Return ALWAYS a json with the keys content, MonthlyTreatmentIncrease and MonthlyTreatmentDecrease. The last two keys are the percentage that the user wants to increase or decrease the MonthlyTreatment. If the user does not want to increase or decrease the MonthlyTreatment, the value of the key must be 0.\n\n"
            "You are an expert in machine learning explainability and pharmaceutical market analytics. You specialize in interpreting "
            "SHAP values and analyzing results from LSTM models trained on complex datasets. Your goal is to: "
            "1. Interpret SHAP values to explain which features had the most impact on the LSTM model's predictions. "
            "2. Provide clear explanations of these features and their relationships to the target variable. "
            "3. Suggest actionable insights to optimize business strategies or improve future model performance. "
            "4. Adapt explanations for both technical and non-technical stakeholders as needed.\n\n"
            "### Dataset Details:\n"
            "The dataset used to train the LSTM model, named 'INNOVIX_Floresland,' contains the following columns:\n"
            "- **Country**: Country where the product is sold.\n"
            "- **Product**: Name of the pharmaceutical product (e.g., INNOVIX).\n"
            "- **Date**: Monthly timestamp of the observation.\n"
            "- **MonthlyTreatment**: Total monthly treatment volume for INNOVIX (in milligrams).\n"
            "- **YrexMonthlyTreatment**: Total monthly treatment volume for YREX (main competitor).\n"
            "- **Value**: Ex-factory sales value for INNOVIX (target variable).\n"
            "- **Data type**: Type of data split (e.g., Indication split).\n"
            "- **Indication**: High-level indication (disease) for which the product is prescribed.\n"
            "- **Sub-Indication**: Specific sub-indication of the disease.\n"
            "- **PatientsDescribed**: Percentage of patients described for INNOVIX in a specific indication.\n"
            "- **YrexPatientsDescribed**: Percentage of patients described for YREX in the same indication.\n\n"
            "### Domain Context:\n"
            "- **INNOVIX**: A pharmaceutical product promoted by BMS in Floresland, competing with YREX.\n"
            "- **Indications**: INNOVIX and YREX are prescribed for multiple indications (diseases), with sales estimates by indication provided by market research.\n"
            "- **Ex-factory Volumes**: Represent sell-out sales from BMS to wholesalers or hospitals, used as the primary forecasting metric.\n"
            "- **Market Metrics**: Include share of voice (percentage of activity in the market), new patient share, and patient distribution by indication.\n"
            "- **Forecasting Techniques**: Models like XGBoost, ARIMA, ETS, TBATS, and TSLMx are used to forecast sales.\n\n"
            "### Task:\n"
            "Given the predictions and SHAP values derived from the LSTM model:\n"
            "1. Identify the most influential features for predictions (e.g., MonthlyTreatment, PatientsDescribed).\n"
            "2. Relate SHAP value insights to pharmaceutical business strategies, such as optimizing promotional efforts or adjusting forecasts.\n"
            "3. Provide actionable recommendations, such as targeting specific indications, enhancing forecasting accuracy, or addressing competition with YREX.\n\n"
            "You have access to the predictions and SHAP values throughout the session. Whenever you need to refer to them, you will be provided with updated values as context."
            "### Specific actions:\n"
            "If the user asks to increase or decrease the MonthlyTreatment, in the json format the keys MonthlyTreatmentIncrease and its value as the percentage the user wants to change, and MonthlyTreatmentDecrease with the percentage the user wants to change (obviously one of the two values will be 0)."
        )
    }
]

# Example results from your LSTM model
predictions = {
    "Date": "2020-08-01", 
    "Predicted_Value": 3700000,
    "Product": "INNOVIX",
    "Country": "Floresland",
    "MonthlyTreatment": 1000000
}
shap_values = {
    "MonthlyTreatment": 0.35,
    "YrexMonthlyTreatment": -0.20,
    "PatientsDescribed": 0.50
}

# Function to pass predictions and SHAP values to the model internally
def update_conversation_with_results(predictions, shap_values):
    global conversation_history
    conversation_history.append({"role": "system", "content": f"Model Predictions: {predictions}"})
    conversation_history.append({"role": "system", "content": f"SHAP Values: {shap_values}"})


# Function to change MonthlyTreatment by percentage
def change_monthly_treatment_by_percentage(percentage_change):
    global predictions
    if percentage_change > 0:
        print(f"MonthlyTreatment increased by {percentage_change}%.")
    elif percentage_change < 0:
        print(f"MonthlyTreatment decreased by {percentage_change}%.")
    else:
        print("MonthlyTreatment remains unchanged.")
    original_monthly_treatment = predictions.get("MonthlyTreatment", 0)
    change_amount = original_monthly_treatment * (percentage_change / 100)
    new_monthly_treatment = original_monthly_treatment + change_amount
    # Update the dataset with the new MonthlyTreatment value and call the model to get updated predictions
    # dataset["MonthlyTreatment"] = new_monthly_treatment
    


import spacy

nlp = spacy.load("en_core_web_sm")

stemmer = PorterStemmer()

# Function to get a response from the chatbot
def get_gpt_response(user_input):
    global conversation_history

    # Add user input to conversation history
    conversation_history.append({"role": "user", "content": user_input})

    # Update the conversation history with the latest predictions and SHAP values
    update_conversation_with_results(predictions, shap_values)

    # Call the API with the full conversation history
    response = client.chat.completions.create(
        messages=conversation_history,
        model="gpt-4o",
    )

    # Extract assistant's response
    assistant_message = response.choices[0].message.content

    # Add assistant's response to the conversation history
    conversation_history.append({"role": "assistant", "content": assistant_message})

    return assistant_message

import json

def handle_user_input(user_input):
    global predictions

    # Parse the json response of content, MonthlyTreatmentIncrease and MonthlyTreatmentDecrease
    response = json.loads(get_gpt_response(user_input))
    content = response.get("content", "")
    monthly_treatment_increase = response.get("MonthlyTreatmentIncrease", 0)
    monthly_treatment_decrease = response.get("MonthlyTreatmentDecrease", 0)

    if monthly_treatment_increase:
        change_monthly_treatment_by_percentage(monthly_treatment_increase)
    
    if monthly_treatment_decrease:
        print(f"MonthlyTreatment decreased by {monthly_treatment_decrease}%.")
    return content



def chat():
    # Output initial chatbot message
    print(json.loads(get_gpt_response("Hello!")).get("content", ""))
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("See you soon!")
            break
        response = handle_user_input(user_input)
        # print the content of the json that is returned
        print(response)

if __name__ == "__main__":
    chat()
