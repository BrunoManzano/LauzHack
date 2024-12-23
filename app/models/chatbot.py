from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize conversation history with a system message
conversation_history = [
    {
        "role": "system",
        "content": (
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
        )
    }
]

# Function to pass predictions and SHAP values to the model internally
def update_conversation_with_results(predictions, shap_values):
    global conversation_history
    
    # Adding the model's predictions and SHAP values to the conversation
    conversation_history.append({"role": "system", "content": f"Model Predictions: {predictions}"})
    conversation_history.append({"role": "system", "content": f"SHAP Values: {shap_values}"})

# Example of how to use it
# Let's assume you have the following results from your LSTM model
predictions = {
    "Date": "2020-08-01", 
    "Predicted_Value": 3700000,
    "Product": "INNOVIX",
    "Country": "Floresland"
}
shap_values = {
    "MonthlyTreatment": 0.35,
    "YrexMonthlyTreatment": -0.20,
    "PatientsDescribed": 0.50
}

# Update the conversation with predictions and SHAP values
update_conversation_with_results(predictions, shap_values)

# Function to get a response from the chatbot
def get_gpt_response(user_input):
    global conversation_history

    # Add user input to conversation history
    conversation_history.append({"role": "user", "content": user_input})

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

def chat():
    print("Chatbot initialized! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("See you soon!")
            break
        response = get_gpt_response(user_input)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    chat()