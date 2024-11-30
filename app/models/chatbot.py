from openai import OpenAI

from dotenv import load_dotenv
import os

# Load your API key from .env
load_dotenv()
# OpenAI.api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_gpt_response(user_input):
    message = {
        "role": "user",
        "content": user_input
    }
    response = client.chat.completions.create(
        messages = [message],
        model = "gpt-4o",
    )
    return response.choices[0].message.content

def chat():
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Bye.")
            break
        response = get_gpt_response(user_input)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    chat()