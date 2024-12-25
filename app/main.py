import os
from tools import initialize_agent_with_tools, add_to_chat_history
from dotenv import load_dotenv

load_dotenv()
user = "Siva Sai"

def get_response(user_input):
    agent = initialize_agent_with_tools(api_key=os.getenv("GROQ_API_KEY"), user_name = user)
    response = agent.invoke(user_input)
    add_to_chat_history(user_input, response['output'])
    return response['output']

def main():
    user_input = input("Enter your query: ")
    response = get_response(user_input)
    print(f"Assistant: {response}")

if __name__ == "__main__":
    main()
