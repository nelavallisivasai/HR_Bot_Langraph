# Initialize an empty chat history list
chat_history = []

def add_to_chat_history(user_input, model_response):
    # Format the conversation as a single string with user input and model response
    conversation = f"User: {user_input}\nAssistant: {model_response}"
    
    # Append the conversation to chat history
    chat_history.append(conversation)
    
    # Maintain only the last 2 conversations
    if len(chat_history) > 2:
        chat_history.pop(0)  # Remove the oldest conversation

# Example usage
user_input1 = "Hello, how are you?"
model_response1 = "I'm doing great, thank you!"

user_input2 = "What's the weather like today?"
model_response2 = "It's sunny with a slight chance of rain."

user_input3 = "Can you tell me a joke?"
model_response3 = "Sure! Why don't scientists trust atoms? Because they make up everything!"

# Add conversations
add_to_chat_history(user_input1, model_response1)
add_to_chat_history(user_input2, model_response2)
add_to_chat_history(user_input3, model_response3)

# Print chat history (only last 2 conversations)
print("\n--- Chat History ---\n" + "\n\n".join(chat_history))
