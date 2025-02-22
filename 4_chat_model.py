from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()

chat = ChatGroq(
    model="llama3-70b-8192",
    temperature=0.7,
    max_tokens=4096,
)

chat_history = []  # use a list to store the chat history

# Set an initial system message
system_message = SystemMessage(content="You are a helpful AI assistant.")
chat_history.append(system_message)

# Start the chat loop
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    # Add user message to chat history
    chat_history.append(HumanMessage(content=user_input))

    # Get response from the model
    response = chat.invoke(chat_history)
    ai_message = AIMessage(content=response.content)

    # Add AI message to chat history
    chat_history.append(ai_message)

    print("AI: " + response.content)
