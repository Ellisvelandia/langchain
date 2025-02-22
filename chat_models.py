from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables from .env file
load_dotenv()

# Initialize the chat model with optimized parameters for Llama 3.3
chat = ChatGroq(
    model="llama3-70b-8192",  # Use the correct model name from Groq's supported models
    temperature=0.7,  # Higher value for more creative responses
    max_tokens=4096  # Maximum tokens for response
)

# Using system and human messages with multilingual capability
result = chat.invoke([HumanMessage(content="What are the three laws of robotics?")])

print(result.content)
