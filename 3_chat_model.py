from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables from .env file
load_dotenv()

# Check if API key is present
if not os.getenv('GEMINI_API_KEY'):
    raise ValueError("Missing GEMINI_API_KEY in environment variables")

# Initialize the model
model = ChatGoogleGenerativeAI(
    model="gemini-pro",
    temperature=0.7,
    google_api_key=os.getenv('GEMINI_API_KEY')
)

def get_model_response(messages):
    """Helper function to get model response and handle potential errors"""
    try:
        response = model.invoke(messages)
        return response.content
    except Exception as e:
        return f"Error getting response: {str(e)}"

# Create a conversation history
messages = [
    HumanMessage(content="As an AI assistant specializing in technology, programming, and data science, what are the best practices for writing clean Python code?"),
    AIMessage(content="Here are key best practices for clean Python code:\n1. Follow PEP 8 style guide\n2. Use meaningful variable names\n3. Write docstrings for functions\n4. Keep functions small and focused"),
    HumanMessage(content="Can you elaborate on PEP 8?")
]

# Get response from the model
print("\nResponse about PEP 8:")
print(get_model_response(messages))

# Example of using the model for code review
code_review_messages = [
    HumanMessage(content="""As a code review expert, please review this Python code:
    
    def calc(x,y):
        z=x+y
        return z
    """)
]

print("\nCode Review Response:")
print(get_model_response(code_review_messages))

# Example of technical explanation
tech_messages = [
    HumanMessage(content="Please explain what an API is and why it's important in modern software development?")
]

print("\nTechnical Explanation:")
print(get_model_response(tech_messages))

if __name__ == "__main__":
    # You could add additional error handling or logging here
    pass
