from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_groq import ChatGroq

chat = ChatGroq(
    model="llama3-70b-8192",
    temperature=0.7,
    max_tokens=4096
)

messages = [
    SystemMessage(content="You are an expert in social media marketing."),
    HumanMessage(content="How can I increase my social media followers?"),
    AIMessage(content="You can increase your social media followers by consistently posting high-quality content, engaging with your audience, and using targeted advertising strategies.")
]

response = chat.invoke(messages)
print(response.content)
