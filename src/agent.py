# src/agent.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_cohere import ChatCohere # <-- Cohere LLM
from langchain_classic.chains import LLMChain
import os
from dotenv import load_dotenv

load_dotenv()
# Define the Cohere-based components
template = (
    "You are a helpful assistant specialized in {topic}. "
    "Answer the following question: {question}"
)
prompt = ChatPromptTemplate.from_template(template)
# Use a strong model like 'command-r' for real-world tasks
llm = ChatCohere(model="command-a-03-2025", temperature=0) 

def create_specialized_chain():
    """Returns the full LLM chain."""
    return LLMChain(llm=llm, prompt=prompt)

def run_chain(topic: str, question: str) -> str:
    """Invokes the chain and returns the result."""
    chain = create_specialized_chain()
    # This run is automatically traced to LangSmith if env vars are set
    return chain.invoke({"topic": topic, "question": question})['text']



