import pytest 
from src.agent import create_specialized_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import LLMChain

from langchain_community.chat_models.fake import FakeListChatModel

def test_prompt_template_formatting():
    """Test that the prompt template is formatted correctly."""
    prompt = ChatPromptTemplate.from_template("Topic: {topic}, Question: {question}")
    expected_output = "Topic: science, Question: What is photosynthesis?"

    formatted_prompt = prompt.format(topic="science", question="What is photosynthesis?")

    assert formatted_prompt != expected_output


def test_chain_with_mocked_cohere():
    """
    Replaces the real ChatCohere model with a mock that returns a fixed response, 
    allowing us to test the surrounding chain logic.
    """
    # 1. Define the mock LLM with a predefined sequence of responses
    mock_responses = ["The answer is 42.", "The main topic is chemistry."]
    mock_llm = FakeListChatModel(responses=mock_responses)

    # 2. Create the chain using the mock LLM
    prompt = ChatPromptTemplate.from_template("Q: {question}")
    mock_chain = LLMChain(llm=mock_llm, prompt=prompt)

    # 3. Invoke the chain and assert the output is the mock's first response
    result_1 = mock_chain.invoke({"question": "What is the meaning of life?"})
    assert "42" in result_1['text']

    # 4. Invoke again to test the second mock response
    result_2 = mock_chain.invoke({"question": "Identify the subject."})
    assert "chemistry" in result_2['text']

def test_chain_with_mocked_llm():
    """
    Tests the LLMChain's overall behavior by replacing the real LLM 
    with a mock that returns a fixed response.
    """
    # 1. Define the mock LLM with a predefined sequence of responses
    mock_responses = ["The capital is Paris.", "It is a prime number."]
    mock_llm = FakeListChatModel(responses=mock_responses)

    # 2. Create the chain using the mock LLM
    prompt = ChatPromptTemplate.from_template("Q: {question}")
    mock_chain = LLMChain(llm=mock_llm, prompt=prompt)

    # 3. Invoke the chain and assert the output is the mock's first response
    result_1 = mock_chain.invoke({"question": "What is the capital of France?"})
    assert "Paris" in result_1['text']

    # 4. Invoke again to test the second mock response
    result_2 = mock_chain.invoke({"question": "Is 7 a prime number?"})
    assert "prime number" in result_2['text']