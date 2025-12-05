# tests/test_integration.py
import pytest
from src.agent import run_chain
from langsmith import testing as t

from dotenv import load_dotenv

load_dotenv()


# The @pytest.mark.langsmith decorator logs this test run 
# as an experiment trace in the configured LangSmith project.
@pytest.mark.langsmith
@pytest.mark.parametrize(
    "topic, question, expected_keyword",
    [
        ("geography", "What is the largest city in Brazil?", "São Paulo"),
        ("chemistry", "What is the atomic number of Gold?", "79"),
    ]
)
def test_full_chain_integration(topic: str, question: str, expected_keyword: str):
    """
    Verifies the end-to-end chain operation using the real LLM.
    Asserts based on a required keyword due to non-determinism.
    """
    # Log inputs and expected outputs to the LangSmith trace
    t.log_inputs({"topic": topic, "question": question})
    t.log_reference_outputs({"expected_keyword": expected_keyword})

    # Run the real chain (this makes the network call to the LLM)
    actual_response = run_chain(topic, question)
    
    # Log the actual output
    t.log_outputs({"response": actual_response})

    # Assert: Check for a critical, flexible keyword
    assert expected_keyword in actual_response
    print(f"Test Passed! Response: {actual_response}")

# To run the integration tests (requires network connection):
# $ pytest tests/test_integration.py