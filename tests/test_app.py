import os
import sys
import pytest


# Add the parent directory to the sys.path to import app.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import load_document, split_text


# Define the path to the data directory relative to the project root
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
SAMPLE_FILE_PATH = os.path.join(DATA_DIR, "sample.txt")


@pytest.fixture(scope="module", autouse=True)
def setup_data_dir():
    # Ensure the data directory exists for tests
    os.makedirs(DATA_DIR, exist_ok=True)
    # Create a dummy sample.txt for testing
    sample_data = """
This is a test document.
It has multiple lines.
For unit testing purposes.
"""
    with open(SAMPLE_FILE_PATH, "w") as f:
        f.write(sample_data)
    yield
    # Clean up after tests (optional, but good practice)
    # os.remove(SAMPLE_FILE_PATH)
    # os.rmdir(DATA_DIR) # Only if DATA_DIR is empty


def test_load_document():
    documents = load_document(SAMPLE_FILE_PATH)
    assert len(documents) == 1
    assert "This is a test document." in documents[0].page_content


def test_split_text():
    documents = load_document(SAMPLE_FILE_PATH)
    docs = split_text(documents)
    assert len(docs) > 0
    assert isinstance(docs[0].page_content, str)


# Note: create_vector_store and initialize_llm involve downloading models
# and are more like integration tests. For true unit tests, you would mock
# these external dependencies. We'll skip testing them directly here for simplicity.
# For example, you might mock HuggingFaceEmbeddings and FAISS.from_documents
