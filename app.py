from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os

# Ensure the data directory exists
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# 1. Load Document
def load_document(file_path):
    loader = TextLoader(file_path)
    documents = loader.load()
    return documents

# 2. Split Text
def split_text(documents):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    return docs

# 3. Create Embeddings and Vector Store
def create_vector_store(docs):
    # Using a local embedding model
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

# 4. Initialize LLM (using a local model for simplicity)
def initialize_llm():
    model_id = "distilgpt2"  # A small, fast model for demonstration
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=50,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.15
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

# Main RAG process
def run_rag(query, file_path="data/sample.txt"):
    print(f"Loading document from {file_path}...")
    documents = load_document(file_path)
    print("Splitting text...")
    docs = split_text(documents)
    print("Creating vector store...")
    vectorstore = create_vector_store(docs)
    print("Initializing LLM...")
    llm = initialize_llm()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    print(f"Running query: {query}")
    result = qa_chain.run(query)
    return result

if __name__ == "__main__":
    # Create a dummy data file for testing
    sample_data = """
    The quick brown fox jumps over the lazy dog.
    This is a sample document for testing the RAG application.
    It contains some information about various topics.
    The capital of France is Paris.
    The highest mountain in the world is Mount Everest.
    """
    with open(os.path.join(DATA_DIR, "sample.txt"), "w") as f:
        f.write(sample_data)

    print("Sample data file created at data/sample.txt")

    # Example usage
    query = "What is the capital of France?"
    answer = run_rag(query)
    print(f"Answer: {answer}")

    query = "What is the highest mountain in the Japan?"
    answer = run_rag(query)
    print(f"Answer: {answer}")

# This is a dummy comment to trigger a new CI/CD run. (v2)
