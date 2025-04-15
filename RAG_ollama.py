from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA

# Load PDF and split text
docs = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(
    PyPDFLoader("Thesis.pdf").load()
)

# Load GPU-accelerated embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cuda"})

# Store embeddings in FAISS
vector_store = FAISS.from_documents(docs, embeddings)
vector_store.save_local("faiss_index")

# Load FAISS and set up retrieval
retriever = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True).as_retriever()

# Connect Mistral with RAG
qa_chain = RetrievalQA.from_chain_type(llm=OllamaLLM(model="mistral"), chain_type="stuff", retriever=retriever)

# Ask a question
query = "Summarize the thesis. Name all the authors"
response = qa_chain.invoke(query)
print(response)
