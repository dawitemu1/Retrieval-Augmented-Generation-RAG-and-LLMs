import os
import streamlit as st
import torch
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS  # ✅ Updated Import
from langchain.embeddings.base import Embeddings
from transformers import GPT2Tokenizer, GPT2Model, pipeline
from langchain_community.llms import HuggingFacePipeline  # ✅ Updated Import

# **Prevent Transformers from using TensorFlow**
os.environ["TRANSFORMERS_NO_TF"] = "1"

# Constants
FAISS_DB_PATH = "faiss_db"

# Load GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Set padding token

# Define a custom embedding class
class GPT2Embedding(Embeddings):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            outputs = self.model(**inputs)
            embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy())
        return embeddings

    def embed_query(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()

# Instantiate GPT-2 embeddings
gpt2_embedding = GPT2Embedding(model, tokenizer)

# Load FAISS vector store with the same embedding model
if os.path.exists(FAISS_DB_PATH):
    try:
        vector_store = FAISS.load_local(FAISS_DB_PATH, gpt2_embedding, allow_dangerous_deserialization=True)
        retriever = vector_store.as_retriever()
    except Exception as e:
        st.error(f"Error loading FAISS database: {e}")
        st.stop()
else:
    st.error("FAISS database not found. Please process the documents first.")
    st.stop()

# ✅ Wrap GPT-2 Text Generation Pipeline for LangChain compatibility
generator = pipeline("text-generation", model="gpt2", max_new_tokens=100)  # ✅ Adjusted max_new_tokens
llm = HuggingFacePipeline(pipeline=generator)  

# Create Retrieval-Augmented Generation (RAG) pipeline
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Streamlit UI
st.title("📄 Bank Procedure & Guideline Assistant")
st.write("Ask questions about the bank's procedures and guidelines.")

# Input field for user queries
query = st.text_input("Enter your query:", "")

if st.button("Get Answer"):
    if query.strip():
        try:
            response = qa_chain.run(query)
            st.write("💡 **Answer:**", response)
        except Exception as e:
            st.error(f"Error generating response: {e}")
    else:
        st.warning("Please enter a question.")
