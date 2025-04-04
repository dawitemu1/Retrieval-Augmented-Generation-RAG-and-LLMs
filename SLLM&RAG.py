import os
import streamlit as st
import torch
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from transformers import GPT2Tokenizer, GPT2Model, pipeline
from langchain_community.llms import HuggingFacePipeline

# Prevent Transformers from using TensorFlow
os.environ["TRANSFORMERS_NO_TF"] = "1"

# Path to the FAISS database
FAISS_DB_PATH = "faiss_db"

# Load GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Define custom embedding class
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

# Instantiate GPT2 embeddings
gpt2_embedding = GPT2Embedding(model, tokenizer)

# Load FAISS vector store
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

# HuggingFace GPT-2 pipeline
generator = pipeline("text-generation", model="gpt2", max_new_tokens=100)
llm = HuggingFacePipeline(pipeline=generator)

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Streamlit UI
st.title("📄 Bank Procedure & Guideline Assistant")
st.write("Ask questions about the Commercial Bank of Ethiopia's procedures and guidelines.")

# Session state tracking
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "question_count" not in st.session_state:
    st.session_state.question_count = 0

# Show previous chat history
for msg in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(msg["user"])
    with st.chat_message("assistant"):
        st.markdown(msg["bot"])

# Chat input
user_message = st.chat_input("Type your question and press Enter to Get Answer...")

if user_message:
    st.chat_message("user").markdown(user_message)

    # Prompt template
    prompt = (
        "You are a helpful assistant trained on Commercial Bank of Ethiopia's procedures and guidelines.\n"
        "Please provide concise and clear answers. Answer the following:\n\n"
        f"User: {user_message.strip()}\nAssistant:"
    )

    try:
        # Generate answer
        response = qa_chain.run(prompt)

        # Show answer clearly
        with st.chat_message("assistant"):
            st.write("💡 **Answer:**", response)

        # Update history
        st.session_state.chat_history.append({"user": user_message.strip(), "bot": f"💡 **Answer:** {response}"})

        # Show guidance message only once
        if st.session_state.question_count == 0:
            st.info("I'm here to help more if you have additional questions!")

        st.session_state.question_count += 1

    except Exception as e:
        st.error(f"Error generating response: {e}")
