import streamlit as st
from llama_index.llms.groq import Groq
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


documents = SimpleDirectoryReader("./data2/").load_data()

Settings.llm = Groq(
    model="llama-3.3-70b-versatile",
    api_key="",
    temperature=0
)

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)

index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

response = query_engine.query("Summarize the documents")
print(response)