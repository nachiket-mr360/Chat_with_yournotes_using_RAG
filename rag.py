# phase 1: indexing

# 1.load,
# 2.chunk
# 3.embed
# 4.store 

# phase 2: querying
# 5.retrieve
# 6.generate 

#RAG -> load -> chunk -> embed -> store -> query -> retrive -> answer

import os #to interact with os
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY","False")
import logging #to store errors
logging.getLogger("chromdb.telemetry").setLevel(logging.critical)
from langchain_community.document_loaders import PyPDFLoadr, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

CHROMA_DIR = os.environ.get("CHROMA_DIR", "chroma_db")

def get_llm(model: str | None=None):
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set")
    return ChatGroq(model = model or "llama-3.1-8b-instant", temperature= 0)


    