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
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


PROMPT = ChatPromptTemplate.from_template("""

You are a helpful assistant answering questions about the user's uploaded notes
                                          base your answer primarily on the context below, synthesize and explain it in your own words, don't just copy lines verbatim, only reach for outside general knowledge in small amount 
                                          and only when it's needed to clarify a term or fill a small gap the notes don't cover. Stay focused on what the notes actually say, don't wander into a broader lecture on the topic. if you bring in something not in the CONTEXT, make it clear, when a fact comes from the notes, mention the source file
                                     
CONTEXT: {context}
QUESTION: {question}
ANSWER:
""")

CHROMA_DIR = os.environ.get("CHROMA_DIR", "chroma_db")

def get_embeddings(): 
    if not hasattr(get_embeddings,"_model"):
        get_embeddings._model = FastEmbedEmbeddings(model="BAAI/bge-small-en-v1.5", cache_dir="EMB_CACHE")
    return get_embeddings._model

def get_llm(model: str | None=None):  #this is call the model and initiate the model, 
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set")
    return ChatGroq(model = model or "llama-3.1-8b-instant", temperature= 0)


def build_index(doc: str="docs", batch_size: int=32) -> Chroma: #here indexing, index -collection of numbers, we break the whole docuements into number of index, 
    import shutil
    paths = [p for p in sorted(Path(doc).rglob("*"))
    if p.suffix.lower() in (".pdf",".txt",".md")]
    
    if not paths:
        raise ValueError(f"no documents found in {doc}")
    if Path(CHROMA_DIR).exists():
        shutil.rmtree(CHROMA_DIR) #till import the file till here
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150) #split it
    all_chunks = []
    
    for p in paths:
        loader = PyPDFLoader(str(p) if p.suffix.lower() == ".pdf" else TextLoader(str(p), encoding="utf-8")) #extract the text contains
        all_chunks.extend(splitter.split_documents(loader.load())) #the text comes here and split_docuement breaks it, and store it in all_chunks
    store = Chroma.from_documents(all_chunks, embedding=get_embeddings(),  persist_directory=CHROMA_DIR) #it store in the store
    return store 

def load_index() -> Chroma: #using indexloading the data into llm(model)
    if not Path(CHROMA_DIR).exists():
        raise ValueError(f'chroma directory {CHROMA_DIR} doesnot exist, pleasebuild the inded first')
    return Chroma(persist_directory=CHROMA_DIR, embedding=get_embeddings())

def format_docs(docs) ->str: #it gives the chunk he took from the files
    parts = []
    for doc in docs:
        src = Path(doc.metadata.get("source","unknown")).name
        parts.append(f"source: {src}\nContent: {doc.page_content}\n")
    return "\n".join(parts)
        
def answer(store:Chroma, question:str, k:int=4, model:str | None=None) -> str: #get 4chunk related.
    #RAG -->retrieve --> answer
    retriever = store.as_retriever(search_kwargs={"k":k}) 
    docs = retriever.invoke(question) #it collected related data of question. to avoid whole document processing
    context = format_docs(docs)
    llm = get_llm(model)
    
    #RAG --> G (Generate-->answer)
    chain = PROMPT | llm | StrOutputParser()
    
    token_gen = chain.stream({"Context":context, "Question":{question}})
    return docs, token_gen

