import json
import pickle
from typing import List, Dict
from tqdm import tqdm
from langchain import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings

from utils import load_config

def load_datastore(datastore_path):
    with open(datastore_path, 'r') as f:
        data = json.load(f)
    return data

def _embedd_documents(docs, metadata, vectorstore_path, embedding_model_name):
    embeddings = HuggingFaceBgeEmbeddings(model_name=embedding_model_name)
    vectorstore = FAISS.from_texts(docs, embeddings, metadata)
    with open(vectorstore_path, 'wb') as f:
        pickle.dump(vectorstore, f)

def process_data(data, chunk_size, chunk_overlap, vectorstore_path, model_name):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )

    processed_docs, metadata = [], []
    for law, content in data.items():
        chunks = text_splitter.split_text(content)
        processed_docs.extend(chunks)
        metadata.extend([{"law": law}] * len(chunks))
    _embedd_documents(processed_docs, metadata, vectorstore_path, model_name)