import pickle
from transformers import AutoTokenizer
from typing import List, Dict
from datasets import load_dataset
from tqdm import tqdm
from langchain import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.RAG.utils import load_config

'''
FAISS를 통해 vectorstore를 구축하여 pkl 파일로 저장
vectorstore를 구축하기 위해선 embedding이 필요한데, 이는 사용하는 LLM의 tokenizer로 구성
'''

def _embedd_documents(docs, metadata, vectorstore_filepath, model_name):
    embeddings = AutoTokenizer.from_pretrained(model_name)
    vectorstore = FAISS.from_texts(docs, embeddings, metadata)
    with open(vectorstore_filepath, 'wb') as f:
        pickle.dump(vectorstore, f)

def process_data(data, chunk_size, chunk_overlap, vectorstore_filepath, model_name):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )

    processed_docs, metadata = [], []
    for doc in tqdm(data):
        chunks = text_splitter.split_text(doc['text'])
        processed_docs.extend(chunks)
        metadata.extend([{"source": doc['link']}] * len(chunks))
    _embedd_documents(processed_docs, metadata, vectorstore_filepath, model_name)