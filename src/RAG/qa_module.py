import pickle
import json
import torch

from utils import load_config
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_community.retrievers import BM25Retriever
from langchain.docstore.document import Document

def load_qa_dataset(qa_path):
    with open(qa_path, 'r') as f:
        data = json.load(f)
    return data

class RetrievalModel:
    def __init__(self, vectorstore_path, datastore_path, search_type, max_docs):
        with open(vectorstore_path, "rb") as f:
            self.vectorstore = pickle.load(f)
        with open(datastore_path, 'r') as f:
            data = json.load(f)
        self.datastore = [
            Document(
                page_content=value,
                metadata={'law':key}
            )
            for key, value in data.items()
        ]
        self.FAISS = self.vectorstore.as_retriever(search_type=search_type, search_kwargs={'k':max_docs})
        self.BM25 = BM25Retriever.from_documents(self.datastore, k=max_docs)
    
    def retrieve_question(self, question, retriever_type):
        if retriever_type == 'FAISS':
            search_results = self.FAISS.get_relevant_documents(question)
            retrieved_docs = [result.page_content for result in search_results]
            metadata = [result.metadata['law'] for result in search_results] 
        else: # BM25
            search_results = self.BM25.get_relevant_documents(question)
            retrieved_docs = [result.page_content for result in search_results]
            metadata = [result.metadata['law'] for result in search_results]
        return retrieved_docs, metadata

class AnsweringModel:
    def __init__(self, prompt_path, model_name):
        with open(prompt_path, 'r') as f:
            self.prompt = f.read()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def answer_question(self, question, relevant_docs, metadata, max_new_tokens):
        formatted_laws = '\n'.join([f"{meta} - {doc}" for doc, meta in zip(relevant_docs, metadata)])
        prompt = self.prompt.format(question, formatted_laws)
        input_prompt = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(**input_prompt, max_new_tokens=max_new_tokens).detach().cpu().tolist()
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
        return answer