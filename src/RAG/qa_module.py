import pickle
import json
import torch

from utils import load_config
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_qa_dataset(qa_path):
    with open(qa_path, 'r') as f:
        data = json.load(f)
    return data

class RetrievalModel:
    def __init__(self, vectorstore_path, search_type, max_docs):
        with open(vectorstore_path, "rb") as f:
            self.store = pickle.load(f)
        self.retriever = self.store.as_retriever(search_type=search_type, search_kwargs={'k':max_docs})
    
    def retrieve_question(self, question):
        search_results = self.retriever.get_relevant_documents(question)
        relevant_docs = [result.page_content for result in search_results]
        metadata = [result.metadata['law'] for result in search_results] 

        return relevant_docs, metadata

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