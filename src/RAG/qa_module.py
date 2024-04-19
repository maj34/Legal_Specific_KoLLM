import os
import pickle
from typing import Tuple

from dotenv import load_dotenv
from utils import load_config
from transformers import AutoModelForCausalLM, AutoTokenizer

'''
RetrevalModel : vectorstore를 활용해 질문에 대한 관련 문서를 가져오는 모델
AnsweringModel : 관련 문서를 기반와 query를 기반으로 답변(answer)를 생성
'''

@staticmethod
def _format_docs(docs): # relevant_docs를 한번에 연결시켜주는 함수
    return "\n\n".join(doc.page_content for doc in docs)

class RetrievalModel:
    def __init__(self, vectorstore_path, model_name, search_type):
        with open(vectorstore_path, "rb") as f:
            self.store = pickle.load(f)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.retriever = self.store.as_retriever(search_type=search_type)
    

    def retrieve_question(self, question, max_docs):
        relevant_docs = self.retriever.get_relevant_documents(question)[:max_docs]
        relevant_docs = _format_docs(relevant_docs)
        #metadata = 
        
        return relevant_docs, #metadata

class AnsweringModel:
    def __init__(self, model_name):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.prompt = "{question} {docs}" # prompt 구성 아이디어 필요
    
    def answer_question(self, question, relevant_docs, metadata, max_new_tokens):
        prompt = self.prompt.format(question, relevant_docs, metadata)
        input_prompt = self.tokenizer.encode(prompt, return_tensors='pt')
        outputs = self.model.generate(input_prompt, max_new_tokens=max_new_tokens)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer