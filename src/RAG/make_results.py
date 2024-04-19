import pandas as pd

'''
make_results : 전체적인 RAG 모델의 결과를 저장하는 함수
'''

def make_results(data, relevant_docs, answer, metadata):
    rows = []
    question = data['query']
    glod_answer = data['gold_answer']

    rows.append({'query' : question, 'relevant_docs' : relevant_docs, 'metadata' : metadata, 'generate_answer' : answer, 'gold_answer' : glod_answer})
    output = pd.DataFrame(rows)
    return output