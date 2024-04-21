import pandas as pd

def make_results(question, gold_answer, answer, relevant_docs, metadata, result_path, result_name):
    rows = []
    rows.append({'query' : question, 'gold_answer' : gold_answer, 'generate_answer' : answer, 'relevant_docs' : relevant_docs, 'metadata' : metadata})
    output = pd.DataFrame(rows)
    output.to_csv(f'{result_path}{result_name}', index=False)
    
    return output