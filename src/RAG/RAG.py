import json
import os
import omegaconf
import numpy as np
from tqdm import tqdm

from utils import load_config, torch_seed
from qa_module import load_qa_dataset, RetrievalModel, AnsweringModel
from evaluation import evaluate_text_similarity

def main(cfg : omegaconf.DictConfig):
    torch_seed(cfg.SEED)

    # QA 데이터 로드
    qa_data = load_qa_dataset(cfg.QA_TEST_PATH)
    print("QA test data loaded")

    # Retriever, Generator 모델 로드
    retriever = RetrievalModel(vectorstore_path=cfg.VECTORSTORE_PATH,
                               datastore_path=cfg.DATA.DATASTORE_PATH,
                               search_type=cfg.MODEL.SEARCH_TYPE,
                               max_docs=cfg.DATA.MAX_DOCS)
    generator = AnsweringModel(prompt_path=cfg.PROMPT_PATH, 
                               model_name=cfg.MODEL.MODEL_NAME)
    print("Models loaded")
    
    results = []
    BLEU = []
    ROUGE_L = []
    METEOR = []
    for key, item in tqdm(qa_data.items(), desc="Processing questions"):
        question = item['question']
        gold_answer = item['answer']
        
        retrieved_docs, metadata = retriever.retrieve_question(question, retriever_type=cfg.MODEL.RETRIEVER_TYPE)
        answer = generator.answer_question(question=question,
                                           relevant_docs=retrieved_docs, 
                                           metadata=metadata,
                                           max_new_tokens=cfg.MODEL.MAX_NEW_TOKENS)
        
        metadata_string = "\n".join(metadata)
        additional_answer = f"참조 조문은 다음과 같습니다:\n{metadata_string}"
        full_answer = f"{answer}\n\n{additional_answer}"

        result_dict = {
            'question': question,
            'gold_answer': gold_answer,
            'answer': full_answer,
        }
        
        scores = evaluate_text_similarity(gold_answer, answer)
        BLEU.append(scores['BLEU-1'])
        ROUGE_L.append(scores['ROUGE-L F1'])
        METEOR.append(scores['METEOR'])

        results.append(result_dict)
    
    avg_BLEU = np.mean(BLEU) * 100
    avg_ROUGE_L = np.mean(ROUGE_L) * 100
    avg_METEOR = np.mean(METEOR) * 100

    results.append({
        'avg_BLEU': avg_BLEU,
        'avg_ROUGE_L': avg_ROUGE_L,
        'avg_METEOR': avg_METEOR
    })

    result_name = f'{cfg.MODEL.MODEL_SHORT_NAME}_RAG_result.json'
    results_dir = os.path.dirname(f'{cfg.RESULT_PATH}/{cfg.MODEL.MODEL_SHORT_NAME}/{result_name}')
    os.makedirs(results_dir, exist_ok=True)

    # RAG 결과 json 파일 저장
    with open(f'{results_dir}/{result_name}', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False)
    print("Results saved")

if __name__ == "__main__":
    cfg = load_config()
    main(cfg)