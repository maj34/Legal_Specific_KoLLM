import pandas as pd
import omegaconf
from tqdm import tqdm

from utils import load_config, torch_seed
from qa_module import load_qa_dataset, RetrievalModel, AnsweringModel
from make_results import make_results

def main(cfg : omegaconf.DictConfig):
    torch_seed(cfg.SEED)

    # generate answer
    qa_data = load_qa_dataset(cfg.QA_PATH)
    print("QA data loaded")

    retriever = RetrievalModel(vectorstore_path=cfg.VECTORSTORE_PATH,
                               search_type=cfg.MODEL.SEARCH_TYPE,
                               max_docs=cfg.DATA.MAX_DOCS)
    generator = AnsweringModel(prompt_path=cfg.PROMPT_PATH, model_name=cfg.MODEL.MODEL_NAME)
    print("Models loaded")
    
    result_name = f'{cfg.MODEL.MODEL_SHORT_NAME}_RAG_result.csv'
    results = []
    for key, item in tqdm(qa_data.items(), desc="Processing questions"):
        question = item['question']
        gold_answer = item['answer']
        relevant_docs, metadata = retriever.retrieve_question(question)
        answer = generator.answer_question(question=question, 
                                           relevant_docs=relevant_docs, 
                                           metadata=metadata, 
                                           max_new_tokens=cfg.MODEL.MAX_NEW_TOKENS)
        results.append(answer)
        output = make_results(question=question,
                              gold_answer=gold_answer,
                              answer=answer,
                              relevant_docs=relevant_docs,
                              metadata=metadata,
                              result_path=cfg.RESULT_PATH,
                              result_name=result_name)
    print("Results saved")

if __name__ == "__main__":
    cfg = load_config()
    main(cfg)