import pandas as pd
import omegaconf

from src.RAG.vectorstore import process_data
from src.RAG.utils import load_config, torch_seed
from src.RAG.qa_module import RetrievalModel, AnsweringModel
from src.RAG.make_results import make_results

def main(cfg : omegaconf.DictConfig):
    torch_seed(cfg.SEED)

    # vectorstore 구축
    datastore = pd.read_csv(cfg.DATA_PATH)
    datastore = datastore.to_dict()
    process_data(data=datastore, 
                 chunk_size=cfg.DATA.CHUNK_SIZE, 
                 chunk_overlap=cfg.DATA.CHUNK_OVERLAP, 
                 vectorstore_filepath=cfg.VECTORSTORE_PATH, 
                 model_name=cfg.MODEL_NAME)

    # generate answer
    qa_data = pd.read_csv(cfg.QA_PATH)
    retriever = RetrievalModel(vectorstore_path=cfg.VECTORSTORE_PATH,
                               model_name=cfg.MODEL.MODEL_NAME,
                               search_type=cfg.MODEL.SEARCH_TYPE)
    generator = AnsweringModel(model_name=cfg.MODEL.MODEL_NAME)
    
    relevant_docs, metadata = retriever.retrieve_question(question=qa_data['question'], max_docs=cfg.MAX_DOCS)
    answer = generator.answer_question(question=qa_data['question'], 
                                       relevant_docs=relevant_docs, 
                                       metadata=metadata, 
                                       max_new_tokens=cfg.MAX_NEW_TOKENS)
    
    output = make_results(data=qa_data, 
                          relevant_docs=relevant_docs, 
                          answer=answer, 
                          metadata=metadata)
    
    # evaluate
    # output['BLUE'] = 
    output.to_csv(f"{cfg.RESULT_PATH}{cfg.MODEL_SHORT_NAME}_RAG_result", index=False)

if __name__ == "__main__":
    cfg = load_config()
    main(cfg)
