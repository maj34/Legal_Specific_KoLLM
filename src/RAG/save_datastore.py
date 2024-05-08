import omegaconf
import json
from langchain.docstore.document import Document

from datastore import load_datastore, embed_documents, process_data
from utils import load_config

def save_vectorstore(cfg: omegaconf.DictConfig):
    datastore = load_datastore(cfg.DATA.DATASTORE_PATH)
    processed_docs, metadata = process_data(data=datastore, chunk_size=cfg.DATA.CHUNK_SIZE, chunk_overlap=cfg.DATA.CHUNK_OVERLAP)
    
    # datastore 구축
    if cfg.RETRIEVER_TYPE == "FAISS":
        embed_documents(docs=processed_docs, 
                        metadata=metadata, 
                        vectorstore_path=cfg.DATA.VECTORSTORE_PATH, 
                        embedding_model_name=cfg.MODEL.EMBEDDING_MODEL_NAME)
    else:
        documents = []
        for doc, meta in zip(processed_docs, metadata):
            law_key = meta['law']
            documents.append(Document(
                page_content=doc,
                metadata={"law": law_key}
            ))

        grouped_docs = {doc.metadata['law']: doc.page_content for doc in documents}
        with open(cfg.DATASTORE_PATH, 'w') as f:
            json.dump(grouped_docs, f, ensure_ascii=False)
    
if __name__ == "__main__":
    cfg = load_config()
    save_vectorstore(cfg)