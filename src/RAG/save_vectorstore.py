import omegaconf

from vectorstore import load_datastore, process_data
from utils import load_config

def save_vectorstore(cfg: omegaconf.DictConfig):
    datastore = load_datastore(cfg.DATA.DATASTORE_PATH)

    process_data(data=datastore, 
                 chunk_size=cfg.DATA.CHUNK_SIZE, 
                 chunk_overlap=cfg.DATA.CHUNK_OVERLAP, 
                 vectorstore_path=cfg.VECTORSTORE_PATH, 
                 model_name=cfg.EMBEDDING_MODEL_NAME)
    
if __name__ == "__main__":
    cfg = load_config()
    save_vectorstore(cfg)