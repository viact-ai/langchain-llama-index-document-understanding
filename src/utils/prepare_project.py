import os 
from typing import Any
from src.constants import * 

def prepare_project_dir(logger: Any) -> None:
    if not os.path.exists(FAISS_LOCAL_PATH):
        logger.info(f"created {FAISS_LOCAL_PATH}")
        os.mkdir(FAISS_LOCAL_PATH)

    if not os.path.exists(GPT_INDEX_LOCAL_PATH):
        logger.info(f"created {GPT_INDEX_LOCAL_PATH}")
        os.mkdir(GPT_INDEX_LOCAL_PATH)

    if not os.path.exists(SAVE_DIR):
        logger.info(f"created {SAVE_DIR}")
        os.mkdir(SAVE_DIR)

