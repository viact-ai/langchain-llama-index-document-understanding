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

    if not os.path.exists(KNOWLEDGE_GRAPH_FOLDER):
        logger.info(f"created {KNOWLEDGE_GRAPH_FOLDER}")
        os.mkdir(KNOWLEDGE_GRAPH_FOLDER)

    if not os.path.exists(TENDER_SPECIFICATION_INDEX_FOLDER):
        logger.info(f"created {TENDER_SPECIFICATION_INDEX_FOLDER}")
        os.mkdir(TENDER_SPECIFICATION_INDEX_FOLDER)

    if not os.path.exists(PRICING_LIST_CSV_FOLDER):
        logger.info(f"created {PRICING_LIST_CSV_FOLDER}")
        os.mkdir(PRICING_LIST_CSV_FOLDER)
