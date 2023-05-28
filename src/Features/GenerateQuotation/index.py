import os
from llama_index import GPTSimpleVectorIndex, GPTVectorStoreIndex

from src.constants import TENDER_SPECIFICATION_INDEX_FOLDER
from src.utils.logger import get_logger


logger = get_logger()


def save_tender_index(index: GPTSimpleVectorIndex, saved_path: str) -> None:
    _path = os.path.join(TENDER_SPECIFICATION_INDEX_FOLDER, saved_path)
    index.save_to_disk(f"{_path}.json")
    logger.info(f"Save index to {_path}.json")
    return _path


def load_tender_index(index_name: str) -> GPTVectorStoreIndex: 
    index_path = os.path.join(TENDER_SPECIFICATION_INDEX_FOLDER, index_name)
    loaded_index = GPTVectorStoreIndex.load_from_disk(
        save_path=index_path,
    )
    logger.info(f"Load index from {index_path}")
    return loaded_index