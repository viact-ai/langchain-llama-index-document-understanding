import os
from llama_index import GPTSimpleVectorIndex, GPTVectorStoreIndex
from llama_index import  ComposableGraph, GPTVectorStoreIndex, GPTListIndex, GPTTreeIndex

from src.constants import TENDER_SPECIFICATION_INDEX_FOLDER, TENDER_GRAPH_FOLDER
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


def save_tender_graph(
    graph: ComposableGraph, 
    graph_name: str, 
    saved_folder: str = TENDER_GRAPH_FOLDER 
) -> None: 
    graph.save_to_disk(f"./{saved_folder}/{graph_name}.json")
    logger.info(f"Save tender graph embeddings graph to `./{saved_folder}/{graph_name}.json`")


def load_tender_graph(
    graph_name: str, 
    loaded_folder: str = TENDER_GRAPH_FOLDER 
) -> ComposableGraph: 
    graph = ComposableGraph.load_from_disk(f"{loaded_folder}/{graph_name}")
    logger.info(f"Load tender graph embeddings graph to `{loaded_folder}/{graph_name}`")
    return graph 
