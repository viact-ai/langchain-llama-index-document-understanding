from src.constants import KNOWLEDGE_GRAPH_FOLDER
from llama_index import  ComposableGraph, GPTVectorStoreIndex, GPTListIndex, GPTTreeIndex


def build_graph_from_indices(
        all_indices: list[GPTVectorStoreIndex], 
        index_summaries: list[str] 
) -> ComposableGraph: 
    graph = ComposableGraph.from_indices(
        GPTTreeIndex,
        all_indices,
        index_summaries=index_summaries,
    )
    return graph 


def save_graph(graph: ComposableGraph, graph_name: str) -> None: 
    graph.save_to_disk(f"./{KNOWLEDGE_GRAPH_FOLDER}/{graph_name}.json")


def load_graph(graph_name: str) -> ComposableGraph: 
    graph = ComposableGraph.load_from_disk(f"{KNOWLEDGE_GRAPH_FOLDER}/{graph_name}")
    return graph 
