FAISS_LOCAL_PATH: str = "./faiss"

GPT_INDEX_LOCAL_PATH: str = "./GPTIndexEmbeddings"

SAVE_DIR: str = "./uploads/"

AGENT_VEROBSE: bool = True

TENDER_SPECIFICATION_INDEX_FOLDER: str = "./tender_specification_embeddings" 

GRAPH_QUERY_CONFIG = [
    {
        "index_struct_type": "simple_dict",
        "query_mode": "default",
        "query_kwargs": {
            "similarity_top_k": 10,
            "response_mode": "tree_summarize"
        }
    },
    {
        "index_struct_type": "simple_dict",
        "query_mode": "default",
        "query_kwargs": {
            "similarity_top_k": 10,
            "response_mode": "tree_summarize"
        }
    },
]


KNOWLEDGE_GRAPH_FOLDER: str = "./knowledge_graph" 


SUMMARY_PROMPT_FOR_EACH_INDEX = """What is the overview for this document?"""