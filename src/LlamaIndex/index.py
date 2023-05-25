import os
import asyncio 

from src.constants import GPT_INDEX_LOCAL_PATH

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI

from llama_index.node_parser import SimpleNodeParser
from llama_index import GPTSimpleVectorIndex, GPTVectorStoreIndex
from llama_index.readers.schema.base import Document
from llama_index import LLMPredictor, ServiceContext


def _convert_pdf_to_documents(
        filepath: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 40
) -> list[Document]:
    loader = PyPDFLoader(filepath)
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    pages = loader.load_and_split(text_splitter=text_splitter)
    documents = [
        Document.from_langchain_format(page)
        for page in pages
    ]
    return documents


def get_embeddings_from_pdf(
        filepath: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 40
) -> GPTSimpleVectorIndex:
    parser = SimpleNodeParser()
    documents = _convert_pdf_to_documents(
        filepath=filepath,
        chunk_overlap=chunk_overlap,
        chunk_size=chunk_size
    )
    nodes = parser.get_nodes_from_documents(documents)
    index = GPTSimpleVectorIndex(nodes)
    return index


async def aget_embeddings_from_pdf(
        filepath: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 40
) -> GPTSimpleVectorIndex:
    parser = SimpleNodeParser()
    documents = _convert_pdf_to_documents(
        filepath=filepath,
        chunk_overlap=chunk_overlap,
        chunk_size=chunk_size
    )
    nodes = parser.get_nodes_from_documents(documents)
    index = await asyncio.to_thread( 
        GPTSimpleVectorIndex, nodes
    )
    return index


def save_index(index: GPTSimpleVectorIndex, saved_path: str) -> None:
    _path = os.path.join(GPT_INDEX_LOCAL_PATH, saved_path)
    index.save_to_disk(f"{_path}.json")


def load_index(index_name: str, llm_predictor: LLMPredictor = None) -> GPTSimpleVectorIndex:

    if index_name not in os.listdir(GPT_INDEX_LOCAL_PATH): 
        print(os.listdir(GPT_INDEX_LOCAL_PATH))
        raise ValueError(f"`{index_name}` not exists")

    index_path = os.path.join(GPT_INDEX_LOCAL_PATH, index_name)

    # if not llm_predictor:
    #     llm = OpenAI(temperature=0.2, max_tokens=-1)
    #     llm_predictor = LLMPredictor(llm=llm)
    #     service_context = ServiceContext.from_defaults(
    #         llm_predictor=llm_predictor)

    loaded_index = GPTVectorStoreIndex.load_from_disk(
        save_path=index_path,
        # service_context=service_context
    )

    return loaded_index