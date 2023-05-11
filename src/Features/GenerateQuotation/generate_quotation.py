from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from llama_index import GPTSimpleVectorIndex
from src.Features.GenerateQuotation.prompt import PRICING_TABLE_PROMPT, SYSTEM_PROMPT, RULES_PROMPT, FORMAT_INSTRUCTION, ASK_FOR_PROJECT_REQUIREMENTS_PROMPT


def extract_project_requirements(index: GPTSimpleVectorIndex) -> str: 
    response = index.query(ASK_FOR_PROJECT_REQUIREMENTS_PROMPT)
    return response.response 


def generate_quotation(
    rules_prompt: str, 
    project_requirement: str, 
    temperature: float = 0.0, 
    pricing_table: str = PRICING_TABLE_PROMPT, 
) -> str: 
    prompt = pricing_table + project_requirement + rules_prompt + FORMAT_INSTRUCTION
    chat = ChatOpenAI(temperature=temperature)
    response = chat([
        SystemMessage(content=SYSTEM_PROMPT), 
        HumanMessage(content=prompt)
    ])
    return response.content 
