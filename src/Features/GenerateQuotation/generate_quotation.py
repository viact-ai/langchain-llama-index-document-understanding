from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, BaseMessage
from llama_index import GPTSimpleVectorIndex
from src.Features.GenerateQuotation.prompt import PRICING_TABLE_PROMPT, SYSTEM_PROMPT, RULES_PROMPT, FORMAT_INSTRUCTION, ASK_FOR_PROJECT_REQUIREMENTS_PROMPT
from src.utils.logger import get_logger


logger = get_logger()


def extract_project_requirements(
    index: GPTSimpleVectorIndex, 
    custom_project_requirements_prompt: str = None
) -> str: 
    # TODO: add LLM temperature
    if not custom_project_requirements_prompt: 
        custom_project_requirements_prompt = ASK_FOR_PROJECT_REQUIREMENTS_PROMPT
    response = index.query(custom_project_requirements_prompt)
    return response.response 


def generate_quotation(
    rules_prompt: str, 
    project_requirement: str, 
    temperature: float = 0.0, 
    pricing_table: str = PRICING_TABLE_PROMPT, 
    format_instruction: str = FORMAT_INSTRUCTION 
) -> str: 
    prompt = f"""
I have the following pricing table: 
{pricing_table}

My construction REQUIRED:
{project_requirement}
{rules_prompt}
{format_instruction}
"""

    chat = ChatOpenAI(
        temperature=temperature, 
        model_name="gpt-3.5-turbo"
    )
    response = chat([
        SystemMessage(content=SYSTEM_PROMPT), 
        HumanMessage(content=prompt)
    ])
    response = response.content

    logger.info(f"Requested prompt: {prompt}")
    logger.info(f"Response result: {response}")
    return response



#######################################
async def aextract_project_requirements(
    index: GPTSimpleVectorIndex, 
    custom_project_requirements_prompt: str = None
) -> str: 
    # TODO: add LLM temperature
    if not custom_project_requirements_prompt: 
        custom_project_requirements_prompt = ASK_FOR_PROJECT_REQUIREMENTS_PROMPT
    response = await index.aquery(custom_project_requirements_prompt)
    response = response.response  
    logger.info(f"Extract requirements prompt: {custom_project_requirements_prompt}")
    logger.info(f"Extract requirements result: {response}")
    return response 


async def agenerate_quotation(
    rules_prompt: str, 
    project_requirement: str, 
    temperature: float = 0.0, 
    pricing_table: str = PRICING_TABLE_PROMPT, 
    format_instruction: str = FORMAT_INSTRUCTION 
) -> str: 
    prompt = f"""
I have the following pricing table: 
{pricing_table}

My construction REQUIRED:
{project_requirement}
{rules_prompt}
{format_instruction}
"""
    chat = ChatOpenAI(
        temperature=temperature, 
        model_name="gpt-3.5-turbo"
    )
    response: BaseMessage = await chat.agenerate(messages=[[
        SystemMessage(content=SYSTEM_PROMPT), 
        HumanMessage(content=prompt)
    ]])
    response = response.generations[0][0].text 

    logger.info(f"Requested prompt: {prompt}")
    logger.info(f"Response result: {response}")
    return response

