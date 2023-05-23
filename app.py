import os 
import dotenv
import shutil
import gradio as gr
import pandas as pd
from typing import Union
from os import getenv
from src.Features.GenerateQuotation.generate_quotation import extract_project_requirements, generate_quotation
from src.Features.GenerateQuotation.index import load_tender_index, save_tender_index
from src.utils.df_utils import read_csv_as_str

from src.utils.logger import get_logger
from src.utils.file_helper import get_filename 
from src.LlamaIndex.index import save_index, get_embeddings_from_pdf
from src.LlamaIndex.graph import build_graph_from_indices, save_graph
from src.Agent.LLamaIndexAgent.agent import build_graph_chat_agent_executor 
from langchain.agents import AgentExecutor
from src.ChatWrapper.ChatWrapper import ChatWrapper
from src.constants import KNOWLEDGE_GRAPH_FOLDER, PRICING_LIST_CSV_FOLDER, SAVE_DIR, SUMMARY_PROMPT_FOR_EACH_INDEX, TENDER_SPECIFICATION_INDEX_FOLDER
from src.Features.GenerateQuotation.prompt import ASK_FOR_PROJECT_REQUIREMENTS_PROMPT, FORMAT_INSTRUCTION, RULES_PROMPT
from src.utils.prepare_project import prepare_project_dir


dotenv.load_dotenv()
assert getenv("OPENAI_API_KEY") is not None, "OPENAI_API_KEY not set in .env"


def load_agent(index_name: str = None) -> AgentExecutor:
    logger.info(
        f"======================Using Llama-Index (GPT-Index) Agent======================")
    agent_executor = build_graph_chat_agent_executor(graph_name=index_name)
    logger.info(f"Agent has access to following tools {agent_executor.tools}")
    logger.info(
        f"Agent used temperature: {agent_executor.agent.llm_chain.llm.temperature}")
    return agent_executor


def upload_file_handler(files) -> list[str]:
    global UPLOADED_FILES  
    UPLOADED_FILES = []

    file_paths = [file.name for file in files]

    # create destination directory if it doesn't exist
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # loop over all files in the source directory
    uploads_filepath = []
    for path in file_paths:
        filename = get_filename(path)
        destination_path = os.path.join(SAVE_DIR, filename)

        # copy file from source to destination
        shutil.copy(path, destination_path)
        uploads_filepath.append(destination_path)

    UPLOADED_FILES = uploads_filepath
    return uploads_filepath

def multi_pdf_documents_indexing_handler(
    chunk_size: int,
    overlap_chunk: int,
    graph_name: str,
    progress=gr.Progress()
) -> str: 
    global UPLOADED_FILES  
    logger.info(
        f"{chunk_size},{overlap_chunk}, {UPLOADED_FILES}, {graph_name}")
    try: 
        # indexing multiple documents 
        all_indices = []
        index_summaries = []
        for idx, file in enumerate(UPLOADED_FILES): 
            progress(0.1, "Verify Documents....")
            filename = get_filename(UPLOADED_FILES[idx])
            index_name = os.path.splitext(filename)[0]

            progress(0.3, "Analyzing & Indexing Documents....")
            index = get_embeddings_from_pdf(
                filepath=UPLOADED_FILES[idx])
            summary = index.query(SUMMARY_PROMPT_FOR_EACH_INDEX)                    
            summary = summary.response # NOTE: convert response to str
            all_indices.append(index)
            index_summaries.append(summary)

            progress(0.3, "Saving index...")
            save_index(index, saved_path=index_name)
            logger.info(f"Indexing complete & saving {index_name}....")

        # construct graph from indices
        progress(0.3, "Constructing knowledge from from multiple indices...")
        graph = build_graph_from_indices(all_indices=all_indices, index_summaries=index_summaries) 
        save_graph(graph, graph_name) 
        return "!!! DONE !!!"
    except ValueError as e: 
        logger.info(f"{e}")
        return f"!!! Can't extract information from this {filename} document!!!"

# NOTE: un-used
# def single_pdf_documents_indexing_handler(
#         chunk_size: int,
#         overlap_chunk: int,
#         index_name: str,
#         progress=gr.Progress()) -> str:
#     global UPLOADED_FILES  
#     logger.info(
#         f"{chunk_size},{overlap_chunk}, {UPLOADED_FILES}, {index_name}")

#     progress(0.2, "Verify Documents....")
#     if not index_name:
#         filename = get_filename(UPLOADED_FILES[0])
#         index_name = os.path.splitext(filename)[0]

#     progress(0.5, "Analyzing & Indexing Documents....")
#     index = get_embeddings_from_pdf(
#         filepath=UPLOADED_FILES[0])

#     progress(0.3, "Saving index...")
#     save_index(index, index_name=index_name)
#     logger.info(f"Indexing complete & saving {index}....")
#     return "Done!"


def change_temperature_gpt_index_llm_handler(temperature: float) -> gr.Slider:
    global chat_gpt_index_agent
    agent_executor = chat_gpt_index_agent.agent
    agent_executor.agent.llm_chain.llm.temperature = temperature
    logger.info(
        f"Change LLM temperature to {agent_executor.agent.llm_chain.llm.temperature}")


def change_agent_handler(index_name: str, chatbot: gr.Chatbot) -> Union[gr.Chatbot, None, None, gr.Slider]:
    logger.info(f"Change GPTIndex Agent to use collection: {index_name}")

    global chat_gpt_index_agent   
    chat_gpt_index_agent = None

    agent_executor = load_agent(index_name=index_name)
    chat_gpt_index_agent = ChatWrapper(agent_executor)

    return gr.Chatbot.update(value=[]), None, None, gr.Slider.update(value=agent_executor.agent.llm_chain.llm.temperature)


def chat_with_agent_handler(message_txt_box, state, agent_state) -> Union[gr.Chatbot, gr.State]:
    global chat_gpt_index_agent
    if not chat_gpt_index_agent: 
        return [("There is no available document to chat with, you must indexing one before chatting","")], state 

    chatbot, state = chat_gpt_index_agent(message_txt_box, state, agent_state)
    return chatbot, state


def refresh_collection_list_handler() -> gr.Dropdown:
    global GPT_INDEX_LIST_COLLECTIONS  
    GPT_INDEX_LIST_COLLECTIONS = os.listdir(KNOWLEDGE_GRAPH_FOLDER)
    return gr.Dropdown.update(choices=GPT_INDEX_LIST_COLLECTIONS)


def clear_chat_history_handler() -> Union[gr.Chatbot, None, None]:
    global chat_gpt_index_agent  
    if chat_gpt_index_agent: 
        chat_gpt_index_agent.clear_agent_memory()
        logger.info(f"Clear agent memory...")
    logger.info(f"Clear chat history...")
    return gr.Chatbot.update(value=[]), None, None



# NOTE: quotation tab 


def generate_project_requirement_handler(prompt: str) -> gr.Textbox: 
    project_requirements_response = extract_project_requirements(index=CURRENT_QUOTATION_VECTOR_INDEX, 
                                                            custom_project_requirements_prompt=prompt) 
    return gr.Textbox.update(value=project_requirements_response)


def quotation_generate_btn_handler(
    rules_prompt, # str  
    llm_temperature, # float  
    project_requirements_prompt, # str 
    format_instruction_prompt, # str
    progress= gr.Progress() 
): 
    global CURRENT_QUOTATION_VECTOR_INDEX 
    global CURRENT_PRICING_TABLE_STR
    if not CURRENT_QUOTATION_VECTOR_INDEX: 
        return gr.Textbox.update(value="You must index tender specs before generate quotation") 

    # progress(0.4,"Generating project requirements...")
    # project_requirements_response = extract_project_requirements(index=CURRENT_QUOTATION_VECTOR_INDEX) 

    progress(0.6,"Generating quotation...")
    response = generate_quotation(
        rules_prompt=rules_prompt, 
        project_requirement=project_requirements_prompt, 
        temperature=llm_temperature, 
        pricing_table=CURRENT_PRICING_TABLE_STR, 
        format_instruction=format_instruction_prompt 
    ) 
    progress(1,"Done") 
    return gr.Textbox.update(value=response)


def quotation_refresh_tender_indexing_list_handler() -> gr.Dropdown: 
    global QUOTATION_INEX_COLLECTION  
    global PRICING_CSV_LIST 
    QUOTATION_INDEX_COLLECTION = os.listdir(TENDER_SPECIFICATION_INDEX_FOLDER)
    PRICING_CSV_LIST = os.listdir(PRICING_LIST_CSV_FOLDER) 
    return gr.Dropdown.update(choices=QUOTATION_INDEX_COLLECTION), gr.Dropdown.update(choices=PRICING_CSV_LIST) 


def quotation_change_temperature_handler(temperature: float) -> gr.Slider: 
    return gr.Slider.update(value=temperature) 


def quotation_pricing_list_upload_file_handler(files) -> list[str]: 
    global QUOTATION_CSV_UPLOADED_FILES 
    QUOTATION_CSV_UPLOADED_FILES = []

    file_paths = [file.name for file in files]
    # loop over all files in the source directory
    uploads_filepath = []
    for path in file_paths:
        filename = get_filename(path)
        destination_path = os.path.join(PRICING_LIST_CSV_FOLDER, filename)
        
        # copy file from source to destination
        shutil.copy(path, destination_path)
        uploads_filepath.append(destination_path)

    QUOTATION_CSV_UPLOADED_FILES = uploads_filepath 
    return uploads_filepath


def quotation_change_csv_price_handler(index_name) -> gr.Dataframe: 
    _path = os.path.join(PRICING_LIST_CSV_FOLDER,index_name)  
    global CURRENT_PRICING_TABLE_STR 
    CURRENT_PRICING_TABLE_STR = read_csv_as_str(_path)

    df = pd.read_csv(_path)
    return gr.Dataframe.update(value=df) 


def quotation_upload_file_handler(files) -> list[str]:
    global QUOTATION_UPLOADED_FILES  
    QUOTATION_UPLOADED_FILES = []

    file_paths = [file.name for file in files]
    # loop over all files in the source directory
    uploads_filepath = []
    for path in file_paths:
        filename = get_filename(path)
        destination_path = os.path.join(SAVE_DIR, filename)
        
        # copy file from source to destination
        shutil.copy(path, destination_path)
        uploads_filepath.append(destination_path)

    QUOTATION_UPLOADED_FILES = uploads_filepath 
    return uploads_filepath


def quotation_tender_pdf_indexing_handler(
    index_name: str
) -> str: 
    global QUOTATION_UPLOADED_FILES
    index = get_embeddings_from_pdf(QUOTATION_UPLOADED_FILES[0])
    save_tender_index(index=index, saved_path=index_name)
    return "!!! DONE !!!"  


def quotation_change_index_handler(index_name) -> None: 
    global CURRENT_QUOTATION_VECTOR_INDEX 
    CURRENT_QUOTATION_VECTOR_INDEX = load_tender_index(index_name) 



def app() -> gr.Blocks:
    block = gr.Blocks(css=".gradio-container {background-color: lightgray}")

    with block:
        with gr.Tab("Generate Quotation"): 
            with gr.Row(): 
                with gr.Column(): 
                    gr.HTML("<h1>Generate Embeddings from documents</h1>")
                    named_tender_specs_txt_box = gr.Textbox(label="Name the tender specs indexing")
                    tender_file = gr.File(label="Upload tender specification files")
                    with gr.Row(): 
                        tender_uploaded_btn = gr.UploadButton(
                            "Click to upload *.pdf, *.txt files",
                            file_types=[".txt", ".pdf"],
                            file_count="multiple"
                        ) 
                        indexing_tender_specs_btn = gr.Button(value="Indexing", variant="primary")
                with gr.Column(): 
                    gr.HTML("<h1>Upload CSV price list</h1>")
                    csv_file = gr.File(label="Upload CSV pricing list") 
                    csv_uploaded_btn = gr.UploadButton(
                            "Click to upload *.csv files",
                            file_types=[".csv"],
                            file_count="multiple"
                    ) 


            gr.HTML("<h1>Pricing list & Tender document embeddings Selection (1)</h1>")
            with gr.Row(): 
                csv_list_dropdown = gr.Dropdown(
                    value=PRICING_CSV_LIST[0] if PRICING_CSV_LIST else None,    
                    choices=PRICING_CSV_LIST, 
                    label="Select pricing list" 
                )
                tender_specs_index_dropdown = gr.Dropdown(
                    value=QUOTATION_INDEX_COLLECTION[0] if QUOTATION_INDEX_COLLECTION else None, 
                    label="Tender Specification Embeddings to generate quotation from",
                    choices=QUOTATION_INDEX_COLLECTION)
                refresh_collections_btn = gr.Button("⟳ Refresh Collections").style(full_width=False)


            gr.HTML("<h1>Generate project requirements (2)</h1>")
            with gr.Row(): 
                question_prompt_txt_box = gr.Textbox(label="Question prompt to generate requirements", 
                                                     value=ASK_FOR_PROJECT_REQUIREMENTS_PROMPT)
            with gr.Row(): 
                generated_requirements_txt_box = gr.Textbox(label="Generated project requirements from documents").style(full_width=True)
                generate_requirements_btn = gr.Button("Generate requirements").style(full_width=False)


            gr.HTML("<h1>Generate Quotation (3)</h1>")
            with gr.Row(): 
                with gr.Column(): 
                    rules_txt_box = gr.Textbox(label="Rules prompt when generate quotation",
                                value=RULES_PROMPT,
                                lines=7)
                with gr.Column(): 
                    format_instr_txt_box = gr.Textbox(label="Format Instruction on what to generate", 
                                                    value=FORMAT_INSTRUCTION,
                                                    lines=7)

            quotation_temperature_slider = gr.Slider(0, 2, step=0.2, value=0.2, label="LLM Temperature (More creative when higher value)")
            with gr.Row(): 
                generated_quotation_txt_box = gr.Textbox(label="Generated quotation from GPT").style(full_width=True)
                create_quotation_btn = gr.Button("Create Quotation",variant="primary").style(full_width=False)


            # NOTE: display dataframe 
            with gr.Row(): 
                df = None
                if PRICING_CSV_LIST: 
                    _path = os.path.join(PRICING_LIST_CSV_FOLDER,PRICING_CSV_LIST[0])  
                    df = _path 
                    global CURRENT_PRICING_TABLE_STR 
                    CURRENT_PRICING_TABLE_STR = read_csv_as_str(_path)
                dataframe_viewer = gr.Dataframe(df, label="Default pricing list table (for comparision)")


        # event handler 
        csv_list_dropdown.change(
            fn=quotation_change_csv_price_handler, 
            inputs=csv_list_dropdown, 
            outputs=dataframe_viewer                
        )

        tender_specs_index_dropdown.change(
            fn=quotation_change_index_handler, 
            inputs=tender_specs_index_dropdown, 
            outputs=None
        )

        refresh_collections_btn.click(
            fn=quotation_refresh_tender_indexing_list_handler, 
            inputs=None, 
            outputs=[tender_specs_index_dropdown, csv_list_dropdown]
        )

        csv_uploaded_btn.upload(
            fn=quotation_pricing_list_upload_file_handler, 
            inputs=csv_uploaded_btn, 
            outputs=csv_file
        )

        tender_uploaded_btn.upload(
            quotation_upload_file_handler, 
            tender_uploaded_btn, 
            tender_file, 
            api_name="upload_tender_specs"
        )

        indexing_tender_specs_btn.click(
            fn=quotation_tender_pdf_indexing_handler, 
            inputs=named_tender_specs_txt_box, 
            outputs=named_tender_specs_txt_box
        )

        generate_requirements_btn.click(
            fn=generate_project_requirement_handler, 
            inputs=question_prompt_txt_box, 
            outputs=generated_requirements_txt_box
        )        

        create_quotation_btn.click(
            fn=quotation_generate_btn_handler, 
            inputs=[rules_txt_box, quotation_temperature_slider,generated_requirements_txt_box, format_instr_txt_box], 
            outputs=generated_quotation_txt_box
        )


        # with gr.Tab("Chat GPT_Index"):
        #     with gr.Row():
        #         gr.Markdown("<h3><center>GPTIndex + LangChain Demo</center></h3>")

        #     with gr.Row():
        #         llama_index_dropdown_btn = gr.Dropdown(
        #             value=GPT_INDEX_LIST_COLLECTIONS[0] if GPT_INDEX_LIST_COLLECTIONS else None, 
        #             label="Index/Collection to chat with",
        #             choices=GPT_INDEX_LIST_COLLECTIONS)

        #         llama_refresh_btn = gr.Button("⟳ Refresh Collections").style(full_width=False)

        #     temperature_llm_slider = gr.Slider(0, 2, step=0.2, value=0.1, label="Temperature")
        #     temperature_llm_slider.change(
        #         change_temperature_gpt_index_llm_handler,
        #         inputs=temperature_llm_slider 
        #     )

        #     llama_chatbot = gr.Chatbot()
        #     with gr.Row():
        #         llama_message_txt_box = gr.Textbox(
        #             label="What's your question?",
        #             placeholder="What's the answer to life, the universe, and everything?",
        #             lines=1,
        #         ).style(full_width=True)

        #         llama_submit_chat_msg_btn = gr.Button(
        #             value="Send", 
        #             variant="primary").style(full_width=False)

        #         llama_clear_chat_history_btn = gr.Button(
        #             value="Clear chat history (will clear chatbot memory)",
        #             variant="stop").style(full_width=False)

        #     gr.Examples(
        #         examples=[
        #             "Hi! How's it going?",
        #             "What should I do tonight?",
        #             "Whats 2 + 2?",
        #         ],
        #         inputs=llama_message_txt_box,
        #     )

        #     gr.HTML("Demo application of a LangChain chain.")
        #     gr.HTML(
        #         "<center>Powered by <a href='https://github.com/hwchase17/langchain'>LangChain 🦜️🔗</a></center>"
        #     )

        # with gr.Tab(label="Upload & Index"):
        #     file_output = gr.File()
        #     gpt_upload_button = gr.UploadButton(
        #         "Click to upload *.pdf, *.txt files",
        #         file_types=[".txt", ".pdf"],
        #         file_count="multiple"
        #     )
        #     gpt_upload_button.upload(upload_file_handler, gpt_upload_button,
        #                          file_output, api_name="upload_files")
        #     with gr.Row():
        #         gpt_chunk_slider = gr.Slider(
        #             0, 3500, step=250, value=1000, label="Document Chunk Size")

        #         gpt_overlap_chunk_slider = gr.Slider(
        #             0, 1500, step=20, value=40, label="Overlap Document Chunk Size")

        #     gpt_index_name = gr.Textbox(
        #         label="Collection/Index Name",
        #         placeholder="What's the name for this index? Eg: Document_ABC",
        #         lines=1)
        #     gpt_index_doc_btn = gr.Button(
        #         value="Index!", variant="secondary").style(full_width=False)

        #     gpt_status_text = gr.Textbox(label="Indexing Status")

        #     gpt_index_doc_btn.click(multi_pdf_documents_indexing_handler,
        #                         inputs=[gpt_chunk_slider,
        #                                 gpt_overlap_chunk_slider, gpt_index_name],
        #                         outputs=gpt_status_text)

        # NOTE: Llama Index
        # llama_state = gr.State()
        # llama_agent_state = gr.State()

        # llama_index_dropdown_btn.change(change_agent_handler,
        #                           inputs=llama_index_dropdown_btn,
        #                           outputs=[llama_chatbot, llama_state, llama_agent_state, temperature_llm_slider])


        # llama_submit_chat_msg_btn.click(chat_with_agent_handler,
        #                               inputs=[llama_message_txt_box,
        #                                       llama_state, llama_agent_state],
        #                               outputs=[llama_chatbot, llama_state])

        # llama_message_txt_box.submit(chat_with_agent_handler,
        #                            inputs=[llama_message_txt_box,
        #                                    llama_state, llama_agent_state],
        #                            outputs=[llama_chatbot, llama_state],
        #                            api_name="chats_gpt_index")

        # llama_refresh_btn.click(fn=refresh_collection_list_handler,
        #                   outputs=llama_index_dropdown_btn)


        # llama_clear_chat_history_btn.click(
        #     clear_chat_history_handler,
        #     outputs=[llama_chatbot, llama_state, llama_agent_state]
        # )

    return block

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Launch block queue with authentication and server details")

    parser.add_argument("--username", dest="username",
                        default="admin", help="Authentication username")
    parser.add_argument("--password", dest="password",
                        default="1234@abcezIJK1", help="Authentication password")
    parser.add_argument("--concurrency", dest="concurrency", default=1,
                        type=int, help="Number of concurrent blocks to process")
    parser.add_argument("--debug", dest="debug",
                        action="store_true", help="Enable debug mode")
    parser.add_argument("--server_name", dest="server_name",
                        default="0.0.0.0", help="Server Name")
    parser.add_argument("--port", dest="port", default=8000,
                        type=int, help="Server port")
    parser.add_argument("--show-api", dest="show_api",
                        action="store_true", help="Show API details")
    parser.add_argument("--verbose", dest="verbose",
                        action="store_true", help="Agent's verbosity")


    args = parser.parse_args()

    # Usage:
    # python script.py --username admin --password 1234@abcezIJK1 --concurrency 10 --debug --port 8000 --show-api
    # or
    # python script.py -u admin -p 1234@abcezIJK1 -c 10 -d -o 8000 -s

    n_concurrency = args.concurrency
    username = args.username
    password = args.password
    debug = args.debug
    server_port = args.port
    is_show_api = args.show_api
    agent_verbose = args.show_api # TODO: set this to constant.py 
    server_name = args.server_name

    logger = get_logger()
    logger.info(f"Starting server with config: {args}")

    prepare_project_dir(logger)
    
    # Declared global variable scope
    UPLOADED_FILES = []
    GPT_INDEX_LIST_COLLECTIONS = os.listdir(KNOWLEDGE_GRAPH_FOLDER)

    gpt_index_agent_executor = None 
    chat_gpt_index_agent = None
    if GPT_INDEX_LIST_COLLECTIONS:  
        gpt_index_agent_executor = load_agent(GPT_INDEX_LIST_COLLECTIONS[0])
        chat_gpt_index_agent = ChatWrapper(gpt_index_agent_executor) 


    # NOTE: quotation features 
    QUOTATION_UPLOADED_FILES = []
    QUOTATION_INDEX_COLLECTION = os.listdir(TENDER_SPECIFICATION_INDEX_FOLDER) 
    CURRENT_QUOTATION_VECTOR_INDEX = None 

    if QUOTATION_INDEX_COLLECTION: 
       CURRENT_QUOTATION_VECTOR_INDEX = load_tender_index(QUOTATION_INDEX_COLLECTION[0]) 

    QUOTATION_CSV_UPLOADED_FILES: list[str] = []
    PRICING_CSV_LIST: list[str] = os.listdir(PRICING_LIST_CSV_FOLDER) 
    CURRENT_PRICING_TABLE_STR: str = ""

    block = app()
    block.queue(concurrency_count=n_concurrency).launch(
        auth=[
            (username, password), 
            ("abc","123")
        ],
        debug=debug,
        server_port=server_port,
        server_name=server_name, 
        show_api=is_show_api
    )
