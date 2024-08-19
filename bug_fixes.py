import os
import json
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
from json import JSONDecodeError
from langchain.experimental.autonomous_agents.autogpt.agent import AutoGPT
from FreeLLM import ChatGPTAPI, HuggingChatAPI, BingChatAPI, BardChatAPI
from langchain.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
from langchain.docstore.document import Document
import asyncio
import nest_asyncio
from contextlib import contextmanager
from typing import Optional
from langchain.agents import tool
from langchain import AutoGPT  # Check if this is a valid import
from langchain.tools.file_management.read import ReadFileTool
from langchain.tools.file_management.write import WriteFileTool
from tempfile import TemporaryDirectory
from langchain.tools import BaseTool, DuckDuckGoSearchRun
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain, BaseCombineDocumentsChain
from pydantic import Field
import faiss
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from Embedding import HuggingFaceEmbedding
from langchain.tools.human.tool import HumanInputRun

# Needed since Jupyter runs an async event loop
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Helper function to read JSON file safely
def read_json_file(file_path: str) -> dict:
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except JSONDecodeError:
        raise ValueError(f"Error decoding JSON from {file_path}. Ensure the file contains valid JSON.")
    except FileNotFoundError:
        raise ValueError(f"File {file_path} not found. Ensure the file exists.")

# Model selection
select_model = input(
    "Select the model you want to use (1, 2, 3 or 4) \n \
1) ChatGPT \n \
2) HuggingChat \n \
3) BingChat \n \
4) Google Bard \n \
>>> "
)

# Initialize language model
llm = None
if select_model == "1":
    CG_TOKEN = os.getenv("CHATGPT_TOKEN")
    if CG_TOKEN:
        os.environ["CHATGPT_TOKEN"] = CG_TOKEN
    else:
        raise ValueError("ChatGPT Token is missing. Update your .env file with a valid token.")
    model = "gpt-4" if os.getenv("USE_GPT4") == "True" else "default"
    llm = ChatGPTAPI.ChatGPT(token=CG_TOKEN, model=model)

elif select_model == "2":
    emailHF = os.getenv("emailHF")
    pswHF = os.getenv("pswHF")
    if emailHF and pswHF:
        os.environ["emailHF"] = emailHF
        os.environ["pswHF"] = pswHF
    else:
        raise ValueError("HuggingChat credentials are missing. Update your .env file with valid credentials.")
    llm = HuggingChatAPI.HuggingChat(email=emailHF, psw=pswHF)

elif select_model == "3":
    cookie_path = Path("cookiesBing.json")
    if cookie_path.exists():
        cookies = read_json_file(str(cookie_path))
    else:
        raise ValueError("File 'cookiesBing.json' not found! Create it and add your cookies.")
    llm = BingChatAPI.BingChat(cookiepath=str(cookie_path), conversation_style="creative")

elif select_model == "4":
    GB_TOKEN = os.getenv("BARDCHAT_TOKEN")
    if GB_TOKEN:
        os.environ["BARDCHAT_TOKEN"] = GB_TOKEN
    else:
        raise ValueError("GoogleBard Token is missing. Update your .env file with a valid token.")
    llm = BardChatAPI.BardChat(cookie=GB_TOKEN)

else:
    raise ValueError("Invalid selection. Choose a valid model number.")

# Ensure HuggingFace token is set
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if HF_TOKEN:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN
else:
    raise ValueError("HuggingFace Token is missing. Update your .env file with a valid token.")

# Tools and agents setup
ROOT_DIR = TemporaryDirectory()

@contextmanager
def pushd(new_dir):
    """Context manager for changing the current working directory."""
    prev_dir = os.getcwd()
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(prev_dir)

@tool
def process_csv(csv_file_path: str, instructions: str, output_path: Optional[str] = None) -> str:
    """Process a CSV file using pandas with instructions."""
    with pushd(ROOT_DIR.name):
        try:
            df = pd.read_csv(csv_file_path)
            agent = create_pandas_dataframe_agent(llm, df, max_iterations=30, verbose=True)
            if output_path:
                instructions += f" Save output to disk at {output_path}"
            result = agent.run(instructions)
            return result
        except Exception as e:
            return f"Error: {e}"

async def async_load_playwright(url: str) -> str:
    """Load and scrape a webpage using Playwright."""
    from bs4 import BeautifulSoup
    from playwright.async_api import async_playwright

    try:
        print(">>> WARNING <<<")
        print("If you are running this for the first time, you need to install Playwright.")
        print(">>> AUTO INSTALLING PLAYWRIGHT <<<")
        os.system("playwright install")
        print(">>> PLAYWRIGHT INSTALLED <<<")
    except Exception as e:
        print(f"Error installing Playwright: {e}")

    results = ""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            await page.goto(url)
            page_source = await page.content()
            soup = BeautifulSoup(page_source, "html.parser")
            for script in soup(["script", "style"]):
                script.extract()
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            results = "\n".join(chunk for chunk in chunks if chunk)
        except Exception as e:
            results = f"Error: {e}"
        finally:
            await browser.close()
    return results

def run_async(coro):
    """Run asynchronous coroutine."""
    return asyncio.get_event_loop().run_until_complete(coro)

@tool
def browse_web_page(url: str) -> str:
    """Scrape a whole webpage."""
    return run_async(async_load_playwright(url))

def _get_text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
        length_function=len,
    )

class WebpageQATool(BaseTool):
    name = "query_webpage"
    description = "Browse a webpage and retrieve the information relevant to the question."
    text_splitter: RecursiveCharacterTextSplitter = Field(default_factory=_get_text_splitter)
    qa_chain: BaseCombineDocumentsChain

    def _run(self, url: str, question: str) -> str:
        """Query a webpage for information."""
        result = browse_web_page(url)
        docs = [Document(page_content=result, metadata={"source": url})]
        web_docs = self.text_splitter.split_documents(docs)
        results = []
        for i in range(0, len(web_docs), 4):
            input_docs = web_docs[i:i + 4]
            window_result = self.qa_chain({"input_documents": input_docs, "question": question}, return_only_outputs=True)
            results.append(f"Response from window {i} - {window_result}")
        results_docs = [Document(page_content="\n".join(results), metadata={"source": url})]
        return self.qa_chain({"input_documents": results_docs, "question": question}, return_only_outputs=True)

    async def _arun(self, url: str, question: str) -> str:
        raise NotImplementedError

query_website_tool = WebpageQATool(qa_chain=load_qa_with_sources_chain(llm))

# Memory and embeddings
embeddings_model = HuggingFaceEmbedding.newEmbeddingFunction
embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model, index, InMemoryDocstore({}), {})

# Search tool
web_search = DuckDuckGoSearchRun()

# Define tools
tools = [
    web_search,
    WriteFileTool(),
    ReadFileTool(),
    process_csv,
    query_website_tool,
    # HumanInputRun(), # Activate if you want to permit asking for help from the human
]

# Initialize and run AutoGPT agent
agent = AutoGPT.from_llm_and_tools(
    ai_name="BingChat",
    ai_role="Assistant",
    tools=tools,
    llm=llm,
    memory=vectorstore.as_retriever(search_kwargs={"k": 5}),
)

agent.run([input("Enter the objective of the AI system: (Be realistic!) ")])
