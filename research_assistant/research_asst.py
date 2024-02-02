from langchain.agents import Tool, load_tools
from langchain_community.tools.file_management.read import ReadFileTool
from langchain_community.tools.file_management.write import WriteFileTool
from langchain_community.tools.semanticscholar.tool import SemanticScholarQueryRun
from langchain_community.utilities import ArxivAPIWrapper, GoogleSerperAPIWrapper
from langchain_community.chat_message_histories import FileChatMessageHistory
from dotenv import load_dotenv

from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool

from langchain.docstore import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from langchain_experimental.autonomous_agents import AutoGPT
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
import faiss

load_dotenv('.env')

# https://github.com/langchain-ai/langchain/blob/master/cookbook/autogpt/autogpt.ipynb

arxiv = ArxivAPIWrapper()
search = GoogleSerperAPIWrapper()
tools = [
    Tool(
        name="arxiv",
        func=arxiv.run,
        description="Useful for when you need to find specific research papers. You should be specific with authors, titles, or technical terms.",
    ),
    Tool(
        name="search",
        func=search.run,
        description="useful for when you need to answer questions about current events. You should ask targeted questions",
    ),
    SemanticScholarQueryRun(),
    WriteFileTool(),
    ReadFileTool(),
]

embeddings_model = OpenAIEmbeddings()

embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

agent = AutoGPT.from_llm_and_tools(
    ai_name="LiteratureReviewGPT",
    ai_role="Assistant",
    tools=tools,
    llm=ChatOpenAI(temperature=0.5, model_name="gpt-4"),
    # llm = Ollama(temperature=0.5, model="mistral"),
    memory=vectorstore.as_retriever(),
    human_in_the_loop=True, # Set to True if you want to add feedback at each step.
    chat_history_memory=FileChatMessageHistory("literature_review_chat_history.txt"),
)
# Set verbose to be true
agent.chain.verbose = True

agent.run([
    "Find papers that expand on Carlini's work in training data extraction attacks. Give me a summary of a few papers.",
    "Use as few steps as possible to complete your task and finish when you have written these five papers' details to a file.",
    "Give me suggestions for a new research direction in this area of ML research."
])
