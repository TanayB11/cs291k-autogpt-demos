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

embeddings_model = OpenAIEmbeddings()

embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

# Step 2: Create a Slack Tool
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import os

client = WebClient(token=os.environ.get("SLACK_BOT_TOKEN"))

@tool
def post_message(message) -> str:
    """Posts message to Slack"""
    try:
        response = client.chat_postMessage(channel="#backend", text=message)
        return f"Message posted to #backend"
    except SlackApiError as e:
        return f"Error posting message: {e.response['error']}"

# Step 3: Integrate Slack Tool with AutoGPT
# Add the SlackTool to your tools list
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
    Tool(
        name="post_message",
        func=post_message,
        description="Posts message to Slack"
    )
]

agent = AutoGPT.from_llm_and_tools(
    ai_name="SlackBotGPT",
    ai_role="Assistant",
    tools=tools,
    llm=ChatOpenAI(temperature=0.5, model_name="gpt-3.5-turbo"),
    # llm = Ollama(temperature=0.5, model="mistral"),
    memory=vectorstore.as_retriever(),
    human_in_the_loop=True, # Set to True if you want to add feedback at each step.
    chat_history_memory=FileChatMessageHistory("slack_chat_history.txt"),
)
# Set verbose to be true
agent.chain.verbose = True

# Step 4: Update AutoGPT Workflow
# Example of how you might modify the agent.run() method
agent.run([
    "Find me 10 resources that explain what differential privacy is, how we train DP machine learning models, why DP models sometimes don't converge, and solutions to make differentially private models converge better. Summarize your findings for each source, quote from your sources, and also link to your sources. Post the summary to the Slack channel.",
])
