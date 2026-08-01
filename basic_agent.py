import asyncio

from llama_index.llms.groq import Groq
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.tools.yahoo_finance import YahooFinanceToolSpec


MODEL_NAME = "llama-3.3-70b-versatile"
API_KEY = ""  # Replace with your actual API key
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
KNOWLEDGE_SOURCE_PATH = "./data2/"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 20
OUTPUT_TOKENS = 512


def get_llm(model_name, api_key):
    return Groq(model=model_name, api_key=api_key)


def initialize_settings():
    Settings.llm = get_llm(MODEL_NAME, API_KEY)
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
    Settings.num_output = OUTPUT_TOKENS
    Settings.node_parser = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )


def multiply(a: float, b: float):
    return a * b


def add(a: float, b: float):
    return a + b


multiply_tool = FunctionTool.from_defaults(fn=multiply)
add_tool = FunctionTool.from_defaults(fn=add)


def load_index(folder_path):
    initialize_settings()
    documents = SimpleDirectoryReader(folder_path).load_data()
    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=Settings.embed_model,
    )
    index.storage_context.persist()
    return index.as_query_engine(llm=Settings.llm)


query_engine = load_index(KNOWLEDGE_SOURCE_PATH)

budget_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="canadian_budget_2023",
    description="A RAG engine with facts from the documents inside the data2 folder.",
)

finance_tools = YahooFinanceToolSpec().to_tool_list()

all_tools = [budget_tool, multiply_tool, add_tool]
all_tools.extend(finance_tools)


agent = ReActAgent(
    tools=all_tools,
    llm=Settings.llm,
    verbose=True,
)


async def main():
    response = await agent.run(
        "Who is the module leader ?"
    )
    print(response)


if __name__ == "__main__":
    asyncio.run(main())