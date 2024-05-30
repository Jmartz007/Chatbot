import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_core.tools import tool, Tool
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from bs4 import BeautifulSoup as Soup

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langsmith import traceable, Client
from dotenv import load_dotenv


from langgraph.prebuilt import create_react_agent


load_dotenv()

client = Client()

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

urls = [
    "https://www.ishtar-collective.net/entries/first-gift#book-gifts-and-bargains",
    "https://www.ishtar-collective.net/entries/second-gift#book-gifts-and-bargains",
    "https://www.ishtar-collective.net/entries/third-gift#book-gifts-and-bargains",
    "https://www.ishtar-collective.net/entries/last-bargain#book-gifts-and-bargains",
    "https://www.ishtar-collective.net/entries/the-wise-womans-tale#book-dragonslayers"
]

def retriever(urls: list[str], use_cache: str = True) -> Tool:

    if os.path.exists("faiss_index") and use_cache == True:
        print("vector already exists")
        vector = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        print(vector)
    
    else:
        print("creating new vector")
        loader = WebBaseLoader(urls)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter()
        documents = text_splitter.split_documents(docs)
        vector = FAISS.from_documents(documents, embeddings)
        print(len(documents))
        vector.save_local("faiss_index")

    retriever = vector.as_retriever()
    destiny_lore = create_retriever_tool(retriever, "destiny_lore", "Finds information in the book of dragonslayers from the Destiny 2 video game lore. You must use this tool when the user asks about Destiny 2 and Ahamkara.")

    return destiny_lore

def recursive_retriever(url: str) -> Tool:
    if os.path.exists("recursive_index"):
        print("vector exists")
        vector = FAISS.load_local("recursive_index", embeddings, allow_dangerous_deserialization=True)
    else:
        loader = RecursiveUrlLoader(url=url, max_depth=2, extractor=lambda x: Soup(x, "html.parser").text)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter()
        documents = text_splitter.split_documents(docs)
        vector = FAISS.from_documents(documents, embeddings)
        print(len(documents))
        vector.save_local("recursive_index")
    
    retriever = vector.as_retriever()
    destiny_lore = create_retriever_tool(retriever, "destiny_lore", "Finds information in the book of dragonslayers from the Destiny 2 video game lore. You must use this tool when the user asks about Destiny 2 and Ahamkara.")
    
    return destiny_lore


def new_agent(query):

    tools = [retriever(urls=urls)]

    system_message = "You are an assistant named Rahool and are full of Destiny 2 information. You must only use information retrieved by your tools and nothing else."

    app = create_react_agent(llm, tools, messages_modifier=system_message)


    messages = app.invoke({"messages": [("human", query)]})
    print(messages)
    return messages["messages"][-1].content

print(new_agent("What are the titles of the lore books?"))