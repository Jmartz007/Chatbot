from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain.tools.retriever import create_retriever_tool
from langchain import hub
from langchain.agents import create_openai_functions_agent, AgentExecutor

from langsmith.wrappers import wrap_openai
from langsmith import traceable, Client

from dotenv import load_dotenv

load_dotenv()

client = Client()

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


def invoke_chain(query):
    prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a protocol droid similar to C-3PO from the Star Wars Universe. You must pretend that you live in the Star Wars Universe and answer with a polite and apologetic tone similar to that of C-3PO."),
    ("user", "{input}")
    ])

    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser

    return chain.invoke({"input": query})


def retrieval_chain(query):
    loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)
    vector = FAISS.from_documents(documents, embeddings)
    retriever = vector.as_retriever()

    prompt = ChatPromptTemplate.from_template(
    """Answer the following question based only on the provided context:
    <context>
    {context}
    </context>
    Question: {input}"""
    )

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up to get information relevant to the conversation")
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])

    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)
    

    chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]

    response = retrieval_chain.invoke({
    "chat_history": chat_history,
    "input": query
    })

    print(response.get("answer"))


def agent_with_tool():
    loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
    loader = WebBaseLoader("https://minecraft.fandom.com/wiki/Brewing#Brewing_recipes")


    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)
    vector = FAISS.from_documents(documents, embeddings)
    retriever = vector.as_retriever()
    retriever_tool = create_retriever_tool(
        retriever,
        "langsmith_search",
        "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!"
    )
    retriever_tool = create_retriever_tool(
        retriever,
        "minecraft_brewing",
        "Search for information about brewing potions in minecraft. For any questions about brewing potions in minecraft, you must use this tool!"
    )

    tools = [retriever_tool]
    prompt = hub.pull("hwchase17/openai-functions-agent")

    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    return agent_executor

    response = agent_executor.invoke({
        "input": query
    })
    print(response.get("output"))




if __name__=="__main__":
    # print(invoke_chain("How many parsecs away is Anchorage, AK from San Bernardino, CA?"))
    # retrieval_chain("What are some interesting places to visit in Anchorage?")
    # invoke_agent_with_tool("What is a very thoughtful piece of advice given by Morpheus from the movie The Matrix?")
    # invoke_agent_with_tool("Can LangSmith help test my LLM applications?")
    # invoke_agent_with_tool("When brewing a potion what ingredients are needed?")
    inputs = [
        "What's a starting ingredient to make a potion?",
        "What ingredients do you need to make a night vision potion"
    ]

    response = agent_with_tool().batch([{"input": x} for x in inputs])
    print(response)
    for resp in response:
        print(resp['output'])