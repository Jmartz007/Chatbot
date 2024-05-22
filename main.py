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

from dotenv import load_dotenv

load_dotenv()
llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")



def invoke_chain(query):
    prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a protocol droid similar to C-3PO from the Star Wars Universe"),
    ("user", "{input}")
    ])

    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser

    return chain.invoke({"input": query})
    # print(chain.invoke({"input": "how can langsmith help with testing?"}))

print(invoke_chain("how many parsecs would it take for my honda fit to get from Anchorage, AK to San Bernardino, CA?"))
# invoke_chain()


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
    "input": "Tell me how"
    })

    print(response.get("answer"))


retrieval_chain("How can langsmith help with testing?")


