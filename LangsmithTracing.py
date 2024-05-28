from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langsmith import Client
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain import hub
from langchain.agents import create_openai_functions_agent, AgentExecutor

from langsmith.evaluation import EvaluationResult
from langsmith.schemas import Example, Run
from langchain.smith import arun_on_dataset, run_on_dataset
from langchain.evaluation import EvaluatorType
from langchain.smith import RunEvalConfig

import functools
import random


unique_id = random.randint(3,1000)
client = Client()

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

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

inputs = [
    "What's a starting ingredient to make a potion?",
    "Whats the fuel to make potions?"
]

response = agent_with_tool().batch([{"input": x} for x in inputs])

outputs = [
"Water bottle  is the starting base for all potions, made by filling a glass bottle from a cauldron or a water source block.",
"Blaze Powder Fuels the brewing stand. Lasts up to 20 separate times."
]

dataset_name = f"agent-qa-{unique_id}"

dataset = client.create_dataset(
    dataset_name=dataset_name,
    description="A dataset for the brewing equipment"
)

client.create_examples(
    inputs=[{"input": query} for query in inputs],
    outputs=[{"output": answer} for answer in outputs],
    dataset_id=dataset.id
)



def check_not_idk(run: Run, example: Example):
    """Illustration of a custom evaluator."""
    agent_response = run.outputs["output"]
    if "don't know" in agent_response or "not sure" in agent_response:
        score = 0
    else:
        score = 1
    # You can access the dataset labels in example.outputs[key]
    # You can also access the model inputs in run.inputs[key]
    return EvaluationResult(
        key="not_uncertain",
        score=score,
    )

evaluation_config = RunEvalConfig(
    # Evaluators can either be an evaluator type (e.g., "qa", "criteria", "embedding_distance", etc.) or a configuration for that evaluator
    evaluators=[
        check_not_idk,
        # Measures whether a QA response is "Correct", based on a reference answer
        # You can also select via the raw string "qa"
        EvaluatorType.QA,
        # Measure the embedding distance between the output and the reference answer
        # Equivalent to: EvalConfig.EmbeddingDistance(embeddings=OpenAIEmbeddings())
        EvaluatorType.EMBEDDING_DISTANCE,
        # Grade whether the output satisfies the stated criteria.
        # You can select a default one such as "helpfulness" or provide your own.
        RunEvalConfig.LabeledCriteria("helpfulness"),
        # The LabeledScoreString evaluator outputs a score on a scale from 1-10.
        # You can use default criteria or write our own rubric
        RunEvalConfig.LabeledScoreString(
            {
                "accuracy": """
Score 1: The answer is completely unrelated to the reference.
Score 3: The answer has minor relevance but does not align with the reference.
Score 5: The answer has moderate relevance but contains inaccuracies.
Score 7: The answer aligns with the reference but has minor errors or omissions.
Score 10: The answer is completely accurate and aligns perfectly with the reference."""
            },
            normalize_by=10,
        ),
    ]
)

chain_results = run_on_dataset(
    dataset_name=dataset_name,
    llm_or_chain_factory=functools.partial(
        agent_with_tool()),
    evaluation=evaluation_config,
    verbose=True,
    client=client,
    project_name=f"tools-agent-test-5d466cbc-{unique_id}",
    # Project metadata communicates the experiment parameters,
    # Useful for reviewing the test results
    project_metadata={
        "env": "testing-notebook",
        "model": "gpt-3.5-turbo",
        "prompt": "5d466cbc",
    },
)