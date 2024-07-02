import os
import sys
import asyncio
from dotenv import load_dotenv
import json
from time import sleep

load_dotenv()
from sqlalchemy.orm import Session
from typing import AsyncIterable, Any
from datetime import datetime
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.chains.llm import LLMChain
from langchain_community.llms import OpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_postgres import PostgresChatMessageHistory
from langsmith import traceable
from langchain.callbacks import AsyncIteratorCallbackHandler
import langchain
from langchain.prompts import ChatPromptTemplate
from typing import List
from prompt import (
    regulation_chat_qa_prompt_template,
    legal_chat_qa_prompt_template,
    condense_question_prompt_template,
)
from langchain.agents import (
    AgentExecutor,
    create_openai_tools_agent,
    create_tool_calling_agent,
    create_structured_chat_agent,
)
from config import settings
import psycopg
from langchain_community.tools.tavily_search import TavilySearchResults

llm = ChatOpenAI(model_name=settings.LLM_MODEL_NAME, temperature=0.3, max_tokens=3000)
question_llm = ChatOpenAI(
    model_name=settings.QUESTION_MODEL_NAME, temperature=0.3, max_tokens=3000
)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


def init_postgres_chat_memory(session_id: str):
    table_name = "message_store"
    sync_connection = psycopg.connect(
        "postgresql://postgres:adaletgpt0318@database-1.cz0c0omykjh2.eu-central-1.rds.amazonaws.com/adaletgpt"
    )
    PostgresChatMessageHistory.create_tables(sync_connection, table_name)
    chat_memory = PostgresChatMessageHistory(
        table_name, session_id, sync_connection=sync_connection
    )

    return chat_memory


@traceable(
    run_type="llm", name="RAG general chat in adaletgpt app", project_name="adaletgpt"
)
def rag_general_chat(question: str):
    """
    making answer witn relevant documents and custom prompt with memory(chat_history) and source link..
    """

    QA_CHAIN_PROMPT = PromptTemplate.from_template(regulation_chat_qa_prompt_template)

    docsearch = PineconeVectorStore(
        pinecone_api_key=settings.PINECONE_API_KEY,
        embedding=embeddings,
        index_name=settings.INDEX_NAME,
    )

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=docsearch.as_retriever(search_kwargs={"k": 6}),
        return_source_documents=True,
        condense_question_llm=question_llm,
        combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT},
    )

    response = qa.invoke({"question": question, "chat_history": []})
    return response["answer"]



@traceable(
    run_type="llm",
    name="RAG with Legal Cases in adaletgpt app",
    project_name="adaletgpt",
)
def rag_legal_chat(question: str):
    QA_CHAIN_PROMPT = PromptTemplate.from_template(legal_chat_qa_prompt_template)
    document_llm_chain = LLMChain(llm=llm, prompt=QA_CHAIN_PROMPT, verbose=False)
    document_prompt = PromptTemplate(
        input_variables=["page_content", "source"],
        template="Context:\n Content:{page_content}\n Source Link:{source}",
    )
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=document_llm_chain,
        document_variable_name="context",
        document_prompt=document_prompt,
    )
    condense_question_prompt = PromptTemplate.from_template(
        condense_question_prompt_template
    )

    question_generator_chain = LLMChain(
        llm=question_llm, prompt=condense_question_prompt
    )

    docsearch = PineconeVectorStore(
        pinecone_api_key=settings.PINECONE_API_KEY,
        embedding=embeddings,
        index_name=settings.LEGAL_CASE_INDEX_NAME,
    )

    # compressor = CohereRerank(top_n=10, cohere_api_key=settings.COHERE_API_KEY)
    # compression_retriever = ContextualCompressionRetriever(
    #     base_compressor=compressor,
    #     base_retriever=docsearch.as_retriever(search_kwargs={"k": 50}),
    # )

    qa = ConversationalRetrievalChain(
        combine_docs_chain=combine_documents_chain,
        question_generator=question_generator_chain,
        verbose=False,
        retriever=docsearch.as_retriever(search_kwargs={"k": 6}),
        return_source_documents=True,
    )

    response = qa.invoke({"question": question, "chat_history": []})
    return response["answer"]

def condense_question(question: str, session_id: str):
    chat_memory = init_postgres_chat_memory(session_id=session_id)
    memory = ConversationSummaryBufferMemory(
        llm=ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0),
        memory_key="chat_history",
        return_messages=True,
        chat_memory=chat_memory,
        max_token_limit=3000,
        output_key="answer",
        ai_prefix="Question",
        human_prefix="Answer",
    )

    condense_question_prompt = PromptTemplate.from_template(
        condense_question_prompt_template
    )

    question_generator_chain = LLMChain(
        llm=question_llm, prompt=condense_question_prompt, verbose=True
    )

    response = question_generator_chain.invoke(
        {"question": question, "chat_history": memory.buffer}
    )

    print("standalone question:", response["text"])

    standalone_question = response["text"]
    return standalone_question

def search_internet(query:str):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                " You are an AI assistant specialized in Turkish Law, and your name is AdaletGPT. Make sure to use travily_search_result_json tool for information",
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{session_id}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    tools = [TavilySearchResults(max_results=1)]

    agent = create_tool_calling_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    response= agent_executor.invoke(
        {"input": query}
    )
    return response["output"]
