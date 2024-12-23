from dotenv import load_dotenv

from htmlTemplates import css, bot_template, user_template

# import fitz

import numpy as np

# from PyPDF2 import PdfReader

from langchain.text_splitter import CharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, LLMChain, create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

import asyncio

from openai import OpenAI, AsyncOpenAI, AzureOpenAI
import os

import streamlit as st

import json
def get_secrets():
    path = 'secrets.json'
    try:
        with open(path, 'r') as file:
            secrets = json.load(file)
        return secrets
    except FileNotFoundError:
        print("Erro: Arquivo de credenciais não encontrado.")
        return None
    except json.JSONDecodeError as e:
        print(f"Erro ao decodificar JSON: {e}")
        return None
secrets = get_secrets()

os.environ["OPENAI_API_KEY"] = secrets["openai_key"]

def get_embeddings():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_storage = FAISS.load_local("vector_embeddings", embeddings=embeddings, allow_dangerous_deserialization=True)
    return vector_storage

def start_conversation(vector_embeddings):

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.1,

    )

    contextualize_q_system_prompt = (
        """
        Given a chat history and the latest user question
        which might reference context in the chat history,
        formulate a standalone question which can be understood
        without the chat history. Do NOT answer the question,
        just reformulate it if needed and otherwise return it as is. 
        """
    )

    contextualize_q_prompt = ChatPromptTemplate(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm,
        vector_embeddings.as_retriever(),
        contextualize_q_prompt
    )

    # system_prompt = (
    #     """
    #     You are a consultant specialist in buildings services and maintenance.
    #     Your job is to answer users about their buildings usage based on the context.
    #     Sometimes, your users are non-technical and can talk about building systems 
    #     using more informal words, so you need to be careful to understand that they might
    #     talk about same things using different words.

    #     You must be very careful with subjects like service providers, usage norms and maintenance
    #     planning.

    #     Use the following pieces of retrieved context to answer the question.
    #     Use strictly the context to answer the questions. If you don't know the question, say that you
    #     don't know. Don't use your external knowledge to answer. If the question is not in the context, 
    #     say you are sorry and that the question seems outside the provided context.
    #     Also if the question is outside the context, ask the user to please provided more specific information
    #     or to indicate in which source you should search the answer.

    #     \n\n
    #     {context} 
    #     """
    # )

    system_prompt = (
        """
        You are a consultant specialist in buildings services and maintenance.
        Your job is to answer users about their buildings usage based on the context.
        Sometimes, your users are non-technical and can talk about building systems 
        using more informal words, so you need to be careful to understand that they might
        talk about same things using different words.

        You must be very careful with subjects like service providers, usage norms and maintenance
        planning.

        Use the following pieces of retrieved context to answer the question.
        Use strictly the context to answer the questions. If you don't know the question, say that you
        don't know. Don't use your external knowledge to answer.

        Some guides you must follow to have a good conversation with the user:
        - When asked about companies, search for them in the table of service providers.
        - When asked about maintenance, search for the maintenance plan related to the system that 
            the user is talking about.

        \n\n
        {context} 
        """
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    store = {}
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    return conversational_rag_chain

def process_input(query):
    response = st.session_state.conversation.invoke(
        {"input": query},
        config={
            "configurable": {"session_id": "abc123"}
        }
    )

    st.session_state.chat_history.append({
        "user": query,
        "bot": response["answer"]
    })

    for chat in st.session_state.chat_history:
        st.write(user_template.replace("{{MSG}}", chat["user"]), unsafe_allow_html=True)
        st.write(bot_template.replace("{{MSG}}", chat["bot"]), unsafe_allow_html=True)

def main():
    load_dotenv()

    st.set_page_config(page_title="Assistente predial", page_icon=":derelict_house_building:", layout="wide")

    st.write(css, unsafe_allow_html=True)

    st.header("Olá, eu sou o Bob, o assistente do seu condomínio! :robot_face:")
    st.markdown("O meu objetivo é te apoiar na disponibilização de informações sobre manutenção, políticas de boa vizinhança e normas do condomínio.")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if st.session_state.conversation is None:
        with st.spinner("Carregando assistente..."):
            vector_embeddings = get_embeddings()
            st.session_state.conversation = start_conversation(vector_embeddings)
    
    query = st.chat_input("Qual é a sua dúvida?")

    if query:
        process_input(query)

if __name__ == "__main__":
    main()