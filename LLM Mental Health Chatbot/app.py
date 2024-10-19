import os
import streamlit as st
from prompt import PROMPT
from dotenv import load_dotenv
from langchain.llms import GooglePalm
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.embeddings import GooglePalmEmbeddings


load_dotenv()
st.title('LLM Mental Health Chatbot')
st.write('Hello! How can I assist you today?')


def process_qa_retrieval_chain(chain, query):
    response = chain.invoke({'query': query})

    result_str = f'Query: {response["query"]}\n\n'
    result_str += f'Result: {response["result"]}\n\n'
    relevant_docs = response['source_documents']

    for i in range(len(relevant_docs)):
        result_str += f'Relevant Doc {i+1}:\n'
        result_str += relevant_docs[i].page_content + '\n'
        result_str += str(relevant_docs[i].metadata) + '\n\n'

    return result_str


def chatbot():
    api_key = os.getenv('GOOGLE_API_KEY')
    llm = GooglePalm(google_api_key=api_key, temperature=0)
    embedding = GooglePalmEmbeddings(google_api_key=api_key)

    vectordb = Chroma(
        persist_directory='../Data/vector_db/chroma_pdfs',
        embedding_function=embedding
    )

    if 'conversation' not in st.session_state:
        st.session_state.messages = []

    user_input = st.text_input("You: ", "")

    if st.button("Send") and user_input:
        st.session_state.messages.append(f"You: {user_input}")

        qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=vectordb.as_retriever(),
            return_source_documents=True,
            chain_type_kwargs={
                'prompt': PromptTemplate.from_template(PROMPT)
            }
        )
        result = process_qa_retrieval_chain(qa_chain, user_input)
        st.session_state.messages.append(f"Bot: {result}")

    for message in st.session_state.messages:
        st.write(message)


if __name__ == "__main__":
    chatbot()
