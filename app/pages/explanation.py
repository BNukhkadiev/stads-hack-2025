import streamlit as st
import pandas as pd
from rag import RAG
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

system_prompt = "Input transaction ID to get explanation"
df_original = pd.read_csv("../data/datathon_data.csv")

# Initialize session state for OpenAI token and messages
if "openai_token" not in st.session_state:
    st.session_state.openai_token = ""
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": system_prompt}
    ]

# sidebar: Input OpenAI token
token_input = st.sidebar.text_input(
    "Enter your OpenAI API key",
    type="password",
    placeholder="sk-...",
    help="Provide your OpenAI API key to use the chatbot."
)

# sidebar: Input transaction ID
transaction_input = st.sidebar.text_input(
    "Enter your Transaction ID",
    type="default",
    placeholder="...",
    help="Provide transaction ID to explain."
)

if token_input:
    st.session_state.openai_token = token_input
    st.sidebar.success("API key saved!")

rag = RAG(index_path="index/refined_transaction_faiss.index", transaction_embeddings_path="weights/refined_transaction_embeddings.npy")
# Generate audit explanation using RAG
explanation = rag.generate_rag_from_id(transaction_input, df_original)

# Chat interface
st.title("RAG explanation")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if user_input := st.chat_input("Type your message here..."):
    if st.session_state.openai_token:

        # LLM initialization
        llm = ChatOpenAI(temperature=0, model="gpt-4o-mini", api_key=token_input)
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt), 
                MessagesPlaceholder(variable_name="messages"),
             ]
        )

        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)
        response = llm.invoke(prompt_template.invoke({"messages": st.session_state.messages}))
        msg = response.content
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("assistant").write(msg)

    else:
        st.error("Please enter your OpenAI API key in the sidebar.")