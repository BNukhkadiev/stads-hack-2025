import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

system_prompt = "You explain anomalies in a dataset to an auditor based on the rules we set."

# Initialize session state for OpenAI token and messages
if "openai_token" not in st.session_state:
    st.session_state.openai_token = ""
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": system_prompt}
    ]

# Sidebar: Input OpenAI token
st.sidebar.title("OpenAI API Key")
token_input = st.sidebar.text_input(
    "Enter your OpenAI API key",
    type="password",
    placeholder="sk-...",
    help="Provide your OpenAI API key to use the chatbot."
)

if token_input:
    st.session_state.openai_token = token_input
    st.sidebar.success("API key saved!")

# Chat interface
st.title("Chatbot")

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