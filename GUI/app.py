import streamlit as st

home = st.Page("pages/home.py", title="Home", icon=":material/home:", default=True)
explanation = st.Page("pages/explanation.py", title="Explanation", icon=":material/graphic_eq:")
chat = st.Page("pages/chat.py", title="Chat", icon=":material/sms:")

pg = st.navigation(
    [
        home,
        explanation,
        chat
    ]
)

pg.run()