import streamlit as st

home = st.Page("pages/home.py", title="Home", icon=":material/home:", default=True)
visualization = st.Page("pages/visualization.py", title="Visualization", icon=":material/graphic_eq:")
chat = st.Page("pages/chat.py", title="Chat", icon=":material/sms:")

pg = st.navigation(
    [
        home,
        visualization,
        chat
    ]
)

pg.run()