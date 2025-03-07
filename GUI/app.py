import streamlit as st

home = st.Page("pages/home.py", title="Home", icon=":material/home:", default=True)
explanation = st.Page("pages/explanation.py", title="explanation", icon=":material/sms:")

pg = st.navigation(
    [
        home,
        explanation
    ]
)

pg.run()