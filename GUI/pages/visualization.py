import pandas as pd
import streamlit as st

df = pd.read_csv("data/datathon_data.csv")

def highlight_anomal(val):
    color = "red" if val == "anomal" else "black"
    return f"color: {color}; font-weight: bold;"


# Display in Streamlit
st.dataframe(df)
#st.write(df.style.applymap(highlight_anomal, subset=["label"]))