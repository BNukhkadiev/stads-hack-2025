import pandas as pd
import streamlit as st

df = pd.read_csv("../data/datathon_data.csv")
# get stratified sample of 1000 rows from the dataset based on the 'label' column
df_subset = df.sample(n=1000, random_state=1)
df_subset = df[df['label'].isin(['anomal'])]
df_subset2 = df.head(1000)
df_subset = pd.concat([df_subset, df_subset2])

def highlight_anomal_rows(row):
    return ['background-color: red' if row['label'] == 'anomal' else '' for _ in row]

# Apply styling
styled_df = df_subset.style.apply(highlight_anomal_rows, axis=1)

# Display in Streamlit
st.dataframe(styled_df)
