
#%%
import pandas as pd
from collections import Counter
from functools import reduce
df = pd.read_csv("data/datathon_data.csv")

#%% get ids of rare values
def get_1_values(s: pd.Series):
    # values which apear only once
    return [value for value, count in Counter(s).items() if count == 1]

def get_1_vals_per_col(df : pd.DataFrame, char_cols : list):
    # apply to each char col (except label)
    return {c: get_1_values(df.get(c)) for c in char_cols}

def get_ids_of_rare(df : pd.DataFrame, char_cols = ["WAERS", "BUKRS", "KTOSL", "PRCTR", "BSCHL", "HKONT"]):
    # get rare values
    rare_vals = get_1_vals_per_col(df, char_cols)
    # where are rare vals
    bool_series_list = [(df.get(n) == v) for n,vals in rare_vals.items() for v in vals]
    # where is any rare val
    combined_bool = reduce(lambda x, y: x | y, bool_series_list)
    # get id
    ids = combined_bool[combined_bool].index
    return ids
 
ids_of_rares = get_ids_of_rare(df)
# %% logically there should be a match. This also alligns with anomalies
ids_of_mismatch = df.loc[df["BUKRS"].str.slice(0,2) != df["WAERS"]].index
# %% combine id lists
easy_outliers = ids_of_mismatch.union(ids_of_rares)

# %%
