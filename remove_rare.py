
#%%
import pandas as pd
from collections import Counter
from functools import reduce
df = pd.read_csv("data/datathon_data.csv")

#%%
def get_1_values(s: pd.Series):
    return [value for value, count in Counter(s).items() if count == 1]

def get_1_vals_per_col(df : pd.DataFrame, char_cols : list):
    return {c: get_1_values(df.get(c)) for c in char_cols}

def get_ids_of_rare(df : pd.DataFrame, char_cols = ["WAERS", "BUKRS", "KTOSL", "PRCTR", "BSCHL", "HKONT"]):
    rare_vals = get_1_vals_per_col(df, char_cols)
    bool_series_list = [(df.get(n) == v) for n,vals in rare_vals.items() for v in vals]
    combined_bool = reduce(lambda x, y: x | y, bool_series_list)
    ids = combined_bool[combined_bool].index
    return ids
 
ids_of_rares = get_ids_of_rare(df)
print(ids_of_rares)

# %%
