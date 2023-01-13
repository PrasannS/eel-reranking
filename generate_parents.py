import pandas as pd
import pickle
from parent_explore.stagewise_finetune.parent_master.parent import parent_score_df

if __name__=="__main__":
    base = "outputs/graph_pickles/exploded_tabtotext_beam12/"

    ind = 0
    with open(base+str(ind), "rb") as file:
        res = pickle.load(file)

    scos, cands, ref, src = res
    newdf = pd.DataFrame()
    newdf['hyp'] = cands
    newdf['ref'] = ref
    newdf['src'] = src

    result = parent_score_df(newdf, "b12_e0.csv")