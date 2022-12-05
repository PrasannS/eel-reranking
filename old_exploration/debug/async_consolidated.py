from efficient_rerank import run_pipeline, XLMCometEmbeds
import torch
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
import pickle
import pandas as pd
import os
from src.recom_search.model.beam_node_reverse import ReverseNode
import numpy as np
import math 

base = "frtest_reversed/"
def test_graph_ind(ind, basedir):
    g = pickle.load(open(basedir+str(ind), 'rb'))
    #if g['input'] in old['src']:
    #    return None, None, None
    #try:
    return g['input'], g['ref'], run_pipeline(g, model)
    
def get_all_preds(basedir):
    l = len(os.listdir(basedir))
    res = []
    print(l)
    for i in range(l):
        try:
            i, r, p = test_graph_ind(i, basedir)
            res.append({
                'src':i,
                'hyp':p,
                'ref':r
            })
            print(i)
        except:
            print("FAILURE")
            res.append({
                'src':None,
                'hyp':None,
                'ref':None
            })
    res = pd.DataFrame(res)
    res.to_csv("latfound"+basedir[:-4]+".csv")
    return res

if __name__ == "__main__":
    # load in model, for french
    #del model
    model = XLMCometEmbeds(drop_rate=0.1)
    model.load_state_dict(torch.load("./torchsaved/maskedcont4.pt"))
    model.eval()
    torch.cuda.memory_allocated(device)
    frpreds = get_all_preds("frtest_reversed/")