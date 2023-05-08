from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
import csv
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
import sys
import random
import pandas as pd
from rerank_score_cands_new import load_cands
import numpy as np
from comet import download_model, load_from_checkpoint
import pickle

csv.field_size_limit(sys.maxsize)

#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#torch.cuda.set_device(1)

cometmodel = "wmt20-comet-da"
comet_path = download_model(cometmodel, "./cometmodel")
comet = load_from_checkpoint(comet_path).to(device)

def make_com_inputs(cands):
    alldata = []
    for c in cands:
        ind = 0
        for cand in c['cands']:
            tmp = {}
            tmp['inp'] = c['src']
            tmp['ref'] = c['ref']
            tmp['hyp'] = cand
            tmp['scores'] = c['cometscores'][ind]
            alldata.append(tmp)
            ind+=1
    return alldata

def get_comet_scores(hyps, srcs, refs):
    cometqe_input = [{"src": src, "mt": mt, "ref":ref} for src, mt, ref in zip(srcs, hyps, refs)]
    # sentence-level and corpus-level COMET
    print(comet.predict(
        cometqe_input[:128], batch_size=128, progress_bar=True,
        gpus=1
    ))
    outs = comet.predict(
        cometqe_input, batch_size=128, progress_bar=True,
        gpus=1
    )
    return outs[0]

#frdata = pd.read_csv("processeddata/cpfdata1.csv")
#frdata = frdata.dropna()

#sco = get_comet_scores(frdata['hyp'], frdata['src'], frdata['ref'])
#frdata['score'] = sco
#frdata.to_csv("processeddata/cpfdata1.csv")

gdata = pd.read_csv("processeddata/cpgdata.csv")

gdata = gdata.dropna()

sco = get_comet_scores(gdata['hyp'], gdata['src'], gdata['ref'])
gdata['score'] = sco
gdata.to_csv("processeddata/cpgdata1.csv")

 # nohup CUDA_VISIBLE_DEVICES=1 python comscodata.py &