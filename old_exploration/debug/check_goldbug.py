import torch
import pickle

from encoding_utils import *
from flatten_lattice import *
from model_construct import *

tok = fl.bert_tok

# loads in odata with keys: tmaps, masks, pgraphs
with open('torchsaved/outputv5.pkl', 'rb') as file:
    odata = pickle.load(file)
    
with open('./a3distrib/lab_vocab.json') as json_file:
    labels = json.load(json_file)

def check_accuracy(setpred, setlabels, sents):
    # simplify prediction tensors
    ysimp = setlabels
    psimp = torch.argmax(setpred, dim=2)
    # clean up labels
    sm = subword_mask_all(sents)
    ysimp[sents==0] = 0
    ysimp[sents==102] = 0
    ysimp[sm==0] = 0
    ysimp[:, 0] = 0
    # apply cleanaup to x 
    psimp[ysimp==0] = 0
    # apply cleanaup to x 
    psimp[ysimp==0] = 0
    # compute accuracy
    acc = 1 - ((ysimp-psimp).count_nonzero())/ysimp.count_nonzero()
    return acc

