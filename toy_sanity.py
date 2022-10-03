from src.recom_search.model.beam_node_reverse import ReverseNode
from transformers import AutoTokenizer, AutoModel

import flatten_lattice as fl
import torch
from bert_models import LinearLatticeBert, LinearPOSBert
from encoding_utils import *
import pickle
import toy_helper as thelp

import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from latmask_bert_models import LatticeBertModel
import json


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

from mask_utils import *
from encoding_utils import *


bert_tok = AutoTokenizer.from_pretrained("bert-base-cased")
mbart_tok = AutoTokenizer.from_pretrained("facebook/mbart-large-50-many-to-one-mmt")

# Model Wrapper
class LinearPOSBertV1(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.bert = LatticeBertModel(AutoConfig.from_pretrained('bert-base-cased'))
        self.probe = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.to(device)

    def parameters(self):
        return self.probe.parameters()
  
    def forward(self, sentences, pos_ids=None, attmasks=None):
        with torch.no_grad(): # no training of BERT parameters
            word_rep, sentence_rep = self.bert(sentences, position_ids=pos_ids, encoder_attention_mask=attmasks, attention_mask=attmasks, return_dict=False)
        return self.probe(word_rep)
    
def prepare_dataset(resset):
    x = []
    y = []
    for res in resset:
        
        cleaned = [clean_expanded(r) for r in res]
        inputs = bert_tok(cleaned, padding="max_length", max_length=500, return_tensors='pt').to(device)

        y.append(posbmodel(inputs.input_ids, attmasks = inputs.attention_mask))
        x.append(inputs.input_ids)
        
    return x, y

def check_accuracy(setpred, setlabels):
    cor = 0
    tot = 0
    for i in range(0, len(setpred)):
        ex = setpred[i]
        for j in range(0, len(ex)):
            if sum(setlabels[i][j])==0:
                continue
            elif torch.argmax(setlabels[i][j])==0:
                continue
            tot+=1
            if torch.argmax(ex[j])==torch.argmax(setlabels[i][j]):
                cor+=1
    return cor/tot

# correct posids
def mod_posids(pids):
    cop = pids
    for p in cop:
        for i in range(0, len(p)):
            if p[i]==0:
                p[i] = i
    return cop

# set posids to default
def def_posids(pids):
    cop = pids
    for p in cop:
        for i in range(0, len(p)):
            p[i] = i
    return cop

def show_labels (pred):
    res = []
    for p in pred:
        res.append(lablist[torch.argmax(p)])
    return res

# Load POS model, label vocabulary 
with open('./lab_vocab.json') as json_file:
    labels = json.load(json_file)
posbmodel = LinearPOSBertV1(len(list(labels.keys())))    
t = torch.load("./a3distrib/ckpt/posbert1way.pth")
posbmodel.load_state_dict(t)
posbmodel.eval()
print(torch.cuda.memory_allocated("cuda:1"))
torch.cuda.empty_cache()

# method that makes padding equal to 1
from mask_utils import ones_padding

def run_pipeline(inplist, resarrs = None, flat = None):
    # construct data structure for toy graph in format used on actual examples
    if flat==None:
        toygraph = thelp.create_toy_graph(inplist, mbart_tok)

        # get list of exploded candidates using same algorithm from numbers
        exploded = fl.get_all_possible_candidates(toygraph)

        # get a flattened version of toy lattice (same method as on actual examples)
        flat_toy = fl.flatten_lattice(toygraph)
    else:
        flat_toy = flat
        exploded = resarrs

    # generate mask (uses same method as actual examples), convert to -inf mask (seems to not do anything)
    mask = connect_mat(flat_toy)
    mask = torch.triu(mask)
    #mask[mask==0] = -float('inf')
    #mask = ones_padding(mask)
    

    # get gold labels for the exploded set
    dsetx, dsety = prepare_dataset([exploded])

    assert len(dsetx)==1

    # from encoding utils, get posids and relevant tokens
    sents, posids = create_inputs([flat_toy])
    
    # get gold label dictionaries for tokens in example, based on averages of tokens on dsety
    _ , tmaps = lattice_pos_goldlabels(dsetx, dsety, sents)

    # generate gold y labels using tmaps and 
    latposylabels = tmap_pos_goldlabels(tmaps, sents)

    # get generated labels for flattened lattice, def_posids can be used for default posids
    # params start as (sents.to(device), mod_posids(posids).to(device), torch.stack([mask]).to(device))
    # posids, mask can be set to None to ablate to default
    pred = posbmodel(sents.to(device), mod_posids(posids).to(device), torch.stack([mask]).to(device))
    #pred = posbmodel(sents.to(device), None, None)
    return pred, latposylabels, tmaps, sents, posids, dsetx, dsety, flat_toy, mask

lablist = [k for k in labels.keys()]
def print_results(CUTOFF):

    # sanity check to look at flat lattice 
    p = flat_toy
    tlist = fl.get_toklist(p)
    res = ""
    for s in tlist:
        res = res+" "+bert_tok.decode(s)
    decstr = res

    # number of tokens, the tokens that are passed into model for lattice
    print("INPUT")
    print(decstr)

    print("PREDICTED")
    print(show_labels(pred[0])[:CUTOFF])
    print("GOLD")
    print(show_labels(latposylabels[0])[:CUTOFF])
    
    # run explodeds through model
    indivlabs = posbmodel(dsetx[0])
    print("")
    print("Exploded paths")
    # show labels for s1, s2 when run through individually
    for i in range(len(inputlist)):
        print(inputlist[i])
        print(show_labels(indivlabs[i])[:20])
        
inputlist = [
    "I like you",
    "I like him",
]

# START SANITY CHECKING CODE

# construct data structure for toy graph in format used on actual examples
toygraph = thelp.create_toy_graph(inputlist, mbart_tok)

# get list of exploded candidates using same algorithm from numbers
exploded = fl.get_all_possible_candidates(toygraph)

# get a flattened version of toy lattice (same method as on actual examples)
flat_toy = fl.flatten_lattice(toygraph)

# generate mask (uses same method as actual examples), convert to -inf mask (seems to not do anything)
mask = connect_mat(flat_toy)
#mask[mask==0] = -float('inf')
#mask = ones_padding(mask)


# get gold labels for the exploded set
dsetx, dsety = prepare_dataset([exploded])

assert len(dsetx)==1

# from encoding utils, get posids and relevant tokens
sents, posids = create_inputs([flat_toy])

# get gold label dictionaries for tokens in example, based on averages of tokens on dsety
_ , tmaps = lattice_pos_goldlabels(dsetx, dsety, sents)

# generate gold y labels using tmaps and 
latposylabels = tmap_pos_goldlabels(tmaps, sents)

# get generated labels for flattened lattice, def_posids can be used for default posids
# params start as (sents.to(device), mod_posids(posids).to(device), torch.stack([mask]).to(device))
# posids, mask can be set to None to ablate to default
pred = posbmodel(sents.to(device), mod_posids(posids).to(device), torch.stack([mask]).to(device))#, mod_posids(posids).to(device), torch.stack([mask]).to(device))
#pred = posbmodel(sents.to(device), mod_posids(posids).to(device), None)

# 1-way sanity check

def subset(inmasks, sets, pids, newdim):
    # switch to tril
    inmasks = torch.triu(inmasks)
    m = torch.stack([inmasks])[:, :newdim, :newdim].to(device)
    s = sets[:, :newdim].to(device)
    p = mod_posids(pids)[:, :newdim].to(device)
    return m, s, p

def get_norminp(inputsents, lim):
    isents = torch.stack([inputsents[:lim]]).to(device)
    ipids = torch.stack([torch.arange(0, lim)]).to(device)
    # switch to torch.tril
    imasks = torch.stack([torch.triu(torch.ones((lim, lim)))]).to(device)
    return isents, ipids, imasks

loss = torch.nn.MSELoss()

# get with lattice
slmask, slsent, slpid = subset(mask, sents, posids, 6)
# get with other
pred = posbmodel(slsent, slpid, slmask)#, mod_posids(posids).to(device), torch.stack([mask]).to(device))
# get with normal model (1 example)
ns, np, nm = get_norminp(dsetx[0][0], 5)
normpred = posbmodel(ns, np, nm)

loss(normpred, pred[:, :5])