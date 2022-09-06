import torch
from scipy.spatial.distance import cosine
from more_itertools import locate
from flatten_lattice import bert_tok

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

MAX_LEN = 500

def find_indices(list_to_check, item_to_find):
    indices = locate(list_to_check, lambda x: x == item_to_find)
    return list(indices)

def clean_expanded(cand):
    tmp = cand.replace("</s>", "")
    tmp = tmp.replace("en_XX", "")
    tmp = tmp.replace("<s>", "")
    return tmp.strip()
            
MAXLEN = 500
def create_inputs(pgraphs):
    result_tok = []
    result_pos = []
    for p in pgraphs:
        tokstmp = []
        postmp = []
        for tok in p:
            tokstmp.append(tok['token_idx'])
            postmp.append(tok['pos']+1)
        if len(tokstmp)>=MAX_LEN-2:
            tokstmp = tokstmp[:MAXLEN-2]
            postmp = postmp[:MAXLEN-2]
        # ACK how could I miss this...
        tokstmp = [101] + tokstmp + [102]
        rem = MAX_LEN - len(tokstmp)
        postmp = [0] + postmp + [max(postmp)+1] + [0]*rem
        tokstmp = tokstmp + [0]*rem
        result_tok.append(tokstmp)
        result_pos.append(postmp)
        
    return torch.tensor(result_tok).to(device), torch.tensor(result_pos).to(device)

def dataset_make_tags(inpx, inpy):
    tokmap = {}
    print(len(inpx))
    for i in range(0, len(inpx)):
        inpi = inpx[i]
        for j in range(0, len(inpi)):
            k = str(int(inpi[j]))
            if k not in tokmap:
                tokmap[k]=[]
            tokmap[k].append(inpy[i][j])
    for k in tokmap.keys():
        
        tokmap[k] = torch.stack(tokmap[k]).to(device)
        tokmap[k] = torch.mean(tokmap[k], dim=0)
        #print(tokmap[k].shape)
    return tokmap
    
def makelattice_pos_data(tokmap, flat):
    res = []
    numlabs = len(tokmap['101'])
    for tok in flat:
        # this should always work given how the lattice is structured
        if str(int(tok)) in tokmap:
            res.append(tokmap[str(int(tok))].to(device))
        else:
            print("missing token")
            res.append(torch.zeros(numlabs).to(device))
    return torch.stack(res).float()

def lattice_pos_goldlabels(datax, datay, sents):
    dataset = []
    tmaps = []
    for i in range(len(datax)):
        tmap = dataset_make_tags(datax[i], datay[i])
        dataset.append(makelattice_pos_data(tmap, sents[i]))
        tmaps.append(tmap)
        print(i)
    return torch.stack(dataset).float().to(device), tmaps

def tmap_pos_goldlabels(tmaps, sents):
    dataset = []
    assert len(tmaps)==len(sents)
    for i in range(len(tmaps)):
        dataset.append(makelattice_pos_data(tmaps[i], sents[i]))
        print(i)
    return torch.stack(dataset).float().to(device)

soft = torch.nn.Softmax(dim=2)
loss = torch.nn.BCEWithLogitsLoss()
l1 = torch.nn.L1Loss()
mse = torch.nn.MSELoss()