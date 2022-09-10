import torch
from scipy.spatial.distance import cosine
from more_itertools import locate
from flatten_lattice import bert_tok

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

MAX_LEN = 500

def avg_nodes(sentences):
    zcnts = [500-list(sent).count(0) for sent in sentences]
    return sum(zcnts)/len(zcnts), max(zcnts)

def find_indices(list_to_check, item_to_find):
    indices = locate(list_to_check, lambda x: x == item_to_find)
    return list(indices)

def clean_expanded(cand):
    tmp = cand.replace("</s>", "")
    tmp = tmp.replace("en_XX", "")
    tmp = tmp.replace("<s>", "")
    return tmp.strip()

def filter_greedy(pgraphs):
    res = []
    for p in pgraphs:
        tmp = []
        for i in range(1, len(p)):
            tmp.append(p[i-1])
            if p[i]['pos']<=p[i-1]['pos']:
                break
            elif i==len(p)-1:
                tmp.append(p[i])
        res.append(tmp)
    return res

def clean_empty(rarrs, pgraphs):
    # clean up empty examples
    for r in range(0, len(rarrs)):
        try:
            if len(rarrs[r])==0:
                print(r)
                del rarrs[r]
                del pgraphs[r]
        except:
            break
    assert len(rarrs)==len(pgraphs)
            
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

def submask_help(words):
    msk = []
    for i in range(len(words)-1):
        if "#" in words[i+1]:
            msk.append(0)
        else:
            msk.append(1)
    msk.append(1)
    return torch.tensor(msk)

def subword_mask_all (sents):
    msk = torch.ones_like(sents)
    for i in range(0, len(sents)):
        tmp = [bert_tok.decode(s) for s in sents[i]]
        msk[i] = submask_help(tmp)
    return msk

# make tags given input set and predicted labels
# inpx.shape = (nex, 500); inpy.shape = (nex, 500, nlabls)
# return (dict with list of label ids for each token)
def dataset_make_tags(inpx, inpy):
    tokmap = {}
    print(len(inpx))
    # first process y labels
    ysimp = torch.argmax(inpy, dim=2)
    # clean up labels
    sm = subword_mask_all(inpx)
    ysimp[inpx==0] = 0
    ysimp[inpx==102] = 0
    ysimp[sm==0] = 0
    ysimp[:, 0] = 0
    # apply cleanaup to x 
    assert ysimp.shape == inpx.shape
    # for each unique token, add unique labels across exploded 
    for i in range(0, len(inpx)):
        inpi = inpx[i]
        for j in range(0, len(inpi)):
            k = str(int(inpi[j]))
            if k not in tokmap:
                tokmap[k]=set()
            tokmap[k].add(int(ysimp[i][j]))
    return tokmap

# make y-label tensor given tokmap, and flat
# TODO may be a more complex way to get better accuracy
def makelattice_pos_data(tokmap, flat):
    res = []
    for tok in flat:
        # this should always work given how the lattice is structured
        if str(int(tok)) in tokmap:
            res.append(list(tokmap[str(int(tok))])[0])
        else:
            print("missing token")
            res.append(0)
    return torch.tensor(res)

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
        #print(i)
    return torch.stack(dataset).float().to(device)

def mod_posids(pids):
    cop = pids
    for p in cop:
        for i in range(0, len(p)):
            if i>0 and p[i]==0:
                p[i] = i
    return cop

def fix_posids(pids):
    cop = pids
    for p in cop:
        for i in range(0, len(p)):
            p[i] = i
    return cop

def submask_help(words):
    msk = []
    for i in range(len(words)-1):
        if "#" in words[i+1]:
            msk.append(0)
        else:
            msk.append(1)
    msk.append(1)
    return torch.tensor(msk)

def subword_mask_all (sents):
    msk = torch.ones_like(sents)
    for i in range(0, len(sents)):
        tmp = [bert_tok.decode(s) for s in sents[i]]
        msk[i] = submask_help(tmp)
    return msk

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

soft = torch.nn.Softmax(dim=2)
loss = torch.nn.BCEWithLogitsLoss()
l1 = torch.nn.L1Loss()
mse = torch.nn.MSELoss()