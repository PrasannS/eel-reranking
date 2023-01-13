from tkinter.tix import MAX
import torch
from scipy.spatial.distance import cosine
from more_itertools import locate
from .new_flatten_lattice import detok

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

MAX_LEN =512

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
            
MAXLEN = 512
# posids are the first posids to be set for each value
# assume everything has same size
def create_inputs(pgraphs):
    result_tok = []
    result_pos = []
    for p in pgraphs:
        # TODO got rid of extra padding, does this change anything?
        tokstmp = [0]*min(MAX_LEN, len(pgraphs[0]))
        postmp = [0]*min(MAX_LEN, len(pgraphs[0]))
        idl = [t['id'] for t in p]
        ind = 0
        for tok in p:
            if ind==MAXLEN:
                break
            tokstmp[ind] = tok['token_idx']
            # TODO this is a bit janky
            for n in tok['nexts']:
                try:
                    if postmp[idl.index(n)]==0:
                        postmp[idl.index(n)] = min(tok['pos'], postmp[ind])+1
                except:
                    ""
            ind+=1
        
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
        tmp = [detok.decode(s) for s in sents[i]]
        msk[i] = submask_help(tmp)
    return msk

# make tags given input set and predicted labels
# inpx.shape = (nex,512); inpy.shape = (nex,512, nlabls)
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
        tmp = [detok.decode(s) for s in sents[i]]
        msk[i] = submask_help(tmp)
    return msk

def check_accuracy(setpred, setlabels, sents):
    # simplify prediction tensors
    ysimp = torch.clone(setlabels)
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
    return acc, ysimp, psimp

def get_tmap_acc(ysmp, psmp, tmaps, sents):
    assert len(tmaps)==len(ysmp)
    cor = 0
    tot = 0
    errinds = 0
    errtot = 0
    for a in ysmp.nonzero():
        if int(psmp[a[0], a[1]]) in tmaps[a[0]][str(int(sents[a[0], a[1]]))]:
            #print(int(ysmp[a[0], a[1]]), " ",  tmaps[a[0]][str(int(sents[a[0], a[1]]))])
            cor+=1
        else:
            #print(int(ysmp[a[0], a[1]]))
            #print(int(psmp[a[0], a[1]]), " ",tmaps[a[0]][str(int(sents[a[0], a[1]]))] )
            ""
            errinds+=a[1]
            errtot+=1
        tot+=1
    print("TMAP ACC: ", cor/tot, " TOT: ", tot)
    print("ERR AVG IND: ", errinds/errtot)
    return cor/tot

soft = torch.nn.Softmax(dim=2)
loss = torch.nn.BCEWithLogitsLoss()
l1 = torch.nn.L1Loss()
mse = torch.nn.MSELoss()