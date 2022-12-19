import torch
import random

def bestprobsingle(mask, row, checknodes, mlen):
    bestnext = -1
    bestp = -1
    # use next with highest prob
    for n in checknodes:
        if n.canvpos<mlen: # keep within bounds
            if n.prob>bestp:
                bestnext = n.canvpos
                bestp = n.prob
    if bestnext>-1:
        mask[row][bestnext] = 1
    
def randomsingle(mask, row, checknodes, mlen):
    if row>0:
        avail = []
        # use next with highest prob
        for n in checknodes:
            if n.canvpos<mlen: # keep within bounds
                avail.append(n)

        mask[row][random.choice(avail).canvpos] = 1

# get adjacency from flat canvas of tokens, with previous tokens
def adj_mat(flat_canv, mlen, forward, single_cont, afunc = bestprobsingle):
    mask = torch.zeros((mlen, mlen))
    # have a cutoff after the calculated limit to circumvent extra computation
    # and potential bugs
    for row in range(mlen):
        if forward:
            qlist = flat_canv[row].nextlist
        else:
            qlist = flat_canv[row].prevs
        if single_cont:
            afunc(mask, row, qlist, mlen)
        else:
            # normal, just use all prevs that are valid
            for p in qlist:
                # TODO verify correctness of this
                if p.canvpos<mlen:
                    mask[row][p.canvpos] = 1
    return mask

# ensure that token canvas has the right values for mask creation
def mask_prep_canv(canv):
    maxpos = -1
    # reset all prevs based on nexts
    for i in range(len(canv)):
        node = canv[i]
        node.prevs = []
    for i in range(len(canv)):
        for n in canv[i].nextlist:
            assert n in canv
            n.prevs.append(canv[i])
    # we're taking in DLReverseNodes
    ind = 0
    for c in canv:
        # maybe should be an assert instead to ward off weirdness?
        c.canvpos = ind
        maxpos = max(maxpos, c.pos)
        # TODO only include as a test, remove later
        for prev in c.prevs:
            assert prev in canv
        ind+=1
    return maxpos  

MAXTOKS = 512
# get connectivity matrix from flat canvas of tokens
# by exponentiating reverse connected adjacency
def get_connect_mask(flat_canv, posadd, forward, params):
    # TODO is there an OBO here?
    mlen = min(MAXTOKS-(posadd), len(flat_canv))
    maxpos = mask_prep_canv(flat_canv)
    if 'afunc' in params.keys():
        # either get forwards or backwards only connectivity
        back_adjac = adj_mat(flat_canv, mlen, forward, True, params['afunc'])
    else:
        back_adjac = adj_mat(flat_canv, mlen, forward, True)

    tot = back_adjac
    tmp = back_adjac
    # keep on going until all nodes hit the back
    if False: # TODO made this from forward, not sure why other option won't work
        for i in range(maxpos+1):
            tmp = torch.mm(back_adjac, tmp)
            tot += tmp
    else:
        numrounds = 0
        # TODO there's a bug, but just run results for now
        while torch.sum((tot[:, 0]>0))<(mlen-1) and numrounds<512:
            tmp = torch.mm(back_adjac, tmp)
            tot += tmp
            numrounds+=1
    tot = tot+ torch.eye(mlen)
    return (tot>0).int()

def get_causal_mask(flatcanv, posadd, params, addforward=False):
    custmax = get_connect_mask(flatcanv, posadd, False, params)
    if addforward:
        custmax = custmax + get_connect_mask(flatcanv, posadd, True, params)
        custmax = (custmax>0).int()
    # everything starts off as 1s
    res = torch.ones((MAXTOKS, MAXTOKS))
    # causal on the right side
    res[:, posadd:] = 0
    addpos = len(custmax)+posadd
    res[posadd:addpos, posadd:addpos] = custmax
    return res

    