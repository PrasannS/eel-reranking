import torch

# TODO more sanity checking on whether or not we're recording the right number of branches

            
def get_adjac_mat(pgraph):
    res = []
    for node in pgraph:
        tmp = []
        for p in pgraph:
            if str(p['token_idx'])+" "+str(p['pos']) in node['nexts']:
                tmp.append(1)
            else:
                tmp.append(0)
        res.append(torch.tensor(tmp))
    return torch.stack(res)

def get_connected(adjac, n):
    res = adjac
    tot = res
    for i in range(0, n):
        res = torch.matmul(res, adjac)
        tot = res+tot
    return tot

def get_connected_inds(adjac, row, ind_dict):
    if str(row) in ind_dict:
        return ind_dict[str(row)]
    try:
        imnext = torch.nonzero(adjac[row])[0]
    except:
        print(row)
        return torch.tensor([0])
    if str(row) not in ind_dict:
        ind_dict[str(row)] = set()
    for im in imnext:
        if im not in ind_dict[str(row)]:
            ind_dict[str(row)].add(im)
            morecon = get_connected_inds(adjac, im, ind_dict)
            for m in morecon:
                ind_dict[str(row)].add(m)
    return ind_dict[str(row)]
    
def connected_manual(adjac):
    ind_dict = {}
    get_connected_inds(adjac, 0, ind_dict)
    return ind_dict

def matpow(adj, n):
    start = adj
    for i in range(n-1):
        start = torch.mm(start, adj)
    return start

def get_poslist(processed):
    res = []
    for p in processed:
        res.append(p['pos'])
    return res

def conmat(adj, longestpath):
    res = torch.zeros_like(adj)
    start = adj
    for i in range(0, longestpath):
        if i>0:
            start = torch.mm(start, adj)
        res = res + start
    return res

# fixes input to now account for [SEP] and [CLS] I believe
def correct_mask_sep(mat):
    for i in range(len(mat)):
        mat[i][i] = 1
    fixed = torch.nn.functional.pad(input=mat, pad=(1, 1, 1, 1), mode='constant', value=1)
    return fixed

# method that makes padding equal to 1
def ones_padding(msk):
    cop = msk
    tmp = cop[0]<=0
    lim = tmp.nonzero()
    if len(lim)==0:
        return cop
    limit = lim[0]
    #print(limit)
    for i in range(0, len(msk)):
        for j in range(limit, len(msk[0])):
            cop[i][j] = 1
            cop[j][i] = 1
    return cop

def connect_mat(pgraph):
    conm = conmat(get_adjac_mat(pgraph), max(get_poslist(pgraph)))
    conm = conm + torch.transpose(conm, 0, 1)

    cmat =  (conm>0).float()
    # handle things to be able to take when really big input
    respad = None
    MAX =512
    # we're within 498
    if len(cmat) < MAX-1:
        # get the version with CLS and SEP
        cmat = correct_mask_sep(cmat)
        # copy that onto fixed input: the rest is padding
        respad = torch.zeros((MAX, MAX))
        cpmax = len(cmat)
        for i in range(cpmax):
            for j in range(cpmax):
                respad[i][j] = cmat[i][j]
    else:
        # we're over the limit, first get truncated version
        respad = torch.zeros((MAX-2, MAX-2))
        # make truncated matrix
        for i in range(MAX-2):
            for j in range(MAX-2):
                respad[i][j] = cmat[i][j]
        # do CLS / SEP tokens on that 
        respad = correct_mask_sep(respad)
        
    respad[respad==0] = -float('inf')
    return ones_padding(respad)