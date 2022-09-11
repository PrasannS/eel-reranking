import flatten_lattice as fl
import torch
from bert_models import LinearPOSBertV1
from encoding_utils import *
import pickle
from mask_utils import *
import json
import os

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
bert_tok = fl.bert_tok
mbart_tok = fl.mbart_tok

# -1 for whole graph, otherwise # of lattice segments to use
STOPS = 30
# v3 - first whole lattice
# v4 - first single lattice
# v5 - single lattice with fixes
# v6 - 10 lattice segments
# v7 - updated whole lattice
# v8 - updated greedy 
# to make new one can change VNUM, STOPS, and make folder in torchsaved -> tmapsmaskedv{VNUM}
VNUM = 9
MOD_NAME = 'bertonewayv1.pth'

# specifies files for pre-loading
LOADED = {
    'amasks': 'attmasksallv'+str(VNUM)+'.pt',
    'tmaps': 'tmapsmaskedv'+str(VNUM)+'/'
}

# Code needed when changing set of gold labels
def prepare_dataset(resset):
    x = []
    y = []
    for res in resset:
        curinps = []
        curmasks = []
        for r in res:
            try:
                msk = torch.zeros(MAX_LEN, MAX_LEN)
                toktmp = torch.tensor(bert_tok(clean_expanded(r)).input_ids)
                msk[:len(toktmp), :len(toktmp)] = torch.ones(len(toktmp), len(toktmp))
                msk = torch.tril(msk)
                #msk = msk[:MAX_LEN, :MAX_LEN]
                #print(toktmp.shape)
                if float(toktmp.shape[0])<MAX_LEN:
                    dlen = MAX_LEN-toktmp.shape[0]
                    toktmp = torch.cat([toktmp, torch.zeros(dlen)])
                else:
                    toktmp = toktmp[:MAX_LEN]
                curinps.append(toktmp)
                curmasks.append(msk)
            except:
                print("weird error happened") 
        print(len(curinps))
        tinp = torch.stack(curinps).long().to(device)
        print(tinp.shape)
        # not taking in 1-way mask 
        y.append(posbmodel(tinp, attmasks=torch.stack(curmasks).long().to(device)))
        x.append(tinp)
        del tinp
        
    return x, y

def get_labset_partial(explodeds, startind, amt):
    dsetx, dsety = prepare_dataset(explodeds[startind:startind+amt])
    print(len(dsetx))
    assert len(dsetx)==amt
    latposylabels, tmaps = lattice_pos_goldlabels(dsetx, dsety, sents[startind:startind+amt])
    del dsetx, dsety
    
    torch.cuda.empty_cache()
    return latposylabels, tmaps

def get_biglabset(split):
    for i in range(0, int(len(resarrs)/split)):
        print("SUBSET - ", i)
        r, tmap = get_labset_partial(resarrs, i*split, split)
        torch.cuda.empty_cache()
        file = open('./torchsaved/tmapsmaskedv'+str(VNUM)+'/tmaps_'+str(i*split)+'.pkl', 'wb')
        # dump information to that file
        pickle.dump(tmap, file)

        # close the file
        file.close()
        del r
        del tmap
        torch.cuda.empty_cache()
        print(torch.cuda.memory_allocated("cuda:3"))

def load_model(labels):
    # load model, same for gold generation and inference
    posbmodel = LinearPOSBertV1(len(list(labels.keys())))
    t = torch.load("./a3distrib/ckpt/"+MOD_NAME)
    posbmodel.load_state_dict(t)
    posbmodel.eval()
    del t
    torch.cuda.empty_cache()

    print("GPU Mem Used = ", torch.cuda.memory_allocated("cuda:3"))

    return posbmodel

if __name__ == "__main__":
    # First we want to generate flattened version of the graph
    processedgraphs = fl.get_processed_graph_data(fl.frenbase, -1, STOPS)

    # get exploded candidates to generate gold labels
    resarrs = [fl.get_cover_paths(p)[0] for p in processedgraphs]

    # extra step for greedy 
    if STOPS==1:
        processedgraphs = filter_greedy(processedgraphs)

    # ensure no empty examples
    clean_empty(resarrs, processedgraphs)
    print("num examples: ", len(resarrs))

    # TODO should I add an example?

    # get attention masks, make if they don't exist allready
    if os.path.exists('./torchsaved/'+LOADED['amasks']):
        print("using loaded masks")
        attmasks = torch.load('./torchsaved/'+LOADED['amasks']).to(device)
    else:
        print("creating new masks")
        masktmp = [connect_mat(p) for p in processedgraphs]
        attmasks = torch.stack(masktmp).to(device)
        torch.save(attmasks, './torchsaved/'+LOADED['amasks'])

    # convert to backwards-only mask
    attmasks = torch.tril(attmasks)

    # credit to tutorial by https://pageperso.lis-lab.fr/benoit.favre/pstaln/09_embedding_evaluation.html for 
    # input / pre-processing setup
    # load labels
    with open('./a3distrib/lab_vocab.json') as json_file:
        labels = json.load(json_file)

    posbmodel = load_model(labels)

    # create inputs
    sents, posids = create_inputs(processedgraphs)

    print("Average, max nodes: ", avg_nodes(sents))

    # generate token label maps if they don't exist
    if os.path.exists('./torchsaved/'+LOADED['tmaps']+'tmaps_0.pkl')==False:
        with torch.no_grad():
            # used directly to generate gold labels
            get_biglabset(1)
    
    # load token label maps, use to get y-labels
    N_EX = 101
    tmaps = []
    for i in range(0, N_EX):
        with open('./torchsaved/'+LOADED['tmaps']+'tmaps_'+str(i)+'.pkl', 'rb') as file:
            tmaps.append(pickle.load(file)[0])

    # TODO y-labels are wrong currently, go back in to get rid of pad / subword stuff
    latposylabels = tmap_pos_goldlabels(tmaps, sents)    
    torch.cuda.empty_cache()

    # save data to run later
    outputdata = {}
    outputdata['tmaps'] = tmaps
    outputdata['masks'] = attmasks
    outputdata['pgraphs'] = processedgraphs

    with open('./torchsaved/outputv'+str(VNUM)+'.pkl', 'wb') as file:
        # dump information to that file
        pickle.dump(outputdata, file)

    # Make all predictions with ablations
    sents, posids = create_inputs(processedgraphs)
    pred1 = posbmodel(sents, mod_posids(posids), attmasks)
    a, ysmp, psmp = check_accuracy(pred1, latposylabels, sents)
    print(a)
    get_tmap_acc(ysmp, psmp, tmaps, sents)

    sents, posids = create_inputs(processedgraphs)
    pred2 = posbmodel(sents, fix_posids(posids), attmasks)
    a, ysmp, psmp = check_accuracy(pred2, latposylabels, sents)
    print(a)
    get_tmap_acc(ysmp, psmp, tmaps, sents)

    sents, posids = create_inputs(processedgraphs)
    pred3 = posbmodel(sents, mod_posids(posids), None)
    a, ysmp, psmp =check_accuracy(pred3, latposylabels, sents)
    print(a)
    get_tmap_acc(ysmp, psmp, tmaps, sents)

    sents, posids = create_inputs(processedgraphs)
    pred4 = posbmodel(sents, fix_posids(posids), None)
    a, ysmp, psmp =check_accuracy(pred4, latposylabels, sents)
    print(a)
    get_tmap_acc(ysmp, psmp, tmaps, sents)



