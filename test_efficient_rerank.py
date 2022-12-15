from encode_utils.sco_funct import weightaddprob, default_scofunct
from encode_utils.efficient_rerank import get_effrerank_model, run_comstyle
from encode_utils.mt_scores import get_scores_auto
import pickle
import os
import pandas as pd
import torch
from generate_tables import metrics_mapping

# runs efficient rerank pipeline, with comet style model
def test_graph_ind(base, ind, model, funct, outfile):
    graph = pickle.load(open(base+str(ind), 'rb'))
    return {
        'hyp':run_comstyle(graph, model, funct, outfile, False)[0],
        'ref':graph['ref'],
        'src':graph['input']
    }

outbase = "outputs/predcsvs/"
graphsbase = "outputs/graph_pickles/"
def get_lattice_preds(gbase, funct, outfile, causal):
    # get a model to work with 
    if "noun" in outfile:
        encodemod = get_effrerank_model("noun")
    elif causal:
        encodemod = get_effrerank_model("comstyle")
    else:
        encodemod = get_effrerank_model("comnocause")
    # load some graphs
    allres = []
    l = len(os.listdir(gbase))
    # TODO remove later when load is lower
    l = 5
    for ind in range(l):
        print(ind)
        allres.append(test_graph_ind(gbase, ind, encodemod, funct, outfile))

    del encodemod
    torch.cuda.empty_cache()
    preddf = pd.DataFrame(allres)
    # save to a csv of the chosen name
    preddf.to_csv(outbase+outfile+".csv")
    return preddf

def get_act_hyps(hyplist, cutoff):
    res = []
    for h in hyplist:
        cind = h.index(cutoff)+len(cutoff)
        res.append(h[cind:].strip())
    return res

CAUSAL = True
# TODO do a script setup so that we can rapid-fire generate these
if __name__=="__main__":
    outname = "noun_comstyle_v2"
    graphsuff = "detest_reversed/"
    if os.path.exists(outbase+outname+".csv"):
        print("load from existing")
        gpreds = pd.read_csv(outbase+outname+".csv", index_col=0)
    else:
        # if we want to generate a new file
        # TODO may need to update score function
        gpreds = get_lattice_preds(graphsbase+graphsuff, default_scofunct, outname, CAUSAL)

    gpreds = gpreds.dropna()
    if "ahyp" not in gpreds.keys():
        gpreds['ahyp'] = get_act_hyps(gpreds['hyp'], "<s>")
    # now generate COMET scores 
    if "noun" in outname:
        gpreds['hyp'] = gpreds['ahyp']
        metrics_mapping("unique_nouns", gpreds)
        metrics_mapping("utnoun", gpreds)
        print("Unique Nouns "+str(sum(gpreds['unique_nouns'])/len(gpreds)))
    elif "comet" not in gpreds.keys():
        comsco = get_scores_auto(gpreds['ahyp'], gpreds['src'], gpreds['ref'], "comet", "")
        gpreds['comet'] = comsco
    
        print("COMET SCORE "+str(sum(gpreds['comet'])/len(gpreds)))
    # save final version
    gpreds.to_csv(outbase+outname+".csv")

########## fren_comstyle ############
# fren_comstyle_v1 - waddprob, comet trained model
# fren_comstyle_v2 - waddprob, comet trained model, zero-out padding gradients (not good) (.603)
# fren_comstyle_v3 - bring down weight to 20
# fren_comstyle_v4 - shrink down weight to 1 (trying to put them on a balanced scale) - works better (.659)
# fren_comstyle_v5 - shrink down weight to .5 (.654)
# fren_comstyle_v6 - try weight of 2 (.642)
# fren_comstyle_v7 - try weight of .7 (.667)
# fren_comstyle_v8 - try weight of .8 (.664)
# v9 - w .75 (.663)
# v10 - w .65 (.662)
########## ende_comstyle ############
# ende_comstyle_v1 - waddprob, comet trained model, zero-out padding gradients (still not good) 
# v2 - weight .72 - (.495)
# v3 - weight 2 - (.469)
# v4 - switch to position-id-relative average normalization, weight of 40 (.469)
# v5 - weight 150 (.456)
# v6 - weight 100 (.460)
# v7 - norm to adhoc as well (may just cancel stuff out), weight .8 (.498)
# v8 - weight 1.2 (.481)
# v9 - .92 weight (.488)
# v10 - 10 weight (.445)
# v11 - 1 weight (.484)
# v12 - 0.1 weight (not good)
# v13 - non-causal model with special mask, buggy forward mask, w .94 (.447)
# v14 - non-causal model with special mask, fixed forward mask, w .94 (.39)
# TODO do some numerical checks here, might just be using bad weights

########### noun_comstyle [on the whole thing] ############
# v1 - initial, forgot to put word "noun" as input
# v2 - input fixed, switch to single scofunct (8.52)
