import argparse
import torch
from comet import download_model, load_from_checkpoint
import json
import pandas as pd

# cache dir for cometqe model
cometqe_dir = "./cometqemodel"
cometqe_model = "wmt20-comet-qe-da"
cometmodel = "wmt20-comet-da"
batch_size = 64

def process_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-candfile', type=str)
    parser.add_argument('-device', type=str, default='cuda:2')


    args = parser.parse_args()
    return args

def load_cands(fname):
    data = []
    with open(fname, 'r') as file:
        while True:
            line = file.readline()
            #print(line)
            # if line is empty
            # end of file is reached
            if not line or len(line)<3:
                break

            data.append(json.loads(line))
    return data

def process_cands(cand_data):
    refs = []
    hyps = []
    srcs = []
    clen = len(cand_data['scores'])
    # return refs, hyps, srcs
    return [cand_data['ref']]*clen, cand_data['cands'], [cand_data['inp']]*clen

def get_cometqe_scores(hyps, srcs):
    cometqe_input = [{"src": src, "mt": mt} for src, mt in zip(srcs, hyps)]
    # sentence-level and corpus-level COMET
    outputs = model.predict(
        cometqe_input, batch_size=40, progress_bar=True
    )
    torch.cuda.empty_cache()
    return outputs

def get_comet_scores(hyps, srcs, refs):
    cometqe_input = [{"src": src, "mt": mt, "ref":ref} for src, mt, ref in zip(srcs, hyps, refs)]
    # sentence-level and corpus-level COMET
    outputs = comet.predict(
        cometqe_input, batch_size=40, progress_bar=True
    )
    torch.cuda.empty_cache()
    return outputs

def test_cometqe(hyp, src):
    cqe_input = [{'src':src, 'mt':hyp}]
    outputs = model.predict(
        cqe_input, batch_size=1, progress_bar=True
    )
    torch.cuda.empty_cache()
    return outputs

rercomettmp = []
firstcomettmp = []
def comet_rerank_info(rrk_info):
    global rercomettmp
    global firstcomettmp
    tophyps = []
    rerhyps = []
    refs = []
    srcs = []
    for r in rrk_info:
        tophyps.append(r['topinitial'])
        rerhyps.append(r['reranked'])
        refs.append(r['ref'])
        srcs.append(r['src'])
    rerhyps = ['' if pd.isna(item) else item for item in rerhyps] 
    tophyps = ['' if pd.isna(item) else item for item in tophyps] 
    srcs = ['' if pd.isna(item) else item for item in srcs] 
    refs = ['' if pd.isna(item) else item for item in refs] 
    rer_comet_sco = get_comet_scores(rerhyps, srcs, refs)
    first_comet_sco = get_comet_scores(tophyps, srcs, refs)
    rercomettmp = rer_comet_sco
    firstcomettmp = first_comet_sco
    print("reranked comet score")
    print(rer_comet_sco[1])
    print("initial top1 comet score")
    print(first_comet_sco[1])
    return rer_comet_sco, first_comet_sco
    
def get_reranked_cands(c_data):
    global allscores
    reranked_info = []
    allhyps = []
    allsrcs = []
    
    for i in range(0, len(c_data)):
        r, h, s = process_cands(c_data[i])
        allhyps.extend(h)
        allsrcs.extend(s)

    allhyps = ['' if pd.isna(item) else item for item in allhyps] 
    allsrcs = ['' if pd.isna(item) else item for item in allsrcs] 
    allscores = get_cometqe_scores(allhyps, allsrcs)
    for i in range(0, len(c_data)):
        try:
            tmp = {}
            r, h, s = process_cands(c_data[i])
            tmp['ref'] = r[0]
            tmp['src'] = s[0]
            tmp['topinitial'] = h[0]
            #scoretmp = get_cometqe_scores(h, s)[0]
            #print(scoretmp)
            slen = len(c_data[i]['scores'])
            scoretmp = allscores[0][i*slen:slen*(i+1)]
            comind = scoretmp.index(max(scoretmp))
            tmp['reranked'] = h[comind]
            tmp['initialsco'] = scoretmp[0]
            tmp['rersco'] = scoretmp[comind]
            reranked_info.append(tmp)
        except:
            print("bug")
    return reranked_info

if __name__ == "__main__":
    args = process_args()
    args.device = args.device if torch.cuda.is_available() else "cpu"
    cand_data = load_cands("./candoutputs/"+args.candfile+".jsonl")

    
    cometqe_path = download_model(cometqe_model, cometqe_dir)
    model = load_from_checkpoint(cometqe_path)
    
    rerank_info = get_reranked_cands(cand_data)
    storererank = {}
    storererank['data'] = rerank_info
    f = open('latticererank.json', 'w')
    json.dump(storererank, f)
    del model
    del cometqe_path
    comet_path = download_model(cometmodel, "./cometmodel")
    comet = load_from_checkpoint(comet_path)
    rrinfo = comet_rerank_info(rerank_info)
    del comet
    del comet_path
        