import argparse
import torch
from comet import download_model, load_from_checkpoint
import json
import pandas as pd
from os.path import exists
import numpy
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# cache dir for cometqe model
cometqe_dir = "./cometqemodel"
# can alternatively use wmt21-comet-qe-mqm
cometqe_model = "wmt20-comet-qe-da"
cometmodel = "wmt20-comet-da"
batch_size = 64

def process_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-candfile', type=str, default='beam40en_de')
    parser.add_argument('-device', type=str, default='cuda:2')
    parser.add_argument('-oracle', type=str, default='no')


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
    
    clen = len(cand_data['scores'])
    # return refs, hyps, srcs
    return [cand_data['ref']]*clen, cand_data['cands'], [cand_data['inp']]*clen

def get_cometqe_scores(hyps, srcs):
    cometqe_input = [{"src": src, "mt": mt} for src, mt in zip(srcs, hyps)]
    # sentence-level and corpus-level COMET
    outputs = model.predict(
        cometqe_input, batch_size=32, progress_bar=True
    )
    torch.cuda.empty_cache()
    return outputs

def get_comet_scores(hyps, srcs, refs):
    cometqe_input = [{"src": src, "mt": mt, "ref":ref} for src, mt, ref in zip(srcs, hyps, refs)]
    # sentence-level and corpus-level COMET
    outputs = comet.predict(
        cometqe_input, batch_size=32, progress_bar=True
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

def get_average_score(cand_data, resco):
    scoresum = 0
    scoretot = 0
    s = "scores"
    if resco:
        s = "rescores"
    for c in cand_data:
        scoresum+=sum(c[s])
        scoretot+=len(c[s])
    return scoresum/scoretot

def amax_score(cand_data):
    scoresum = 0
    scoretot = 0
    for c in cand_data:
        if len(c['scores'])==0:
            continue
        scoresum+=max(c['scores'])
        scoretot+=1
    return scoresum/scoretot

def get_average_candlen(cand_data):
    scoresum = 0
    scoretot = 0
    empty = 0
    for c in cand_data:
        if len(c['scores'])==0:
            empty+=1
            continue
        scoresum+=len(c['scores'])
        scoretot+=1
    print("Are empty :"+str(empty))
    return scoresum/scoretot

def comet_rerank_info(rrk_info):
    
    tophyps = []
    comhyps = []
    qehyps = []
    refs = []
    srcs = []
    for r in rrk_info:
        tophyps.append(r['topinitial'])
        if cheat=='yes' or cheat=='both':
            comhyps.append(r['comranked'])
        if cheat=='no' or cheat=='both':
            qehyps.append(r['qeranked'])
        refs.append(r['ref'])
        srcs.append(r['src'])
    if cheat=='yes' or cheat=='both':
        comhyps = ['' if pd.isna(item) else item for item in comhyps] 
    if cheat=='no' or cheat=='both':
        qehyps = ['' if pd.isna(item) else item for item in qehyps] 
    tophyps = ['' if pd.isna(item) else item for item in tophyps] 
    srcs = ['' if pd.isna(item) else item for item in srcs] 
    refs = ['' if pd.isna(item) else item for item in refs] 
    comsco = 0
    qesco = 0
    if cheat=='yes' or cheat=='both':
        comsco = get_comet_scores(comhyps, srcs, refs)
    if cheat=='no' or cheat=='both':
        qesco = get_comet_scores(qehyps, srcs, refs)
    first_comet_sco = get_comet_scores(tophyps, srcs, refs)
    
    result = {}
    print("oracle")
    print(comsco[1])
    print("cometqe")
    print(qesco[1])
    print("top1")
    print(first_comet_sco[1])
    result['oracle'] = comsco
    result['qererank'] = qesco
    result['topinitial'] = first_comet_sco
    #print(result)
    return result
    
def get_reranked_cands(c_data):
    global comet_path, comet, cometqe_path, model
    reranked_info = []
    allhyps = []
    allsrcs = []
    allrefs = []
    for i in range(0, len(c_data)):
        r, h, s = process_cands(c_data[i])
        # TODO, should remove empty candidates, but may introduce bugs
        #if len(h)==0:
        #    continue
        allhyps.extend(h)
        allsrcs.extend(s)
        allrefs.extend(r)
    allhyps = ['' if pd.isna(item) else item for item in allhyps] 
    allsrcs = ['' if pd.isna(item) else item for item in allsrcs] 
    allrefs = ['' if pd.isna(item) else item for item in allrefs] 
    
    if cheat=='no' or cheat=='both':
        cometqe_path = download_model(cometqe_model, cometqe_dir)
        model = load_from_checkpoint(cometqe_path)
        qescores = get_cometqe_scores(allhyps, allsrcs)
        del cometqe_path
        del model
    if cheat=='yes' or cheat=='both':
        comet_path = download_model(cometmodel, "./cometmodel")
        comet = load_from_checkpoint(comet_path)
        comscores = get_comet_scores(allhyps, allsrcs, allrefs)
        del comet_path
        del comet
    
    assert len(qescores)==len(comscores)
    start = 0
    for i in range(0, len(c_data)):
        try:
            tmp = {}
            r, h, s = process_cands(c_data[i])
            tmp['ref'] = r[0]
            tmp['src'] = s[0]
            # used rescored best
            if rescore:
                # this is a loss, so lower is better?
                tmp['topinitial'] = h[c_data[i]['sco_ranks'][0]]
            else:
                tmp['topinitial'] = h[0]
            tmp['cands'] = h
            tmp['modelscores'] = c_data[i]['scores']
            #scoretmp = get_cometqe_scores(h, s)[0]
            #print(scoretmp)
            slen = len(c_data[i]['scores'])
            # TODO I think this is the bug...
            if cheat=='yes' or cheat=='both':
                comtmp = comscores[0][start:start+slen]
                assert len(comtmp)==len(h)
                assert isinstance(comtmp, list)
                assert isinstance(comtmp[0], float)
                comind = comtmp.index(max(comtmp))
                tmp['comranked'] = h[comind]
                tmp['cometscores'] = comtmp
            if cheat=='no' or cheat=='both':
                qetmp = qescores[0][start:start+slen]
                assert len(qetmp)==len(h)
                
                qeind = qetmp.index(max(qetmp))
                tmp['qeranked'] = h[qeind]
                tmp['qescores'] = qetmp
            start+=slen
            # TODO double check if htis is needed
            reranked_info.append(tmp)
        except:
            print("bug")
    return reranked_info

def get_mbart_nll(cand, ind, tok, mod, dev):

    inp = cand['inp']
    out = cand['cands'][ind]

    inputs = tok(inp, return_tensors="pt").to(dev)
    with tok.as_target_tokenizer():
        labels = tok(out, return_tensors="pt").to(dev)

    # forward pass
    output = mod(**inputs, labels=labels.input_ids)
    #print(type(labels))
    #print(labels.attention_mask)
    return output.loss
   
setup = "de"
def rescore_cands(c_list):
    device = "cuda:2" if torch.cuda.is_available() else "cpu"
    if setup == "de":
        mname = "facebook/mbart-large-50-one-to-many-mmt"
        src_l = "en_XX"
        tgt_l = "de_DE"
    else:
        mname = "facebook/mbart-large-50-many-to-one-mmt"
        src_l = "fr_XX"
        tgt_l = "en_XX"
    tok = AutoTokenizer.from_pretrained(mname, src_lang=src_l, tgt_lang=tgt_l)
    mod = AutoModelForSeq2SeqLM.from_pretrained(mname)
    mod.to(device)
    print("rescoring candidates")
    i = 0
    for c in c_list:
        try:
            c['rescores'] = []
            for can in range(0, len(c['cands'])):
                c['rescores'].append(float(get_mbart_nll(c, can, tok, mod, device)))
            c['sco_ranks'] = list(numpy.argsort(c['rescores']))
        except:
            print("rescore bug")
            continue
        print(i)
        i+=1
    del tok
    del mod
    return c_list

import json

def savejson(fname, cands):
    tmpfile = open(fname+".jsonl", "w")
    for l in cands:
        try:
            l['rescores'] = [float(a) for a in l['rescores']]
            l['sco_ranks'] = [int(a) for a in l['sco_ranks']]
            tmpfile.write(json.dumps(l))
            tmpfile.write('\n')
        except:
            print("something off")
    tmpfile.close()

rescore = False
if __name__ == "__main__":
    args = process_args()
    args.device = args.device if torch.cuda.is_available() else "cpu"
    cheat = args.oracle
    if rescore:
        if exists("./candoutputs/"+args.candfile+"rescored.jsonl"):
            cand_data = load_cands("./candoutputs/"+args.candfile+"rescored.jsonl")
        else:
            cand_data = load_cands("./candoutputs/"+args.candfile+".jsonl")
            cand_data = rescore_cands(cand_data)
            savejson("./candoutputs/"+args.candfile+"rescored", cand_data)
    else:
        cand_data = load_cands("./candoutputs/"+args.candfile+".jsonl")

    if exists('rerank_outputs/'+args.candfile+'.json'):
        with open('rerank_outputs/'+args.candfile+'.json') as f:
            rerank_info = json.load(f)['data']
    else:
        rerank_info = get_reranked_cands(cand_data)
        storererank = {}
        storererank['data'] = rerank_info
        f = open('rerank_outputs/'+args.candfile+'.json', 'w')
        json.dump(storererank, f)
        f.close()
    
    comet_path = download_model(cometmodel, "./cometmodel")
    comet = load_from_checkpoint(comet_path)
    rrinfo = comet_rerank_info(rerank_info)
    del comet
    del comet_path
    rrinfo['avglen'] = get_average_candlen(cand_data)
    rrinfo['avgsco'] = get_average_score(cand_data, False)
    rrinfo['bestsco'] = amax_score(cand_data)
    f = open('rerankdata/'+args.candfile+'.json', 'w')
    json.dump(rrinfo, f)
    f.close()
        