import csv
import pandas as pd
import random
from os.path import exists
import sys
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import numpy
import json 
import argparse
from mbart_qe import download_mbart_qe, load_mbart_qe
from comet import download_model, load_from_checkpoint
from distill_comet import load_distill_model, run_distill_comet

from transquest.algo.sentence_level.monotransquest.run_model import (
    MonoTransQuestModel,
    MonoTransQuestArgs,
)

csv.field_size_limit(sys.maxsize)

def process_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-candfile', type=str, default='beam40en_de')
    parser.add_argument('-device', type=str, default='cuda:2')


    args = parser.parse_args()
    return args

def proc_cands(cand_data):
    
    clen = len(cand_data['modelscores'])
    # return refs, hyps, srcs
    return [cand_data['ref']]*clen, cand_data['cands'], [cand_data['src']]*clen

def convert_wmt ():
    #TODO there's definetely a way to clean this up
    with open("translation_data/news-commentary-v15.de-en.tsv") as file:
        tsv_file = csv.reader(file, delimiter="\t")
        i = 0
        res = []
        # generate data for everything, shuffle and sample later
        for f in tsv_file:
            if i == 100000:
                break
            res.append(f)
            i = i+1
    random.shuffle(res)
    print("GOT HERE SOMEHOW")
    res = res[:10000]
    tmpdf = pd.DataFrame(res)
    tmpdf['de'] = tmpdf[0]
    tmpdf['en'] = tmpdf[1]
    del tmpdf[0]
    del tmpdf[1]
    with open('mt-data/use/en-de.de', 'w') as f:
        for line in tmpdf['de']:
            f.write(line)
            f.write('\n')
    with open('mt-data/use/en-de.en', 'w') as f:
        for line in tmpdf['en']:
            f.write(line)
            f.write('\n')
    
    return tmpdf

def get_mbart_nll(cand, ind, inptok, labtok, mod, dev):
    
    inp = cand['src']
    out = cand['cands'][ind]

    inputs = inptok(inp, return_tensors="pt").to(dev)
    with labtok.as_target_tokenizer():
        labels = labtok(out, return_tensors="pt").to(dev)

    # forward pass
    output = mod(**inputs, labels=labels.input_ids)
    #print(type(labels))
    #print(labels.attention_mask)
    return output.loss


setup = "de"
def rescore_cands(c_list, dset):
    device = "cuda:2" if torch.cuda.is_available() else "cpu"
    if "de" in dset:
        mname = "facebook/mbart-large-50-one-to-many-mmt"
        src_l = "en_XX"
        tgt_l = "de_DE"
    else:
        mname = "facebook/mbart-large-50-many-to-one-mmt"
        src_l = "fr_XX"
        tgt_l = "en_XX"
    inptok = AutoTokenizer.from_pretrained(mname)
    labtok = AutoTokenizer.from_pretrained(mname, src_lang=src_l, tgt_lang=tgt_l)
    mod = AutoModelForSeq2SeqLM.from_pretrained(mname)
    mod.to(device)
    print("rescoring candidates")
    i = 0
    result = []
    for c in c_list:
        try:
            c['rescores'] = []
            for can in range(0, len(c['cands'])):
                c['rescores'].append(float(get_mbart_nll(c, can, inptok, labtok, mod, device)))
            c['sco_ranks'] = list(numpy.argsort(c['rescores']))
            result.append(c)
        except:
            print("bug")
            continue
        
        print(i)
        i+=1
    del inptok
    del labtok
    return c_list

def get_mbartqe_scores(hyps, srcs, mod, langpair):
    mbartqe_input = [{"src": src, "mt": mt, "lp":langpair} for src, mt in zip(srcs, hyps)]
    # sentence-level and corpus-level COMET
    _, segment_scores = mod.predict(
        mbartqe_input, batch_size=64, show_progress=True
    )
    mbart_score = [s[0] for s in segment_scores]
    mbart_uncertainty = [s[1] for s in segment_scores]
    torch.cuda.empty_cache()
    return mbart_score, mbart_uncertainty

def get_allhyps_srcs(c_list):
    allhyps = []
    allsrcs = []
    allrefs = []
    for i in range(0, len(c_list)):
        r, h, s = proc_cands(c_list[i])
        # TODO, should remove empty candidates, but may introduce bugs
        #if len(h)==0:
        #    continue
        allhyps.extend(h)
        allsrcs.extend(s)
        allrefs.extend(r)
    allhyps = ['' if pd.isna(item) else item for item in allhyps] 
    allsrcs = ['' if pd.isna(item) else item for item in allsrcs] 
    allrefs = ['' if pd.isna(item) else item for item in allrefs]
    return allhyps, allsrcs, allrefs

def mbartqe_score_all(c_list, dset):
    #device = "cuda:2" if torch.cuda.is_available() else "cpu"
    if "de" in dset:
        lpair = "en-de"
    else:
        lpair = "fr-en"
    mbartqe_dir = "./mbartqemodel"
    mbartqe_model = "wmt21-mbart-m2"
    mbart_path = download_mbart_qe(mbartqe_model, mbartqe_dir)
    mbart = load_mbart_qe(mbart_path)

    allhyps, allsrcs, allrefs = get_allhyps_srcs(c_list)
    print("mbartqe scoring candidates")
    mbartqe_all, mbartqe_unc = get_mbartqe_scores(allhyps, allsrcs, mbart, lpair)
    del mbart
    del mbart_path
    i = 0
    start = 0
    for c in c_list:
        r, h, s = proc_cands(c)
        
        # used rescored best
        
        slen = len(c['modelscores'])
        scotmp = mbartqe_all[start:start+slen]
        unctmp = mbartqe_unc[start:start+slen]
        assert len(scotmp)==len(h)
        # TODO I think this is the bug...
        c["mbartqescores"] = scotmp
        c["mbartqeuncs"] = unctmp
        start+=slen

    return c_list

def transquest_score_all (c_list):
    allhyps, allsrcs, allrefs = get_allhyps_srcs(c_list)
    print("transquest scoring candidates")
    tquest_all = transquest(allhyps, allsrcs)
    
    start = 0
    for c in c_list:
        
        slen = len(c['modelscores'])
        scotmp = list(tquest_all[start:start+slen])

        c["tquestscores"] = scotmp
        assert len(scotmp)==len(c["modelscores"])
        start+=slen

    return c_list

def transquest(hyps, srcs):
    transquest_model = "TransQuest/monotransquest-da-multilingual"

    transquest_args = MonoTransQuestArgs(eval_batch_size=64)
    transquest_model = MonoTransQuestModel(
        "xlmroberta",
        transquest_model,
        num_labels=1,
        use_cuda=torch.cuda.is_available(),
        args=transquest_args,
    )
    transquest_input = [[src, mt] for src, mt in zip(srcs, hyps)]
    transquest_scores, _ = transquest_model.predict(transquest_input)
    torch.cuda.empty_cache()
    return transquest_scores

def distillcom_all(c_list):
    allhyps, allsrcs, allrefs = get_allhyps_srcs(c_list)
    print("distillcomet scoring candidates")
    dcom_all = distillcomet(allhyps, allsrcs)
    start = 0
    for c in c_list:
        
        slen = len(c['modelscores'])
        scotmp = list(dcom_all[start:start+slen])

        c["distillscores"] = scotmp
        assert len(scotmp)==len(c["modelscores"])
        start+=slen

    return c_list

def distillcomet (hyps, srcs):
    discomodel = load_distill_model()
    scores = run_distill_comet(srcs, hyps, discomodel)
    del discomodel
    return scores
    
def savepostjson(fname, cands, pid):

    for l in cands:
        try:
            l['rescores'] = [float(a) for a in l['rescores']]
            l['sco_ranks'] = [int(a) for a in l['sco_ranks']]
        except:
            print("something off")
    tmp = {}
    tmp['data'] = cands
    with open("rerank_outputs/post"+str(pid)+fname+".json", "w") as outfile:
        json.dump(tmp, outfile)


if __name__ == "__main__":
    args = process_args()
    args.device = args.device if torch.cuda.is_available() else "cpu"
    dset = "en_de"
    if "fr_en" in args.candfile or "fr-en" in args.candfile:
        dset = "fr_en" 
    # convert_wmt()
    with open("rerank_outputs/"+args.candfile+".json") as f:
        original = json.load(f)['data']
    if "rescores" not in original[0].keys():
        original = rescore_cands(original, dset)
        savepostjson(args.candfile, original, 2)
    if "distillscores" not in original[0].keys():
        original = distillcom_all(original)
        savepostjson(args.candfile, original, 2)
    """
    if "mbartqescores" not in original[0].keys():
        original = mbartqe_score_all(original, dset)
        savepostjson(args.candfile, original, 1)

    if "tquestscores" not in original[0].keys():
        original = transquest_score_all(original)
        savepostjson(args.candfile, original, 1)
    """
    

    savepostjson(args.candfile, original, 2)
    
