import csv
import pandas as pd
import random
from os.path import exists
import sys
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from comet import download_model, load_from_checkpoint
from distill_comet import load_distill_model, run_distill_comet
csv.field_size_limit(sys.maxsize)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

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

def get_mbart_nllsco(inpu, outpu, inptok, labtok, mod, dev):
    
    inp = inpu
    out = outpu

    inputs = inptok(inp, return_tensors="pt").to(dev)
    with labtok.as_target_tokenizer():
        labels = labtok(out, return_tensors="pt").to(dev)

    # forward pass
    output = mod(**inputs, labels=labels.input_ids)
    #print(type(labels))
    #print(labels.attention_mask)
    return output.loss

def rescore_cands(dset, hyplist, srclist):
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
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
    mod.eval()
    print("rescoring candidates")
    i = 0
    result = []
    for i in range(0, len(hyplist)):
        if i%500==0:
            print(i)
        with torch.no_grad():
            result.append(float(get_mbart_nllsco(srclist[i], hyplist[i], inptok, labtok, mod, device)))
    
        #result.append(0)
            
        #print(i)
        #i+=1
    del inptok
    del labtok
    del mod
    torch.cuda.empty_cache()
    return result

# cache dir for cometqe model
cometqe_dir = "./cometqemodel"
# can alternatively use wmt21-comet-qe-mqm
cometqe_model = "wmt20-comet-qe-da"
cometmodel = "wmt20-comet-da"


def get_cometqe_scores(hyps, srcs, commodel):
    cometqe_input = [{"src": src, "mt": mt} for src, mt in zip(srcs, hyps)]
    # sentence-level and corpus-level COMET
    outputs = commodel.predict(
        cometqe_input, batch_size=32, progress_bar=True
    )
    torch.cuda.empty_cache()
    return outputs

def get_comet_scores(hyps, srcs, refs, comet):
    cometqe_input = [{"src": src, "mt": mt, "ref":ref} for src, mt, ref in zip(srcs, hyps, refs)]
    # sentence-level and corpus-level COMET
    outputs = comet.predict(
        cometqe_input, batch_size=32, progress_bar=True
    )
    torch.cuda.empty_cache()
    return outputs

# sco is the score funct, dset is either model name or 
# is the language 
def get_scores_auto(hyps, srcs, refs, sco="cqe", dset = ""):
    # comet qe
    if sco=='cqe':
        cometqe_path = download_model(cometqe_model, cometqe_dir)
        model = load_from_checkpoint(cometqe_path).to(device)
        scos = get_cometqe_scores(hyps, srcs, model)
        scos = scos[0]
        del model 
        del cometqe_path
        return scos
    if sco=='comet':
        comet_path = download_model(cometmodel, "./cometmodel")
        comet = load_from_checkpoint(comet_path).to(device)
        scos = get_comet_scores(hyps, srcs, refs, comet)
        scos = scos[0]
        del comet
        del comet_path
        return scos
    if sco=='posthoc':
        return rescore_cands(dset, hyps, srcs)
    if sco=='cqeut':
        assert len(dset)>0
        utmod = load_distill_model(dset)
        scos = run_distill_comet(srcs, hyps, utmod)
        del utmod
        return scos
    # should never reach this
    print("invalid score")
    assert False
        