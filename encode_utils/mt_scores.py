import csv
import time
import sys
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
import torch
from comet import download_model, load_from_checkpoint
from generate_utils.distill_comet import load_distill_model, run_distill_comet

from bleurt import score
import sys
sys.path.insert(1, '/mnt/data1/prasann/latticegen/lattice-generation/COMET')
csv.field_size_limit(sys.maxsize)
from COMET.comet.models.regression.referenceless import ReferencelessRegression
from COMET.comet.models import load_from_checkpoint as lfc

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if "de" in dset:
        mname = "facebook/mbart-large-50-one-to-many-mmt"
        src_l = "en_XX"
        tgt_l = "de_DE"
    elif "ru" in dset:
        mname = "facebook/mbart-large-50-one-to-many-mmt"
        src_l = "en_XX"
        tgt_l = "ru_RU"
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
    starttime = time.time()
    for i in range(0, len(hyplist)):
        if i%500==0:
            print(i)
        with torch.no_grad():
            result.append(float(get_mbart_nllsco(srclist[i], hyplist[i], inptok, labtok, mod, device)))
    
        #result.append(0)
            
        #print(i)
        #i+=1
    totaltime = round((time.time() - starttime), 2)
    print("TOTAL TIME ", totaltime)
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

def get_bleurt_scores(hyps, refs, bsize):
    tokenizer = AutoTokenizer.from_pretrained("Elron/bleurt-base-512")
    model = AutoModelForSequenceClassification.from_pretrained("Elron/bleurt-base-512").to(device)
    model.eval()
    num_batches = len(hyps)/bsize
    allsco = []
    with torch.no_grad():
        for i in range(int(num_batches)):
            inputs = tokenizer(list(refs[i*bsize:(i+1)*bsize]), list(hyps[i*bsize:(i+1)*bsize]), return_tensors='pt', padding=True, truncation=True).to(device)
            scores = model(**inputs)[0].squeeze()
            allsco.extend(scores)
            torch.cuda.empty_cache()
            if i%100==0:
                print(i)

    return [float(a) for a in allsco]

def get_comet_scores(hyps, srcs, refs, comet):
    cometqe_input = [{"src": src, "mt": mt, "ref":ref} for src, mt, ref in zip(srcs, hyps, refs)]
    # sentence-level and corpus-level COMET
    outputs = comet.predict(
        cometqe_input, batch_size=32, progress_bar=True
    )
    torch.cuda.empty_cache()
    return outputs

DSET_CHKS = {
    "copcqe":"/mnt/data1/prasann/latticegen/lattice-generation/COMET/lightning_logs/version_38/checkpoints/epoch=3-step=140000.ckpt",
    "dupcqe":"/mnt/data1/prasann/latticegen/lattice-generation/COMET/lightning_logs/version_43/checkpoints/epoch=3-step=140000.ckpt",
    "utnoun":"/mnt/data1/prasann/latticegen/lattice-generation/COMET/lightning_logs/version_44/checkpoints/epoch=9-step=40000.ckpt"
}
# sco is the score funct, dset is either model name or 
# is the language (can be style as well in certain cases)
def get_scores_auto(hyps, srcs, refs, sco="cqe", dset = ""):
    totaltime = -1
    # comet qe
    if sco=='cqe':
        cometqe_path = download_model(cometqe_model, cometqe_dir)
        model = load_from_checkpoint(cometqe_path).to(device)
        model.eval()
        starttime = time.time()
        with torch.no_grad():
            scos = get_cometqe_scores(hyps, srcs, model)
        totaltime = round((time.time() - starttime), 2)
        print("TOOK TIME ", totaltime)
        scos = scos[0]
        del model 
        del cometqe_path
        return scos
    if dset == "comstyle":
        reflessmod = lfc(DSET_CHKS[sco], False).to(device)
        reflessmod.eval()
        starttime = time.time()
        with torch.no_grad():
            scos = get_cometqe_scores(hyps, srcs, reflessmod)
        totaltime = round((time.time() - starttime), 2)
        print("TOOK TIME ", totaltime)
        scos = scos[0]
        del reflessmod 
        return scos
    if sco=='comet':
        comet_path = download_model(cometmodel, "./cometmodel")
        try:
            comet = load_from_checkpoint(comet_path).to(device)
        except:
            comet = load_from_checkpoint(comet_path, False).to(device)
        starttime = time.time()
        scos = get_comet_scores(hyps, srcs, refs, comet)
        totaltime = round((time.time() - starttime), 2)
        print("TOOK TIME ", totaltime)
        scos = scos[0]
        del comet
        del comet_path
        return scos
    if sco=='posthoc':
        return rescore_cands(dset, hyps, srcs)
    if sco=='cqeut':
        assert len(dset)>0
        utmod = load_distill_model(dset)
        starttime = time.time()
        scos = run_distill_comet(srcs, hyps, utmod)
        totaltime = round((time.time() - starttime), 2)
        print("TOOK TIME ", totaltime)
        del utmod
        return scos
    if sco=="bleurt":
        return get_bleurt_scores(hyps, refs, 64)
    # model will be passed in 
    if sco == "custom":
        #assert len(dset)>0
        utmod = dset
        starttime = time.time()
        scos = run_distill_comet(srcs, hyps, utmod)
        totaltime = round((time.time() - starttime), 2)
        print("TOOK TIME ", totaltime)
        del utmod
        return scos
    # should never reach this
    print("invalid score")
    assert False
        