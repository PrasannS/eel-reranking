import csv
import time
import sys
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
import torch
from comet import download_model, load_from_checkpoint
from generate_utils.distill_comet import load_distill_model, run_distill_comet

from bleurt import score

csv.field_size_limit(sys.maxsize)

from encode_utils.efficient_rerank import get_effrerank_model
from encode_utils.eval_utils import batch_hyp_sco

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

def get_comstyle_correct(srcs, hyps, model, device, bsize = 32):

    maxl = len(srcs)
    assert len(srcs)==len(hyps)
    inptok = AutoTokenizer.from_pretrained("xlm-roberta-base")
    args = {
        'tok':inptok,
        'model':model, 
        'device':device
    }
    allsco = []
    ind = 0 
    LOG_STEPS = 20
    while ind<maxl:
        if (ind/32)%LOG_STEPS == 0:
            print(100*(ind/maxl),"%")
        # get all appropriate scores
        try:
            sco = batch_hyp_sco(srcs[ind:ind+bsize], hyps[ind:ind+bsize], args)
        except:
            sco = [0]*len(srcs[ind:ind+bsize])
            print("encountered bug at index ", ind)
        allsco.extend([float(m) for m in sco])
        ind+=bsize
    return allsco

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

loss_fcnt = torch.nn.CrossEntropyLoss(reduction='none')

def get_mbart_nllsco(inpu, outpu, inptok, labtok, mod, dev):
    
    inp = inpu
    out = outpu

    inputs = inptok(inp, padding=True, truncation=True, return_tensors="pt").to(dev)
    with labtok.as_target_tokenizer():
        labels = labtok(out, padding=True, truncation=True, return_tensors="pt").to(dev)

    # forward pass
    output = mod(**inputs, labels=labels.input_ids)
    l = loss_fcnt(output.logits.view(-1, mod.config.vocab_size), labels['input_ids'].view(-1))
    losses = torch.mean(l.reshape(len(inpu), int(len(l)/len(inpu))), 1)
    #print(type(labels))
    #print(labels.attention_mask)
    return losses

def rescore_cands(dset, hyplist, srclist):
    device = "cuda:3" if torch.cuda.is_available() else "cpu"
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
    
    if "table" in dset:
        inptok = AutoTokenizer.from_pretrained("facebook/bart-base")
        new_tokens = ['<H>', '<R>', '<T>']
        new_tokens_vocab = {}
        new_tokens_vocab['additional_special_tokens'] = []
        for idx, t in enumerate(new_tokens):
            new_tokens_vocab['additional_special_tokens'].append(t)
        num_added_toks = inptok.add_special_tokens(new_tokens_vocab)
        # first get cond gen model
        ckpt = torch.load("/mnt/data1/prasann/latticegen/lattice-generation/parent_explore/plms-graph2text/webnlg-bart-base.ckpt")
        state_dict = ckpt['state_dict']
        # make weight keys compatible 
        for key in list(state_dict.keys()):
            if key[:len("model.")]=="model.":
                state_dict[key[len("model."):]] = state_dict.pop(key)
        # TODO let's check and see if this works correctly
        mod = AutoModelForSeq2SeqLM.from_pretrained(
            "facebook/bart-base", state_dict=ckpt['state_dict'], vocab_size=50268
        )
        labtok = inptok
    else:
        inptok = AutoTokenizer.from_pretrained(mname)
        labtok = AutoTokenizer.from_pretrained(mname, src_lang=src_l, tgt_lang=tgt_l)
        mod = AutoModelForSeq2SeqLM.from_pretrained(mname)
    mod.to(device)
    mod.eval()
    print("rescoring candidates")
    i = 0
    result = []
    starttime = time.time()
    bsize = 8
    for i in range(0, int(len(hyplist)/bsize)+1):
        if i%100==0:
            print(i)
        with torch.no_grad():
            try:
                result.extend([float(f) for f in (get_mbart_nllsco(srclist[i*bsize:(i+1)*bsize], hyplist[i*bsize:(i+1)*bsize], inptok, labtok, mod, device))])
            except:
                print("error")
                result.append([0]*bsize)
    
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
    "utnoun":"/mnt/data1/prasann/latticegen/lattice-generation/COMET/lightning_logs/version_44/checkpoints/epoch=9-step=40000.ckpt",
    "parentqe": "/mnt/data1/prasann/latticegen/lattice-generation/COMET/lightning_logs/version_57/checkpoints/epoch=1-step=50000.ckpt"
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
    if sco=='parentqe':
        model = get_effrerank_model("parentqe", False)
        model.eval()
        starttime = time.time()
        with torch.no_grad():
            scos = get_cometqe_scores(hyps, srcs, model)
        totaltime = round((time.time() - starttime), 2)
        print("TOOK TIME ", totaltime)
        scos = scos[0]
        del model 
        return scos
    if dset == "comstyle":
        # TODO this is new, now using batched version of eval model
        if "noun" in sco:
            reflessmod = get_effrerank_model("noun")
        # use mt model (causal)
        else:
            reflessmod = get_effrerank_model("comstyle")
        #reflessmod = lfc(DSET_CHKS[sco], True).to(device)
        reflessmod.eval()
        starttime = time.time()
        #with torch.no_grad():
        scos = get_comstyle_correct(srcs, hyps, reflessmod, device)
        totaltime = round((time.time() - starttime), 2)
        print("TOOK TIME ", totaltime)
        #scos = scos[0]
        del reflessmod 
        return scos
    if sco=='comet':
        comet_path = download_model(cometmodel, "./cometmodel")
        try:
            comet = load_from_checkpoint(comet_path).to(device)
        except:
            comet = load_from_checkpoint(comet_path, False).to(device)
        comet.eval()
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
        