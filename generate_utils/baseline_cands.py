from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import json
import argparse
import mt_data
import pandas as pd
import time
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')


def process_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-method', type=str)
    parser.add_argument('-device', type=str, default='cuda:2')
    parser.add_argument('-dataset', default='fr_en', type=str)
    parser.add_argument('-top_p', type=float, default=0.6)
    parser.add_argument('-num_examples', type=int, default=800)
    parser.add_argument('-max_len', type=int, default=35)
    parser.add_argument('-num_hyps', type=int, default=1)


    args = parser.parse_args()
    return args

def load_model(dset, dev):
    if "en_" in dset:
        tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50-one-to-many-mmt")
        model = AutoModelForSeq2SeqLM.from_pretrained("facebook/mbart-large-50-one-to-many-mmt")
        model.to(dev)
    elif "_en" in dset:
        tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50-many-to-one-mmt")
        tokenizer.src_lang = "fr_XX"
        model = AutoModelForSeq2SeqLM.from_pretrained("facebook/mbart-large-50-many-to-one-mmt")
        model.to(dev)
    return tokenizer, model

def beam_generate(model_inputs, tokenizer, model, per_example, batch, lcode, mlen=80):
    
    starttime = time.time()

    gen = model.generate(
        **model_inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id[lcode],
        num_beams=per_example,
        max_new_tokens=mlen, 
        num_return_sequences=per_example,
        return_dict_in_generate=True,
        output_scores=True,
    )
    totaltime = round((time.time() - starttime), 2)
    return gen, totaltime

def nucleus_generate(model_inputs, tokenizer, model, per_example, batch, lcode, args):
    return model.generate(
        **model_inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id[lcode],
        do_sample=True, 
        max_length=args.max_len, 
        top_p=args.top_p, 
        num_return_sequences=per_example,
        return_dict_in_generate=True,
        output_scores=True,
    )

smax = torch.nn.Softmax(dim=0)
def extract_greedy_score(sco):
    seqscores = [0]*len(sco[0])
    seqtots = [0]*len(sco[0])

    for i in range(1, len(sco)):
        for j in range(0, len(seqscores)):
            tmp = sco[i][j]
            if (tmp is not None) and (tmp[0] is not None):
                tmp = torch.log(smax(tmp))
                seqscores[j]+=torch.max(tmp)
                seqtots[j]+=1
    for j in range(0, len(seqscores)):
        seqscores[j]/=seqtots[j]
    return seqscores
    
#bs_cands = []
# TODO update referneces with the new args
def get_generate_candidates(gfunction, inputs, refs, tokenizer, model, per_example, batch, dset, device, method, mlen):
    #global bs_cands
    bs_cands = []
    lcode = "de_DE"
    if "_en" in dset:
        lcode = "en_XX"
    tmpfile = open("./candoutputs/test"+method+str(per_example)+dset+".jsonl", "w")
    tdistr = []
    for i in range(0, int(len(inputs)/batch)):
        print(i*batch)
        ins = inputs[i*batch:batch*(i+1)]
        ins = ['' if pd.isna(item) else item for item in ins]

        model_inputs = tokenizer(ins, return_tensors="pt", truncation=True, padding=True).to(device)
        generated_tokens, ttime = gfunction(model_inputs, tokenizer, model, per_example, batch, lcode, mlen)
        tdistr.append(ttime)
        torch.cuda.empty_cache()
        allsco = []
        if per_example>1:
            allsco = generated_tokens.sequences_scores
        else:
            allsco = extract_greedy_score(generated_tokens.scores)
            if i==0:
                print(allsco)
        allcands = tokenizer.batch_decode(generated_tokens.sequences, skip_special_tokens=True)
        for j in range(0, batch):
            tmp = {}
            if per_example>1:
                tmp['scores'] = list(allsco[j*per_example:per_example*(j+1)])
                tmp['scores'] = [tensor.item() for tensor in tmp['scores']]
            else:
                tmp['scores'] = [float(allsco[j])]
            tmp['cands'] = list(allcands[j*per_example:per_example*(j+1)])
            tmp['inp'] = inputs[i*batch+j]
            tmp['ref'] = refs[i*batch+j]
            #print(tmp)
            bs_cands.append(tmp)
            tmpfile.write(json.dumps(tmp))
            tmpfile.write('\n')
            
        # ensure space is cleared
        del model_inputs
        del generated_tokens
        del allsco
        del allcands
        torch.cuda.empty_cache()
    tmpfile.close()
    return bs_cands, tdistr
 

if __name__ == "__main__":
    args = process_args()
    args.device = args.device if torch.cuda.is_available() else "cpu"
    
    print("Loading dataset - "+args.dataset)
    dataset = mt_data.load_generate_set(args.num_examples, args.dataset)
    assert dataset is not None

    print("Loading model, tokenizer")
    tokenizer, model = load_model(args.dataset, args.device)
    assert model is not None
    assert tokenizer is not None

    print("generating candidates")
    
    gmethod = beam_generate
    if args.method=="nucleus":
        gmethod = nucleus_generate
    inp = dataset[args.dataset[:2]].to_list()
    ref = dataset[args.dataset[3:]].to_list()
    
    # adjust batch size to fit on GPU
    bsize = 8
    if args.num_hyps>10:
        bsize = 4
    if args.num_hyps>30:
        bsize = 2
        
    # function, inputs, refs, tokenizer, model, per_example=10, batch=8
    candidates = get_generate_candidates(gmethod, inp, ref, tokenizer, model, args.num_hyps, bsize)
    print("Sucessfully generated and stored!")



# select device
#device = "cuda:2" if torch.cuda.is_available() else "cpu"

