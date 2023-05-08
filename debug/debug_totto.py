import pickle
from encode_utils.rerank_data import rerank_dist, rerank_single
from encode_utils.efficient_rerank import get_effrerank_model, run_comstyle
from encode_utils.sco_funct import weightaddprob, default_scofunct
from encode_utils.mt_scores import get_scores_auto
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
import torch
import random
import os
import re
from encode_utils.new_mask_utils import randomsingle, useall
from encode_utils.eval_utils import all_lattice_multi
import time
import random
from ast import literal_eval
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

if __name__=="__main__":
    # set up information for set
    col = {
        "noun_xsum": ["nounsum_reversed/", "nounxsumlargeexplodev2.csv"],
        "noun_fren": ["frtest_reversed/", "nounlargeexplodev1.csv"],
        "mt_fren": ["frtest_reversed/", "frenchlargeexplodev1.csv"],
        "mt_ende": ["detest_reversed/", "germanlargeexplodev1.csv"],
        "mt_ende_b12": ["reversed_mtende_beam12/", "mtendebeam12.csv"],
        "mt_ende_b50": ["reversed_mtende_beam50/", "mtendebeam50.csv"],
        "mt_enru": ["rutest_reversed/", "russianlargeexplodev1.csv"],
        "mt_enru_b12": ["reversed_mtenru_beam12/", "mtenrubeam12.csv"],
        "mt_enru_b50": ["reversed_mtenru_beam50/", "mtenrubeam50.csv"],
        "mt_fren_b12": ['reversed_mtfren_beam12/', 'mtfrenbeam12v2.csv'],
        "mt_fren_b50": ['reversed_mtfren_beam50/', 'mtfrenbeam50v2.csv'],
        "noun_fren_b12": ['reversed_mtfren_beam12/', 'nounfrenbeam12v2.csv'],
        "noun_fren_b50": ['reversed_mtfren_beam50/', 'nounfrenbeam50v2.csv'],
        "noun_xsum_b12": ['reversed_xsum_beam12/', 'nounxsumbeam12v2.csv'],
        "noun_xsum_b50": ['reversed_xsum_beam50/', 'nounxsumbeam50v2.csv'],
        "tabtotext_b12": ['reversed_tabtotext_beam12/', 'parentbeam12.csv'],
        "tabtotext_b50": ['reversed_tabtotext_beam50/', 'parentbeam50.csv'],
    }
    curcol = "tabtotext_b12"
    gsuffix = col[curcol][0]
    expl_fname = col[curcol][1]
    base = "outputs/graph_pickles/"+gsuffix
    if "noun" in curcol:
        goldmetric = "utnoun"
    elif "tabtotext" in curcol:
        goldmetric = "pqe"
    else:
        goldmetric = "dupcqe"
    explode_df = pd.read_csv("outputs/score_csvs/"+expl_fname)
    #TODO switch back for french-english
    """
    For now do everything, see if we need this
    if "fren" in curcol:
        SETLEN = len(os.listdir(base))
    else:
        SETLEN = int(len(os.listdir(base))/2)
    """
    SETLEN = len(os.listdir(base))

    # use noun model
    if "noun" in expl_fname:
        encodemod = get_effrerank_model("noun")
        xlm_tok = AutoTokenizer.from_pretrained("xlm-roberta-base")
    # use mt model (causal)
    elif goldmetric == "pqe":
        encodemod = get_effrerank_model("parent", True)
        xlm_tok = encodemod.encoder.tokenizer
    else:
        encodemod = get_effrerank_model("comstyle")
        xlm_tok = AutoTokenizer.from_pretrained("xlm-roberta-base")

    args = {
        'setlen':int(SETLEN),
        'tok':xlm_tok, 
        'dev':device,
        'model':encodemod,
        'explode_df':explode_df,
        'base':base,
        'goldmetric':goldmetric,
        'device':device, 
        'efficient':False,
        'noregen':False,
        'efficient_eval':False
    }

    latpreds = all_lattice_multi(1, default_scofunct, randomsingle, args)