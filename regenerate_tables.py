import pickle
from encode_utils.rerank_data import rerank_dist, rerank_single
from encode_utils.efficient_rerank import get_effrerank_model, run_comstyle
from encode_utils.sco_funct import weightaddprob, default_scofunct
from encode_utils.mt_scores import get_scores_auto
from encode_utils.new_flatten_lattice import get_dictlist
from encode_utils.new_mask_utils import randomsingle, useall
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
import torch
import random
from generate_tables import metrics_mapping
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def regen_preds(inpcsv, scotype):
    df = pd.read_csv("outputs/score_csvs/"+inpcsv, index_col=0)
    # get rid of old 
    # score
    # TODO drop old bad columns, note there's some weird model discrepancy based on lfc method
    df = df.rename(columns={scotype:scotype+"prev2"})
    # get new score and return that
    metrics_mapping(scotype, df)
    df.to_csv("outputs/score_csvs/"+inpcsv)


# regenerate all preds
if __name__=="__main__":
    col = {
        #"noun_xsum_b12": ['reversed_xsum_beam12/', 'nounxsumbeam12v2.csv'],
        "mt_fren_b12": ['reversed_mtfren_beam12/', 'mtfrenbeam12v2.csv'],
        #"noun_xsum_b50": ['reversed_xsum_beam50/', 'nounxsumbeam50v2.csv'],
        "mt_fren_b50": ['reversed_mtfren_beam50/', 'mtfrenbeam50v2.csv'],
        #"noun_xsum": ["nounsum_reversed/", "nounxsumlargeexplodev2.csv"],
        #"noun_fren": ["frtest_reversed/", "nounlargeexplodev1.csv"],
        "mt_fren": ["frtest_reversed/", "frenchlargeexplodev1.csv"],
        "mt_ende": ["detest_reversed/", "germanlargeexplodev1.csv"],
        "mt_enru": ["rutest_reversed/", "russianlargeexplodev1.csv"],
    }
    for k in col.keys():
        if "noun" in k:
            regen_preds(col[k][1], "utnoun")
        else:
            regen_preds(col[k][1], "dupcqe")