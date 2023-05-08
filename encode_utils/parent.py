# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
# I have just rewritten stuff so that it works without tensorflow
# I also added multi-precessing to greatly speedup the computations

"""Script to compute PARENT metric."""
from functools import partial

import tqdm.notebook as tqdm_notebook
import multiprocessing as mp
import numpy as np
import pandas as pd
import collections
import itertools
import argparse
import math
import tqdm
import json
import os
import spacy

def overlap_probability(ngram, table, smoothing=0.0, stopwords=None):
    """Returns the probability that the given n-gram overlaps with the table.

    A simple implementation which checks how many tokens in the n-gram are also
    among the values in the table. For tables with (attribute, value) pairs on the
    `value` field is condidered. For tables with (head, relation, tail) triples a
    concatenation of `head` and `tail` are considered.

    E.g.:
    >>> overlap_probability(["michael", "dahlquist"],
                             [(["name"], ["michael", "dahlquist"])])
    >>> 1.0

    Args:
    ngram: List of tokens.
    table: List of either (attribute, value) pairs or (head, relation, tail)
      triples. Each member of the pair / triple is assumed to already be
      tokenized into a list of strings.
    smoothing: (Optional) Float parameter for laplace smoothing.
    stopwords: (Optional) List of stopwords to ignore (assign P = 1).

    Returns:
    prob: Float probability of ngram being entailed by the table.
    """
    # pylint: disable=g-complex-comprehension
    if len(table[0]) == 2:
        table_values = set([tok for _, value in table for tok in value])
    else:
        table_values = set([tok for head, _, tail in table for tok in head + tail])
    
    overlap = 0
    for token in ngram:
        if stopwords is not None and token in stopwords:
            overlap += 1
            continue
        if token in table_values:
            overlap += 1
    return float(overlap + smoothing) / float(len(ngram) + smoothing)


def _mention_probability(table_entry, sentence, smoothing=0.0):
    """Returns the probability that the table entry is mentioned in the sentence.

    A simple implementation which checks the longest common subsequence between
    the table entry and the sentence. For tables with (attribute, value) pairs
    only the `value` is considered. For tables with (head, relation, tail) triples
    a concatenation of the `head` and `tail` is considered.

    E.g.:
    >>> _mention_probability((["name"], ["michael", "dahlquist"]),
                             ["michael", "dahlquist", "was", "a", "drummer"])
    >>> 1.0

    Args:
    table_entry: Tuple of either (attribute, value) or (head, relation, tail).
      Each member of the tuple is assumed to already be tokenized into a list of
      strings.
    sentence: List of tokens.
    smoothing: Float parameter for laplace smoothing.

    Returns:
    prob: Float probability of entry being in sentence.
    """
    if len(table_entry) == 2:
        value = table_entry[1]
    else:
        value = table_entry[0] + table_entry[2]
    overlap = _len_lcs(value, sentence)
    return float(overlap + smoothing) / float(len(value) + smoothing)


def _len_lcs(x, y):
    """Returns the length of the Longest Common Subsequence between two seqs.

    Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence

    Args:
    x: sequence of words
    y: sequence of words

    Returns
    integer: Length of LCS between x and y
    """
    table = _lcs(x, y)
    n, m = len(x), len(y)
    return table[n, m]


def _lcs(x, y):
    """Computes the length of the LCS between two seqs.

    The implementation below uses a DP programming algorithm and runs
    in O(nm) time where n = len(x) and m = len(y).
    Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence

    Args:
    x: collection of words
    y: collection of words

    Returns:
    Table of dictionary of coord and len lcs
    """
    n, m = len(x), len(y)
    table = dict()
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 or j == 0:
                table[i, j] = 0
            elif x[i - 1] == y[j - 1]:
                table[i, j] = table[i - 1, j - 1] + 1
            else:
                table[i, j] = max(table[i - 1, j], table[i, j - 1])
    return table


def nwise(iterable, n=2):
    """Yields all ngrams of given order n in iterable."""
    iterables = itertools.tee(iterable, n)
    [next(iterables[i]) for i in range(n) for j in range(i)]
    return zip(*iterables)


def _ngram_counts(sequence, order):
    """Returns count of all ngrams of given order in sequence."""
    if len(sequence) < order:
        return collections.Counter()
    return collections.Counter(nwise(sequence, order))


def parent_instance_level(package,
                          lambda_weight=0.5,
                          smoothing=0.00001,
                          max_order=4,
                          entailment_fn=overlap_probability,
                          mention_fn=_mention_probability):
    """
    In the case of multiple references, score is the max among references.
    """
    
    prediction, references, table = package  # unpacking
    
    # Compute recall against table fields (it doesn't depend on the ref).
    table_mention_probs = [mention_fn(entry, prediction)
                     for entry in table]
    table_rec = sum(table_mention_probs) / len(table) or smoothing
    
    multi_c_prec, multi_c_rec, multi_c_f = list(), list(), list()
    
    for reference in references:
        # Weighted ngram precisions and recalls for each order.
        ngram_prec, ngram_rec = list(), list()
        for order in range(1, max_order + 1):
            # Collect n-grams and their entailment probabilities.
            pred_ngram_counts = _ngram_counts(prediction, order)
            pred_ngram_weights = {ngram: entailment_fn(ngram, table)
                                  for ngram in pred_ngram_counts}
            ref_ngram_counts = _ngram_counts(reference, order)
            ref_ngram_weights = {ngram: entailment_fn(ngram, table)
                                 for ngram in ref_ngram_counts}

            # Precision.
            numerator, denominator = 0., 0.
            for ngram, count in pred_ngram_counts.items():
                denominator += count
                prob_ngram_in_ref = min(
                    1., float(ref_ngram_counts.get(ngram, 0) / count))
                numerator += count * (
                    prob_ngram_in_ref +
                    (1. - prob_ngram_in_ref) * pred_ngram_weights[ngram])
            if denominator == 0.:
                # Set precision to 0.
                ngram_prec.append(0.0)
            else:
                ngram_prec.append(numerator / denominator)

            # Recall.
            numerator, denominator = 0., 0.
            for ngram, count in ref_ngram_counts.items():
                prob_ngram_in_pred = min(
                    1., float(pred_ngram_counts.get(ngram, 0) / count))
                denominator += count * ref_ngram_weights[ngram]
                numerator += count * ref_ngram_weights[ngram] * prob_ngram_in_pred
            if denominator == 0.:
                # Set recall to 1.
                ngram_rec.append(1.0)
            else:
                ngram_rec.append(numerator / denominator)

        # Smoothing.
        for order in range(1, max_order):
            if ngram_prec[order] == 0.:
                ngram_prec[order] = smoothing
            if ngram_rec[order] == 0.:
                ngram_rec[order] = smoothing

        # Compute geometric averages of precision and recall for all orders.
        w = 1. / max_order
        if any(prec == 0. for prec in ngram_prec):
            c_prec = 0
        else:
            sp = (w * math.log(p_i) for p_i in ngram_prec)
            c_prec = math.exp(math.fsum(sp))
        if any(rec == 0. for rec in ngram_rec):
            ref_rec = smoothing
        else:
            sr = [w * math.log(r_i) for r_i in ngram_rec]
            ref_rec = math.exp(math.fsum(sr))
            
        # Combine reference and table recalls.
        if ref_rec == 0. or table_rec == 0.:
            c_rec = 0
        else:
            if lambda_weight is None:
                lw = sum([mention_fn(entry, reference) for entry in table
               ]) / len(table)
                lw = 1. - lw
            else:
                lw = lambda_weight

            c_rec = math.exp((1. - lw) * math.log(ref_rec) + (lw) * math.log(table_rec))

        # F-score.
        c_f = (2. * c_prec * c_rec) / (c_prec + c_rec + 1e-8)
        
        multi_c_prec.append(c_prec)
        multi_c_rec.append(c_rec)
        multi_c_f.append(c_f)
           
    return max(multi_c_prec), max(multi_c_rec), max(multi_c_f)


def _parent(predictions,
            references,
            tables,
            lambda_weight=0.5,
            smoothing=0.00001,
            max_order=4,
            entailment_fn=overlap_probability,
            mention_fn=_mention_probability,
            n_jobs=-1,
            use_tqdm=True):
    """
    Metric for comparing predictions to references given tables.
    Upgrade from original version (see first line of this file):
    It now uses multiprocessing to go faster (minutes to seconds).

    ARGS:
    predictions: An iterator over tokenized predictions.
                 Each prediction is a list.
    references: An iterator over lists of tokenized references.
                Each prediction can have multiple references.
    tables: An iterator over the tables. Each table is a list of tuples, with
            tuples being either (attribute, value) or (head, relation, tail).
            The members of the tuples are assumed to be themselves tokenized
            lists of strings. E.g.
                `[(["name"], ["michael", "dahlquist"]),
                  (["birth", "date"], ["december", "22", "1965"])]`
            is one table in the (attribute, value) format with two entries.
    lambda_weight: Float weight in [0, 1] to multiply table recall.
    smoothing: Float value to replace zero values of precision and recall.
    max_order: Maximum order of the ngrams to use.
    entailment_fn: A python function for computing the probability that an
                   ngram is entailed by the table. Its signature should match
                   that of `overlap_probability` above.
    mention_fn: A python function for computing the probability that a
                table entry is mentioned in the text. Its signature should
                match that of `_mention_probability` above.
    n_jobs: An int to specify number of parallel workers. 
            -1 to use all available.
    use_tqdm: A boolean or str to specify whether or not to use tqm.
              Usefull to deactivate when using the function in a notebook.
              if str, use either 'classic' or 'notebook'. If boolean, defaults
              to classic

    RETURNS:
    precision: Average precision of all predictions.
    recall: Average recall of all predictions.
    f1: Average F-scores of all predictions.
    all_f_scores: List of all F-scores for each item.
    """
    # sanity check
    references, _tqdm = validate_parent_args(predictions, references, tables,
                                             lambda_weight, smoothing, max_order,
                                             use_tqdm)
    
    print(_tqdm)
    
    precisions, recalls, all_f_scores = list(), list(), list()
    
    _parent = partial(parent_instance_level, 
                      lambda_weight=lambda_weight,
                      smoothing=smoothing,
                      max_order=max_order,
                      entailment_fn=entailment_fn,
                      mention_fn=mention_fn)
    
    n_jobs = mp.cpu_count() if n_jobs < 0 else n_jobs
    print(f'Using {n_jobs} processes, starting now')
    with mp.Pool(processes=n_jobs) as pool:
        _iterable = pool.imap(
            _parent, 
            zip(predictions, references, tables),
            chunksize=n_jobs  # empirically seems to be the best, could be wrong though
        )

        if _tqdm is not None:
            for p, r, f in _tqdm.tqdm(
                    _iterable, total=len(tables), desc='Computing PARENT'):
                precisions.append(p)
                recalls.append(r)
                all_f_scores.append(f)
        else:
            
            for p, r, f in _iterable:
                precisions.append(p)
                recalls.append(r)
                all_f_scores.append(f)
        
    return precisions, recalls, all_f_scores


def validate_parent_args(predictions, references, tables,
                         lambda_weight, smoothing, max_order, 
                         use_tqdm):
    assert len(predictions) == len(tables)
    
    # handle multi-references. Also handle empty line at end of file.
    if len(predictions) != len(references[:len(predictions)]):
        # Transposing references so that references[idx] contains all refs
        # for predictions[idx].
        references = [[r for r in refs if r] for refs in zip(*references)]
    else:
        references = [[ref] for ref in references]
    references = references[:len(predictions)]  # remove empty line at eof
    assert all(len(refs)>=1 for refs in references)  # check for empty refs
    
    assert isinstance(lambda_weight, float)
    assert 0 <= lambda_weight <= 1
    
    assert isinstance(smoothing, float)
    
    assert isinstance(max_order, int)
    assert max_order > 0
    
    if isinstance(use_tqdm, bool):
        _tqdm = tqdm if use_tqdm else None
    if isinstance(use_tqdm, str):
        if use_tqdm == 'classic':
            _tqdm = tqdm
        elif use_tqdm == 'notebook':
            _tqdm = tqdm_notebook
        else:
            raise ValueError('use_tqdm should be in [classic|notebook].'
                             f'Was given <{use_tqdm}>.')
    return references, _tqdm


def parent(predictions,
           references,
           tables,
           lambda_weight=0.5,
           smoothing=0.00001,
           max_order=4,
           entailment_fn=overlap_probability,
           mention_fn=_mention_probability,
           avg_results=True,
           n_jobs=-1,
           use_tqdm=True):
    
    """
    Metric for comparing predictions to references given tables.
    Upgrade from original version (see first line of this file):
    It now uses multiprocessing to go faster (minutes to seconds).

    ARGS:
    predictions: An iterator over tokenized predictions.
                 Each prediction is a list.
    references: An iterator over lists of tokenized references.
                Each prediction can have multiple references.
    tables: An iterator over the tables. Each table is a list of tuples, with
            tuples being either (attribute, value) or (head, relation, tail).
            The members of the tuples are assumed to be themselves tokenized
            lists of strings. E.g.
                `[(["name"], ["michael", "dahlquist"]),
                  (["birth", "date"], ["december", "22", "1965"])]`
            is one table in the (attribute, value) format with two entries.
    lambda_weight: Float weight in [0, 1] to multiply table recall.
    smoothing: Float value to replace zero values of precision and recall.
    max_order: Maximum order of the ngrams to use.
    entailment_fn: A python function for computing the probability that an
                   ngram is entailed by the table. Its signature should match
                   that of `overlap_probability` above.
    mention_fn: A python function for computing the probability that a
                table entry is mentioned in the text. Its signature should
                match that of `_mention_probability` above.
    avg_results: A boolean to specify if results should be the average or
                 all single scores.
    n_jobs: An int to specify number of parallel workers. 
            -1 to use all available.
    use_tqdm: A boolean to specify whether or not to use tqm. 
              Usefull to deactivate when using the function in a notebook.
              
    RETURNS:
    precision, recall, f_score: either three floats or three lists of floats.
    """
    
    precisions, recalls, f_scores = _parent(
            predictions,
            references,
            tables,
            lambda_weight=lambda_weight,
            smoothing=smoothing,
            max_order=max_order,
            entailment_fn=entailment_fn,
            mention_fn=mention_fn,
            n_jobs=n_jobs,
            use_tqdm=use_tqdm)
        
    if avg_results:
        precisions = sum(precisions) / len(precisions)
        recalls = sum(recalls) / len(recalls)
        f_scores = sum(f_scores) / len(f_scores)

    return precisions, recalls, f_scores

"""
python parent.py --predictions $PREDICTION_PATH --references $REFERENCES_PATH --tables $TABLES_PATH --avg_results
"""

def nlp_parse(inpdoc, nlp):
    doc = nlp(inpdoc.strip())
    return [token.text for token in doc]

def nlp_quickparse(inp, nlp):
    pre_proc = [t.strip() for t in inp]
    result = []
    ind = 0
    # we only need the tokenizer right? 
    for doc in nlp.pipe(pre_proc, disable=["tagger", "parser", "ner", "lemmatizer", "textcat"]):
        if ind%1000==0:
            print(ind)
        result.append([token.text for token in doc])
        ind+=1
    return result

BASEPATH = "/mnt/data1/prasann/latticegen/lattice-generation/parent_explore/stagewise_finetune/"
traingen = False
if traingen:
    table_path = "parent_master/webnlgtrain.jl"
    ref_paths = ["pos/traindata/test.target"]
else:
    table_path = "parent_master/webnlg.jl"
    ref_paths = ["pos/test.target_eval", "pos/test.target2_eval", "pos/test.target3_eval"]

# method that takes in a dataframe of hypotheses (should have references)
# correctly maps stuff to the right references and generates parent scores
def parent_score_df(hypdfs, cache_name=None):
    nlp = spacy.load('en_core_web_sm')

    # 1) get all of the right stuff, get it synched up
    with open(BASEPATH+table_path, mode="r", encoding='utf8') as f:
        tables = [json.loads(line) for line in f if line.strip()]

    references = list()
    for filename in ref_paths:
        with open(BASEPATH+filename, mode="r", encoding='utf8') as f:
            tmp = []
            for line in f:
                doc = line.strip() # TODO nlp used to be here
                tmp.append(doc)
            references.append(tmp)

    if False: #os.path.exists(BASEPATH+"testrefs.csv"):
        parent_df = pd.read_csv(BASEPATH+"testrefs.csv", index_col=0)
        print("using old")
    else:
        parent_df = pd.DataFrame()
        parent_df['tab_inp'] = tables
        parent_df['ref'] = references[0]
        if len(references)>1:
            parent_df['ref2'] = references[1]
            parent_df['ref3'] = references[2]

        print("references loaded")

        # do parsing of stuff.  # 2) nlp tokenize everything 
        parent_df['r1p'] = nlp_quickparse(list(parent_df['ref']), nlp)
        if len(references)>1:
            parent_df['r2p'] = nlp_quickparse(list(parent_df['ref2']), nlp)
            parent_df['r3p'] = nlp_quickparse(list(parent_df['ref3']), nlp)
        """
        parent_df.apply(lambda x: nlp_parse(x.ref, nlp), 1)
        parent_df['r2p'] = parent_df.apply(lambda x: nlp_parse(x.ref2, nlp), 1)
        parent_df['r3p'] = parent_df.apply(lambda x: nlp_parse(x.ref3, nlp), 1)
        """
       

        print("references processed")

        if cache_name is not None:
            parent_df.to_csv(BASEPATH+"testrefs.csv")
    
    if False: #os.path.exists(BASEPATH+cache_name):
        merged = pd.read_csv(BASEPATH+cache_name, index_col=0)
    else:
        # connect associated columns based on un-parsed reference
        merged = pd.merge(hypdfs, parent_df, on='ref')
        merged['hyp_parsed'] = nlp_quickparse(list(merged['hyp']), nlp)
        #merged['hyp_parsed'] = merged.apply(lambda x: nlp_parse(x.hyp, nlp), 1)   

    if cache_name is not None:
        merged.to_csv(BASEPATH+cache_name)
    print("hyps processed") 
    if len(references)>1:
        refinps = [[], [], []]
    else:
        refinps = [[]]
    refinps[0].extend(list(merged['r1p']))
    if len(references)>1:
        refinps[1].extend(list(merged['r2p']))
        refinps[2].extend(list(merged['r3p']))

    # 3) score with parent
    precisions, recalls, f_scores = parent(
                list(merged['hyp_parsed']),
                refinps,
                list(merged['tab_inp']),
                lambda_weight=.5,
                smoothing=.00001,
                max_order=4,
                avg_results=False,
                n_jobs=4,
                use_tqdm=True)

    # 4) return scores
    merged['precision'] = precisions
    merged['recall'] = recalls
    merged['f1'] = f_scores

    return merged