#python baseline_cands.py -method="nucleus" -num_hyps=1 -dataset='fr_en'
python baseline_cands.py -method="beam" -num_hyps=1 -dataset='en_de' -num_examples=808 -max_len=70
python rerank_score_cands.py -candfile="beam1en_de"

python baseline_cands.py -method="beam" -num_hyps=10 -dataset='en_de' -num_examples=808 -max_len=70
python rerank_score_cands.py -candfile="beam10en_de"

python baseline_cands.py -method="beam" -num_hyps=40 -dataset='en_de' -num_examples=808 -max_len=70
python rerank_score_cands.py -candfile="beam40en_de"

#python baseline_cands.py -method="nucleus" -num_hyps=10 -dataset='en_de'

#python rerank_score_cands.py -candfile="beam1fr_en"

#PYTHONPATH=./ python src/recom_search/command/run_pipeline.py -nexample 200  -ngram_suffix 4 -beam_size 10 -min_len 15 -max_len 70 -model astar -merge zip  -avg_score 0.75  -adhoc 