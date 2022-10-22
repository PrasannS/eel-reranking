#python baseline_cands.py -method="beam" -num_hyps=51 -dataset='fr_en' -num_examples=10000 -max_len=70
#python baseline_cands.py -method="beam" -num_hyps=10 -dataset='en_de' -num_examples=100 -max_len=80
#python baseline_cands.py -method="beam" -num_hyps=4 -dataset='en_de' -num_examples=100 -max_len=80


#python baseline_cands.py -method="beam" -num_hyps=1 -dataset='fr_en' -num_examples=100 -max_len=70
#python rerank_score_cands_new.py -candfile="beam1fr_en" -oracle="both"

#python baseline_cands.py -method="beam" -num_hyps=1 -dataset='en_de' -num_examples=100 -max_len=80
#python rerank_score_cands_new.py -candfile="beam1en_de" -oracle="both"
#python rerank_score_cands_new.py -candfile="beam4en_de" -oracle="both"
#python rerank_score_cands_new.py -candfile="beam10en_de" -oracle="both"
#python rerank_score_cands_new.py -candfile="beam50en_de" -oracle="both"

#python baseline_cands.py -method="beam" -num_hyps=10 -dataset='en_de' -num_examples=808 -max_len=70
#python rerank_score_cands.py -candfile="beam10en_de"
#python baseline_cands.py -method="beam" -num_hyps=4 -dataset='fr_en' -num_examples=100 -max_len=80
#python baseline_cands.py -method="beam" -num_hyps=10 -dataset='fr_en' -num_examples=100 -max_len=80

#python rerank_score_cands_new.py -candfile="beam10fr_en" -oracle="both"
#python rerank_score_cands_new.py -candfile="beam4fr_en" -oracle="both"
#python baseline_cands.py -method="beam" -num_hyps=40 -dataset='en_de' -num_examples=808 -max_len=70
#python rerank_score_cands.py -candfile="beam40en_de"

#python baseline_cands.py -method="nucleus" -num_hyps=10 -dataset='en_de'

#python rerank_score_cands_new.py -candfile="rerank_outputs/post1explodedmtn1_fr-en_bfs_recom_4_-1_False_0.4_True_False_4_5_rcb_0.9_0.0_0.9.json" -oracle="both"
python modify_data.py -candfile="post1explodedmtn1_fr-en_bfs_recom_4_-1_False_0.4_True_False_4_5_rcb_0.9_0.0_0.9"

