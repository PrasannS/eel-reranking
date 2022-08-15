# First command, beam size 2, use total scoring
#PYTHONPATH=./ python src/recom_search/scripts/run_eval.py -model bfs_recom -dfs_expand -dataset fr-en -beam_size 4 -task mtn1 -min_len 5 -max_len -1 -ngram_suffix 4 -merge zip -device cuda:2 -avg_score 0.9
#PYTHONPATH=./ python src/recom_search/scripts/run_eval.py -model bfs_recom -dfs_expand -dataset fr-en -beam_size 4 -task mtn1 -min_len 5 -max_len -1 -ngram_suffix 4 -merge zip -device cuda:2 -avg_score 0.6
#PYTHONPATH=./ python src/recom_search/scripts/run_eval.py -model bfs_recom -dfs_expand -dataset fr-en -beam_size 8 -task mtn1 -min_len 5 -max_len -1 -ngram_suffix 4 -merge zip -device cuda:2 -avg_score 0.6
#PYTHONPATH=./ python src/recom_search/scripts/run_eval.py -model bfs_recom -dfs_expand -dataset fr-en -beam_size 4 -task mtn1 -min_len 5 -max_len -1 -ngram_suffix 4 -merge zip -device cuda:2 -avg_score 0.9
#PYTHONPATH=./ python src/recom_search/scripts/run_eval.py -model bfs_recom -dfs_expand -dataset fr-en -beam_size 4 -task mtn1 -min_len 5 -max_len -1 -ngram_suffix 4 -merge rcb -device cuda:2 -avg_score 1.2
#PYTHONPATH=./ python src/recom_search/scripts/run_eval.py -model bfs_recom -dfs_expand -dataset en-de -beam_size 4 -task mt1n -min_len 5 -max_len 80 -ngram_suffix 4 -merge rcb -device cuda:2 -avg_score 0.9
PYTHONPATH=./ python src/recom_search/scripts/run_eval.py -model bfs_recom -dfs_expand -dataset fr-en -beam_size 4 -task mtn1 -min_len 5 -max_len -1 -ngram_suffix 4 -merge rcb -device cuda:2 -avg_score 0.91 -nexample 5
#PYTHONPATH=./ python src/recom_search/scripts/run_eval.py -model bfs_recom -dfs_expand -dataset en-de -beam_size 1 -task mt1n -min_len 5 -max_len 80 -ngram_suffix 4 -merge rcb -device cuda:2 -avg_score 0.9 -nexample 300 

# Get the candidate files
#python lattice_cands.py -dataset="fr_en" -path_output="mtn1_fr-en_bfs_recom_4_-1_False_0.4_True_False_4_5_rcb_0.9_0.0_0.9"
#python lattice_cands.py -dataset="fr_en" -path_output="mtn1_fr-en_bfs_recom_4_-1_False_0.4_True_False_4_5_zip_0.9_0.0_0.9"
#python lattice_cands.py -dataset="fr_en" -path_output="mtn1_fr-en_bfs_recom_8_-1_False_0.4_True_False_4_5_zip_0.6_0.0_0.9"
#python lattice_cands.py -dataset="fr_en" -path_output="mtn1_fr-en_bfs_recom_4_-1_False_0.4_True_False_4_5_zip_0.9_0.0_0.9"
#python lattice_cands.py -dataset="fr_en" -path_output="mtn1_fr-en_bfs_recom_4_-1_False_0.4_True_False_4_5_rcb_1.2_0.0_0.9"
#python lattice_cands.py -dataset="fr_en" -path_output="mtn1_fr-en_bfs_recom_4_-1_False_0.4_True_False_4_5_rcb_0.9_0.0_0.9"
#python lattice_cands.py -dataset="fr_en" -path_output="mtn1_fr-en_bfs_recom_4_-1_False_0.4_True_False_4_5_none_0.9_0.0_0.9"
#python lattice_cands.py -dataset="en_de" -exploded="False" -path_output="mt1n_en-de_bfs_recom_1_80_False_0.4_True_False_4_5_rcb_0.9_0.0_0.9"
#python lattice_cands.py -dataset="fr_en" -exploded="False" -path_output="mtn1_fr-en_bfs_recom_1_-1_False_0.4_True_False_4_5_rcb_0.9_0.0_0.9"

# Do rerank
#python rerank_score_cands_new.py -candfile="mt1n_en-de_bfs_recom_1_80_False_0.4_True_False_4_5_rcb_0.9_0.0_0.9" -oracle="both"
#python rerank_score_cands_new.py -candfile="mtn1_fr-en_bfs_recom_1_-1_False_0.4_True_False_4_5_rcb_0.9_0.0_0.9" -oracle="both"
#python rerank_score_cands_new.py -candfile="mtn1_fr-en_bfs_recom_8_-1_False_0.4_True_False_4_5_zip_0.6_0.0_0.9" -oracle="both"
#python rerank_score_cands_new.py -candfile="mtn1_fr-en_bfs_recom_4_-1_False_0.4_True_False_4_5_zip_0.9_0.0_0.9" -oracle="both"
#python rerank_score_cands_new.py -candfile="explodedmt1n_en-de_bfs_recom_4_80_False_0.4_True_False_4_5_rcb_0.9_0.0_0.9" -oracle="both"
#python rerank_score_cands_new.py -candfile="explodedmtn1_fr-en_bfs_recom_4_-1_False_0.4_True_False_4_5_rcb_0.9_0.0_0.9" -oracle="both"
#python rerank_score_cands_new.py -candfile="explodedmt1n_en-de_bfs_recom_4_80_False_0.4_True_False_4_5_rcb_0.9_0.0_0.9" -oracle="both"

#python rerank_score_cands_new.py -candfile="explodedmt1n_en-de_bfs_recom_4_80_False_0.4_True_False_4_5_rcb_0.9_0.0_0.9" -oracle="both"
#python rerank_score_cands_new.py -candfile="mtn1_fr-en_bfs_recom_4_-1_False_0.4_True_False_4_5_none_0.9_0.0_0.9" -oracle="both"
#python modify_data.py -candfile="mt1n_en-de_bfs_recom_1_80_False_0.4_True_False_4_5_rcb_0.9_0.0_0.9"
#python modify_data.py -candfile="mtn1_fr-en_bfs_recom_1_-1_False_0.4_True_False_4_5_rcb_0.9_0.0_0.9"


