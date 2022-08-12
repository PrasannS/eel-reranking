# getting lattice candidates
#python lattice_cands.py -dataset="fr_en" -path_output="mtn1_fr-en_bfs_recom_2_-1_False_0.4_True_False_4_5_zip_-1_0.0_0.9"
# python modify_data.py -candfile="beam50fr_en"
# python modify_data.py -candfile="beam50en_de"

#python modify_data.py -candfile="mt1n_en-de_bfs_recom_4_80_False_0.4_True_False_4_5_rcb_0.9_0.0_0.9"
#python modify_data.py -candfile="explodedmt1n_en-de_bfs_recom_4_80_False_0.4_True_False_4_5_rcb_0.9_0.0_0.9"
#python rerank_score_cands_new.py -candfile="beam50en_de" -oracle="both"

#python modify_data.py -candfile="beam50en_de"
#python modify_data.py -candfile="explodedmt1n_en-de_bfs_recom_4_80_False_0.4_True_False_4_5_rcb_0.9_0.0_0.9"
#python modify_data.py -candfile="post1explodedmtn1_fr-en_bfs_recom_4_-1_False_0.4_True_False_4_5_rcb_0.9_0.0_0.9"
python rerank_score_cands_new.py -candfile="explodedmtn1_fr-en_bfs_recom_4_-1_False_0.4_True_False_4_5_rcb_0.9_0.0_0.9" -oracle="both"
python modify_data.py -candfile="explodedmtn1_fr-en_bfs_recom_4_-1_False_0.4_True_False_4_5_rcb_0.9_0.0_0.9"