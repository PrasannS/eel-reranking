# Get stuff from French 50
python rerank_score_cands.py -candfile="beam50fr_en" -oracle="both"
# Get stuf from recent minimal lattice 
python rerank_score_cands.py -candfile="fr-en_bfs_recom_2_-" -oracle="both"


