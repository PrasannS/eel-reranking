cd $1
# conda init bash
# conda activate run-gector
python3  predict.py --model_path  roberta_1_gectorv2.th \
                   --input_file $2 --output_file $3 --output_cnt_file $4
