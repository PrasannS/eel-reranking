export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=3
# -m debugpy --listen 0.0.0.0:5678 --wait-for-client
python comet/cli/train.py --cfg configs/models/referenceless_metric.yaml