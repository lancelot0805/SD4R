CONFIG_PATH=./projects/SD4R/configs/vod-SD4R_16x4_48e.py
CHECKPOINT_PATH=./projects/SD4R/checkpoints/best.pth
OUTPUT_NAME=vod-SD4R
PRED_RESULTS=./tools_det3d/view-of-delft-dataset/pred_results/$OUTPUT_NAME 

GPUS="4"
PORT=${PORT:-29500}
CUDA_VISIBLE_DEVICES="4,5,6,7" \
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/tools_det3d/test.py \
    --format-only \
    --eval-options submission_prefix=$PRED_RESULTS \
    --config $CONFIG_PATH \
    --checkpoint $CHECKPOINT_PATH \
    --launcher pytorch ${@:4}

# python tools_det3d/test.py \
# --format-only \
# --eval-options submission_prefix=$PRED_RESULTS \
# --config $CONFIG_PATH \
# --checkpoint $CHECKPOINT_PATH

python tools_det3d/view-of-delft-dataset/FINAL_EVAL.py \
--pred_results $PRED_RESULTS
