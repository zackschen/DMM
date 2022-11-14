GPU_ID=$1

IMDB="coco_minus_refer"
ITERS=1250000
TAG="notime"
NET="res101"
DATASET="refcoco"
SPLITBY="unc"
ID=finetune
F_ID="finetune"

CUDA_VISIBLE_DEVICES=${GPU_ID} python -u ./tools/testModel.py \
    --imdb_name ${IMDB} \
    --net_name ${NET} \
    --iters ${ITERS} \
    --tag ${TAG} \
    --dataset ${DATASET} \
    --splitBy ${SPLITBY} \
    --id ${ID} \
    --learning_rate 4e-4 \
    --learning_rate_decay_start 6 \
    --learning_rate_decay_every 6 \
    --f_id ${F_ID} \
    2>&1 | tee log/testModel.log
