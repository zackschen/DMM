GPU_ID=$1
DATASET=$2
SPLITBY=$3

IMDB="coco_minus_refer"
ITERS=1250000
TAG="notime"
NET="res101"
# ID=$4
TASK=10
ID=${TASK}task-finetune1
F_ID=${ID}

CUDA_VISIBLE_DEVICES=${GPU_ID} nohup python -u ./tools/finetune_sequence_train.py \
    --imdb_name ${IMDB} \
    --net_name ${NET} \
    --iters ${ITERS} \
    --tag ${TAG} \
    --dataset ${DATASET} \
    --splitBy ${SPLITBY} \
    --max_category_iters 150000 \
    --max_category_epoch 20 \
    --id ${ID} \
    --f_id ${F_ID} \
    --task ${TASK} \
    1>./log/${DATASET}/${ID} 2>&1 &
