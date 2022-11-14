GPU_ID=$1
DATASET=$2
SPLITBY=$3

IMDB="coco_minus_refer"
ITERS=1250000
TAG="notime"
NET="res101"
TASK=5
TYPE="high"
ID=${TASK}task-buffer_${TYPE}
F_ID=${ID}


CUDA_VISIBLE_DEVICES=${GPU_ID} nohup python -u ./tools/buffer_sequence_train.py \
    --imdb_name ${IMDB} \
    --net_name ${NET} \
    --iters ${ITERS} \
    --tag ${TAG} \
    --dataset ${DATASET} \
    --splitBy ${SPLITBY} \
    --max_category_iters 150000 \
    --max_category_epoch 20 \
    --buffer_start_epoch 1 \
    --with_st 1 \
    --id ${ID} \
    --f_id ${F_ID} \
    --subject_flag 0 \
    --task ${TASK} \
    --buffer_type ${TYPE} \
    1>./log/${DATASET}/${ID} 2>&1 &
