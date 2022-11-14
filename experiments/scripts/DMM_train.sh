GPU_ID=$1
DATASET=$2
SPLITBY=$3

IMDB="coco_minus_refer"
ITERS=1250000
TAG="notime"
NET="res101"
TASK=10
ID=${TASK}task-DMM-buffer_80
F_ID=${ID}


CUDA_VISIBLE_DEVICES=${GPU_ID} nohup python -u ./tools/mas_buffer_sequence_train.py \
    --imdb_name ${IMDB} \
    --net_name ${NET} \
    --iters ${ITERS} \
    --tag ${TAG} \
    --dataset ${DATASET} \
    --splitBy ${SPLITBY} \
    --max_category_iters 10000 \
    --max_category_epoch 20 \
    --buffer_start_epoch 0 \
    --buffer_size 120 \
    --with_st 1 \
    --id ${ID} \
    --f_id ${F_ID} \
    --subject_flag 1 \
    --multi_buffer 0 \
    --module_sum 0 \
    --module_normalize 1 \
    --sub_module 1 \
    --freeze_id 0 \
    --task ${TASK}\
    1>./log/${DATASET}/${ID} 2>&1 &
