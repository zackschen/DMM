GPU_ID=$1
DATASET=$2
SPLITBY=$3

IMDB="coco_minus_refer"
ITERS=1250000
TAG="notime"
NET="res101"
TASK=5
ID=${TASK}task-MAS_shared1
F_ID=${ID}


# weighted:0 freeze_id:0 mas
# weighted:0 freeze_id:2 shared
# weighted:1 freeze_id:0 weighted

CUDA_VISIBLE_DEVICES=${GPU_ID} nohup python -u ./tools/mas_sequence_train.py \
    --imdb_name ${IMDB} \
    --net_name ${NET} \
    --iters ${ITERS} \
    --tag ${TAG} \
    --dataset ${DATASET} \
    --splitBy ${SPLITBY} \
    --max_category_iters 10000 \
    --max_category_epoch 20 \
    --with_st 1 \
    --module_sum 0 \
    --module_normalize 0 \
    --sub_module 0 \
    --id ${ID} \
    --f_id ${F_ID} \
    --weighted 0 \
    --freeze_id 2 \
    --task ${TASK}\
    1>./log/${DATASET}/${ID} 2>&1 &
