

GPU_ID=$1
DATASET=$2
SPLITBY=$3

IMDB="coco_minus_refer"
ITERS=1250000
TAG="notime"
NET="res101"
# ID="mrcn_cmr_with_st"
ID="matching_sliding"


CUDA_VISIBLE_DEVICES=${GPU_ID} python ./tools/finetune_sequence_train.py \
    --imdb_name ${IMDB} \
    --net_name ${NET} \
    --iters ${ITERS} \
    --tag ${TAG} \
    --dataset ${DATASET} \
    --splitBy ${SPLITBY} \
    --max_category_iters 10000 \
    --with_st 1 \
    --id ${ID} \
