

GPU_ID=$1
DATASET=$2
SPLITBY=$3

IMDB="coco_minus_refer"
ITERS=1250000
TAG="notime"
NET="res101"
# ID="mrcn_cmr_with_st"
ID="test_tensorflow_pool5"

CUDA_VISIBLE_DEVICES=${GPU_ID} nohup python ./tools/cross_train.py \
    --imdb_name ${IMDB} \
    --net_name ${NET} \
    --iters ${ITERS} \
    --tag ${TAG} \
    --dataset ${DATASET} \
    --splitBy ${SPLITBY} \
    --max_iters 50000 \
    --with_st 1 \
    --id ${ID} \
    1>test_tensorflow_pool5.log 2>test_tensorflow_pool5.err &
