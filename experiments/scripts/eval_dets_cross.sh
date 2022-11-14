

GPU_ID=$1
DATASET=$2
SPLITBY=$3

# IMDB="coco_minus_refer"
# ITERS=1150000
# TAG="notime"
# NET="res101"
ID="test_tensorflow_pool5"

case ${DATASET} in
    flickr30k)
        for SPLIT in test
        do
            CUDA_VISIBLE_DEVICES=${GPU_ID} python ./tools/eval_det_cross.py \
                --dataset ${DATASET} \
                --splitBy ${SPLITBY} \
                --split ${SPLIT} \
                --id ${ID}
        done
    ;;
    refcoco)
        for SPLIT in val testA testB
        do
            CUDA_VISIBLE_DEVICES=${GPU_ID} python ./tools/eval_det_cross.py \
                --dataset ${DATASET} \
                --splitBy ${SPLITBY} \
                --split ${SPLIT} \
                --id ${ID}
        done
    ;;
    refcoco+)
        for SPLIT in val testA testB
        do
            CUDA_VISIBLE_DEVICES=${GPU_ID} python ./tools/eval_det_cross.py \
                --dataset ${DATASET} \
                --splitBy ${SPLITBY} \
                --split ${SPLIT} \
                --id ${ID}
        done
    ;;
    refcocog)
        for SPLIT in val test
        do
            CUDA_VISIBLE_DEVICES=${GPU_ID} python ./tools/eval_det_cross.py \
                --dataset ${DATASET} \
                --splitBy ${SPLITBY} \
                --split ${SPLIT} \
                --id ${ID}
        done
    ;;
esac
