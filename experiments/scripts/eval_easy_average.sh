GPU_ID=$1
DATASET=$2
SPLITBY=$3

ID=$4
TASK=5

case ${DATASET} in
    refcoco)
        for SPLIT in test
        do
            CUDA_VISIBLE_DEVICES=${GPU_ID} python ./tools/eval_easy_average.py \
                --dataset ${DATASET} \
                --splitBy ${SPLITBY} \
                --split ${SPLIT} \
                --id ${ID}\
                --task ${TASK}
        done
    ;;
    refcoco+)
        for SPLIT in test
        do
            CUDA_VISIBLE_DEVICES=${GPU_ID} python ./tools/eval_easy_average.py \
                --dataset ${DATASET} \
                --splitBy ${SPLITBY} \
                --split ${SPLIT} \
                --id ${ID}\
                --task ${TASK}
        done
    ;;
    refcocog)
        for SPLIT in test
        do
            CUDA_VISIBLE_DEVICES=${GPU_ID} python ./tools/eval_easy_average.py \
                --dataset ${DATASET} \
                --splitBy ${SPLITBY} \
                --split ${SPLIT} \
                --id ${ID}\
                --task ${TASK}
        done
    ;;
esac
