#!/bin/sh

## MODIFY PATH for YOUR SETTING
ROOT_DIR=/home/yanpengxiang/deeplab-v2

CAFFE_DIR=${ROOT_DIR}
CAFFE_BIN=${CAFFE_DIR}/.build_release/tools/caffe.bin

EXP=carvana

if [ "${EXP}" = "carvana" ]; then
    NUM_LABELS=2
    DATA_ROOT=/media/Disk/yanpengxiang/dataset/carvana/
else
    NUM_LABELS=0
    echo "Wrong exp name"
fi

MASK_CHANNEL=32 
GCN_KERL_SIZE=15
GCN_KERL_PAD=7
#----modify the exper path-----
EXP=${ROOT_DIR}/exper/${EXP}
#-----------------------------

## Specify which model to train
########### carvana ################
NET_ID=VGG #SimplifyModel
DEV_ID=2
# 0 Testing, 1 Training
PHASE=1

RESTORE_TRAIN=0
#RESTORE_ITER=80000
RESTORE_ITER=6006

#####

## Create dirs

CONFIG_DIR=${EXP}/config/${NET_ID}
MODEL_DIR=${EXP}/model/${NET_ID}
VISUAL_DIR=${EXP}/visual/${NET_ID}
mkdir -p ${VISUAL_DIR}
rm ${VISUAL_DIR}/*
mkdir -p ${MODEL_DIR}
LOG_DIR=${EXP}/log/${NET_ID}
mkdir -p ${LOG_DIR}
export GLOG_log_dir=${LOG_DIR}

## Run

if [ ${PHASE} -eq 1 ]; then
  echo "--------- Training -------------"
  RUN_TRAIN=1
  RUN_TEST=0
else
  echo "--------- Testing --------------"
  RUN_TRAIN=0
  RUN_TEST=1
fi

## Training #1 (on train_aug)

if [ ${RUN_TRAIN} -eq 1 ]; then
    #
    LIST_DIR=${EXP}/list
    TRAIN_SET=train_val
    #TRAIN_SET=train${TRAIN_SET_SUFFIX}
    #if [ -z ${TRAIN_SET_WEAK_LEN} ]; then
    #    TRAIN_SET_WEAK=${TRAIN_SET}_diff_${TRAIN_SET_STRONG}
    #    comm -3 ${LIST_DIR}/${TRAIN_SET}.txt ${LIST_DIR}/${TRAIN_SET_STRONG}.txt > ${LIST_DIR}/${TRAIN_SET_WEAK}.txt
    #else
    #    TRAIN_SET_WEAK=${TRAIN_SET}_diff_${TRAIN_SET_STRONG}_head${TRAIN_SET_WEAK_LEN}
    #    comm -3 ${LIST_DIR}/${TRAIN_SET}.txt ${LIST_DIR}/${TRAIN_SET_STRONG}.txt | head -n ${TRAIN_SET_WEAK_LEN} > ${LIST_DIR}/${TRAIN_SET_WEAK}.txt
    #fi
    #
    # MODEL=${EXP}/model/${NET_ID}/mask6_up_three_scale_ch64_iter_8800.caffemodel

    MODEL=${EXP}/model/${NET_ID}/save/train_iter_${RESTORE_ITER}.caffemodel


    FEATURE_DIR=/media/Disk/yanpengxiang/exper/carvana/features/${NET_ID}

    #
    echo Training net ${EXP}/${NET_ID}
    for pname in train solver; do
        sed "$(eval echo $(cat sub.sed))" \
            ${CONFIG_DIR}/${pname}.prototxt > ${CONFIG_DIR}/${pname}_${TRAIN_SET}.prototxt
    done
        CMD="${CAFFE_BIN} train \
         --solver=${CONFIG_DIR}/solver_${TRAIN_SET}.prototxt \
         --gpu=${DEV_ID}"
    if [ ${RESTORE_TRAIN} -eq 1 ]; then
        CMD="${CMD} --snapshot=${EXP}/model/${NET_ID}/train_iter_${RESTORE_ITER}.solverstate"
    elif [ -f ${MODEL} ]; then
        CMD="${CMD} --weights=${MODEL}"
    fi
    echo Running ${CMD} && ${CMD}
fi

## Test #1 specification (on val or test)

if [ ${RUN_TEST} -eq 1 ]; then
  for TEST_SET in val; do
    TEST_ITER=`cat ${EXP}/list/${TEST_SET}.txt | wc -l`
    cat ${EXP}/list/${TEST_SET}.txt
    MODEL=${EXP}/model/${NET_ID}/save/train_iter_${RESTORE_ITER}.caffemodel
    if [ ! -f ${MODEL} ]; then
      MODEL=`ls -t ${EXP}/model/${NET_ID}/train_iter_*.caffemodel | head -n 1`
    fi
    #
    echo Testing net ${EXP}/${NET_ID}
    FEATURE_DIR=/media/Disk/yanpengxiang/exper/carvana/features/${NET_ID}
    #mkdir -p ${FEATURE_DIR}/${TEST_SET}/fc8
    #mkdir -p ${FEATURE_DIR}/${TEST_SET}/fc8_softmax
    mkdir -p ${FEATURE_DIR}/${TEST_SET}/final_fusion
    sed "$(eval echo $(cat sub.sed))" \
		${CONFIG_DIR}/${TEST_SET}.prototxt > ${CONFIG_DIR}/test_${TEST_SET}.prototxt
	CMD="${CAFFE_BIN} test \
        --model=${CONFIG_DIR}/test_${TEST_SET}.prototxt \
        --weights=${MODEL} \
        --gpu=${DEV_ID} \
        --iterations=${TEST_ITER}"
	echo Running ${CMD} && ${CMD}
    done
fi
