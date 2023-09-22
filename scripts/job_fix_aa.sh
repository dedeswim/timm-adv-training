#!/bin/bash

export DATASET=cifar10
export DATA_DIR=/p/vast1/MLdata
export BBANK=asyncmlb
export BTIME=12:00
export BOUTDIR=logs
export BASE_CONFIG=configs/sweep-cifar10.yaml
export EXPERIMENT=robust-hw
export OUTPUT_DIR=/p/gpfs1/robustHW/
export MEAN="0.4914 0.4822 0.4465"
export STD="0.2023 0.1994 0.2010"

# Validation-only params
export VAL_BATCH_SIZE=2048
export EPS=8
export N_GPUS=4

VAL_BATCH_SIZE=128
while read path; do
  EXPERIMENT_DIR=${path}
  {
      python check_end_of_training.py ${OUTPUT_DIR}/${EXPERIMENT_DIR}
      sh scripts/chainer_main.sh "validate_robustbench.py" "--data-dir=$DATA_DIR --model=resnet70_16_dm --checkpoint=${OUTPUT_DIR}/${EXPERIMENT_DIR}/best.pth.tar --batch-size=$VAL_BATCH_SIZE --eps=$EPS --mean $MEAN --std $STD --gpus=$N_GPUS --log-wandb --log-to-file --aa-state-path ${OUTPUT_DIR}/${EXPERIMENT_DIR}/aa-state.json" $OUTPUT_DIR "$EXPERIMENT_DIR" aa; sleep 1
  } &
  sleep 1
done <aa_to_run_dm.txt

while read path; do
  EXPERIMENT_DIR=${path}
  {
      python check_end_of_training.py ${OUTPUT_DIR}/${EXPERIMENT_DIR}
      sh scripts/chainer_main.sh "validate_robustbench.py" "--data-dir=$DATA_DIR --model=resnet70_16 --checkpoint=${OUTPUT_DIR}/${EXPERIMENT_DIR}/best.pth.tar --batch-size=$VAL_BATCH_SIZE --eps=$EPS --mean $MEAN --std $STD --gpus=$N_GPUS --log-wandb --log-to-file --aa-state-path ${OUTPUT_DIR}/${EXPERIMENT_DIR}/aa-state.json" $OUTPUT_DIR "$EXPERIMENT_DIR" aa; sleep 1
  } &
  sleep 1
done <aa_to_run_no_dm.txt