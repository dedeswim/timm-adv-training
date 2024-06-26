#!/bin/bash

# declare -a model_names=("wide_resnet28_10" "wide_resnet34_10" "wide_resnet34_20" "wide_resnet70_16" "wide_resnet28_10_dm" "wide_resnet34_10_dm" "wide_resnet34_20_dm" "wide_resnet70_16_dm")
declare -a model_names=("wide_resnet70_16_dm")
declare -a adv_training_techniques=("pgd" "trades")
declare -a attack_steps=("1" "2" "5" "7" "10")
# declare -a attack_steps=("1" "2" "7" "10")
declare -a ema_arguments=("--epochs 390 --model-ema --cutmix 1.")
declare -a synthetic_data_arguments=("" "--combine-dataset deepmind_cifar10 --combined-dataset-ratio 0.7")

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

for m in "${model_names[@]}"
do
  if [ "$m" = "wide_resnet34_20" ] || [ "$m" = "wide_resnet28_10_dm" ]; then
    TRAIN_BATCH_SIZE=32
    VAL_BATCH_SIZE=512
    TRAIN_BATCH_SIZE_CONFIG="--batch-size=$TRAIN_BATCH_SIZE"
  elif [ "$m" = "wide_resnet70_16" ] || [ "$m" = "wide_resnet34_10_dm" ] || [ "$m" = "wide_resnet34_20_dm" ] || [ "$m" = "wide_resnet70_16_dm" ]; then
    TRAIN_BATCH_SIZE=16
    VAL_BATCH_SIZE=256
    TRAIN_BATCH_SIZE_CONFIG="--batch-size=$TRAIN_BATCH_SIZE"
  else
    TRAIN_BATCH_SIZE_CONFIG=""
    VAL_BATCH_SIZE=2048
  fi
  for a in "${adv_training_techniques[@]}"
  do
    for s in "${attack_steps[@]}"
    do
      for ema in "${ema_arguments[@]}"
      do
        if [ "$ema" = "" ]; then
          is_ema="no_ema"
        else
          is_ema="ema"
        fi
        for synthetic_data in "${synthetic_data_arguments[@]}";
        do
          if [ "$synthetic_data" = "" ]; then
            is_synthetic=""
          else
            is_synthetic="_synthetic"
          fi
            EXPERIMENT_DIR=${EXPERIMENT}_${m}_${a}_${s}_${is_ema}${is_synthetic}
            {
              if [ "$1" = "train" ]; then
                # if [ -d "${OUTPUT_DIR}/${EXPERIMENT_DIR}" ]; then
                #  echo "Skipping ${OUTPUT_DIR}/${EXPERIMENT_DIR}"
                #  continue
                # fi
                sh scripts/chainer_main.sh "-m torch.distributed.run --nproc_per_node=4 --master_port=6712 train.py" "${DATA_DIR} --config=$BASE_CONFIG --output $OUTPUT_DIR --experiment=$EXPERIMENT_DIR --log-wandb --wandb-project=robust-hw --mean $MEAN --std $STD --model=$m --adv-training=$a --attack-steps=$s $TRAIN_BATCH_SIZE_CONFIG $ema $synthetic_data" $OUTPUT_DIR "$EXPERIMENT_DIR" train 0 
              elif [ "$1" = "validate" ]; then
                python check_end_of_training.py ${OUTPUT_DIR}/${EXPERIMENT_DIR}
                if [ "$?" -eq 0  ]; then # training has finished, validate 
                  sh scripts/chainer_main.sh "validate_robustbench.py" "--data-dir=$DATA_DIR --model=$m --checkpoint=${OUTPUT_DIR}/${EXPERIMENT_DIR}/best.pth.tar --batch-size=$VAL_BATCH_SIZE --eps=$EPS --mean $MEAN --std $STD --gpus=$N_GPUS --log-wandb --log-to-file --aa-state-path ${OUTPUT_DIR}/${EXPERIMENT_DIR}/aa-state.json" $OUTPUT_DIR "$EXPERIMENT_DIR" aa; sleep 1
                else
                  echo "Skipping ${OUTPUT_DIR}/${EXPERIMENT_DIR} because training did not finish."
                fi
              else
                echo "Invalid argument $1"
                exit 1
              fi
            } &
          sleep 1
        done
        sleep 1
      done
      sleep 3
    done
    sleep 3
  done
done
