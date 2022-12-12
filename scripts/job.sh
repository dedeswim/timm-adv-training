declare -a model_names=("resnet18_32" "wide_resnet28_10" "wide_resnet34_10" "wide_resnet34_20")
declare -a adv_training_techniques=("pgd" "trades")
declare -a attack_steps=("1" "2" "5" "7" "10")
declare -a ema_arguments=("--epochs=300 --decay-milestones 200 --model-ema --cutmix 1." "")

export DATASET=cifar10
export DATA_DIR=/p/vast1/MLdata
export BBANK=safeml
export BTIME=12:00
export BOUTDIR=logs
export BASE_CONFIG=configs/sweep-cifar10.yaml
export EXPERIMENT=robust-hw
export OUTPUT_DIR=/p/gpfs1/robustHW
export MEAN="0.4914 0.4822 0.4465"
export STD="0.2023 0.1994 0.2010"

# Validation-only params
export VAL_BATCH_SIZE=2048
export EPS=8
export N_GPUS=4

for m in "${model_names[@]}"
do
  if [ "$m" = "wide_resnet34_20" ]; then
    TRAIN_BATCH_SIZE=64
    VAL_BATCH_SIZE=1024
    TRAIN_BATCH_SIZE_CONFIG="--batch-size=$TRAIN_BATCH_SIZE"
  else if [ "$m" = "wide_resnet70_16" ]; then
    TRAIN_BATCH_SIZE=32
    VAL_BATCH_SIZE=512
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
          EXPERIMENT_DIR=${EXPERIMENT}_${m}_${a}_${s}_${is_ema}
          {
            # sh scripts/chainer_main.sh "-m torch.distributed.run --nproc_per_node=4 --master_port=6712 train.py" "${DATA_DIR} --config=$BASE_CONFIG --experiment=$EXPERIMENT_DIR --log-wandb --wandb-project=robust-hw --mean $MEAN --std $STD --model=$m --adv-training=$a --attack-steps=$s $TRAIN_BATCH_SIZE_CONFIG $ema" $OUTPUT_DIR $EXPERIMENT_DIR train
            sh scripts/chainer_main.sh "validate_robustbench.py --data-dir=$DATA_DIR --model=$m --checkpoint=${OUTPUT_DIR}/${EXPERIMENT_DIR}/best.pth.tar --batch-size=$VAL_BATCH_SIZE --eps=$EPS --mean $MEAN --std $STD --gpus=$N_GPUS --aa-state-path ${OUTPUT_DIR}/${EXPERIMENT_DIR}/aa-state.json" $OUTPUT_DIR $EXPERIMENT_DIR aa
          } &
      done
    done
  done
done
