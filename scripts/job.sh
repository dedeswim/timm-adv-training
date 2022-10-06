declare -a model_names=("resnet18_32" "wide_resnet28_10" "wide_resnet34_10")
declare -a adv_training_techniques=("trades")
declare -a attack_steps=("1" "2" "5" "7" "10")

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
  for a in "${adv_training_techniques[@]}"
  do
    for s in "${attack_steps[@]}"
    do
      EXPERIMENT_DIR=${EXPERIMENT}_${m}_${a}_${s}
      {
        sh scripts/chainer_main.sh "-m torch.distributed.run --nproc_per_node=4 --master_port=6712 train.py" "${DATA_DIR} --config=$BASE_CONFIG --experiment=$EXPERIMENT_DIR --mean $MEAN --std $STD --model=$m --adv-training=$a --attack-steps=$s" $OUTPUT_DIR $EXPERIMENT_DIR train
        # sh scripts/chainer_main.sh validate_robustbench.py "--data-dir=$DATA_DIR --model=$m --checkpoint=${OUTPUT_DIR}/${EXPERIMENT_DIR}/best.pth.tar --batch-size=$VAL_BATCH_SIZE --eps=$EPS --mean $MEAN --std $STD --gpus=$N_GPUS" $OUTPUT_DIR $EXPERIMENT_DIR aa
      } &
    done
  done
done
