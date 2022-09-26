declare -a model_names=("resnet18_32" "wide_resnet28_10" "wide_resnet34_10" "wide_resnet34_20")
declare -a adv_training_techniques=("pgd")
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
        bsub -o logs/${m}_${a}_${s}_train_%J.out -nnodes 1 -W $BTIME -G $BANK \
          python -m torch.distributed.run --nproc_per_node=4 --master_port=6712 \
          train.py ${DATA_DIR} --config=$BASE_CONFIG --experiment=$EXPERIMENT_DIR \
          --mean=$MEAN --std=$STD --model=$m --adv-training=$a --attack-steps=$s \
        && bsub -o logs/${m}_${a}_${s}_aa_%J.out -nnodes 1 -W $BTIME -G $BANK \
          python validate_robustbench.py --data-dir=$DATA_DIR --model=$m --checkpoint=$EXPERIMENT_DIR/best.pth.tar --batch_size=$VAL_BATCH_SIZE \
          --eps=$EPS --mean=$MEAN --std=STD --gpus=N_GPUS
      } &
    done
  done
done

validate_robustbench.py --data-dir /p/vast1/MLdata --dataset cifar10 --model wide_resnet34_10 --batch-size 2048 --checkpoint /p/gpfs1/robustHW/trades-repro-5/last.pth.tar --eps 8 --mean 0.4914 0.4822 0.4465 --std 0.2023 0.1994 0.2010 --log-wandb --gpus 4