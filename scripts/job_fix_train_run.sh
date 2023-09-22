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

m=wide_resnet34_20_dm
a=pgd
s=2
is_ema="ema"
is_synthetic=""
synthetic_data=""

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
  
EXPERIMENT_DIR=${EXPERIMENT}_${m}_${a}_${s}_${is_ema}${is_synthetic}
sh scripts/chainer_main.sh "-m torch.distributed.run --nproc_per_node=4 --master_port=6712 train.py" "${DATA_DIR} --config=$BASE_CONFIG --output $OUTPUT_DIR --experiment=$EXPERIMENT_DIR --log-wandb --wandb-project=robust-hw --mean $MEAN --std $STD --model=$m --adv-training=$a --attack-steps=$s $TRAIN_BATCH_SIZE_CONFIG $ema $synthetic_data" $OUTPUT_DIR "$EXPERIMENT_DIR" train 0 &

sleep 1

m=wide_resnet34_20
a=trades
s=2
is_ema="ema"
is_synthetic=""
synthetic_data=""

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
  
EXPERIMENT_DIR=${EXPERIMENT}_${m}_${a}_${s}_${is_ema}${is_synthetic}
sh scripts/chainer_main.sh "-m torch.distributed.run --nproc_per_node=4 --master_port=6712 train.py" "${DATA_DIR} --config=$BASE_CONFIG --output $OUTPUT_DIR --experiment=$EXPERIMENT_DIR --log-wandb --wandb-project=robust-hw --mean $MEAN --std $STD --model=$m --adv-training=$a --attack-steps=$s $TRAIN_BATCH_SIZE_CONFIG $ema $synthetic_data" $OUTPUT_DIR "$EXPERIMENT_DIR" train 0 &

sleep 1

m=wide_resnet70_16
a=pgd
s=2
is_ema="ema"
is_synthetic="_synthetic"
synthetic_data="--combine-dataset deepmind_cifar10 --combined-dataset-ratio 0.7"

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
  
EXPERIMENT_DIR=${EXPERIMENT}_${m}_${a}_${s}_${is_ema}${is_synthetic}
sh scripts/chainer_main.sh "-m torch.distributed.run --nproc_per_node=4 --master_port=6712 train.py" "${DATA_DIR} --config=$BASE_CONFIG --output $OUTPUT_DIR --experiment=$EXPERIMENT_DIR --log-wandb --wandb-project=robust-hw --mean $MEAN --std $STD --model=$m --adv-training=$a --attack-steps=$s $TRAIN_BATCH_SIZE_CONFIG $ema $synthetic_data" $OUTPUT_DIR "$EXPERIMENT_DIR" train 0 &

sleep 1

m=wide_resnet70_16
a=trades
s=7
is_ema="ema"
is_synthetic=""
synthetic_data=""

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
  
EXPERIMENT_DIR=${EXPERIMENT}_${m}_${a}_${s}_${is_ema}${is_synthetic}
sh scripts/chainer_main.sh "-m torch.distributed.run --nproc_per_node=4 --master_port=6712 train.py" "${DATA_DIR} --config=$BASE_CONFIG --output $OUTPUT_DIR --experiment=$EXPERIMENT_DIR --log-wandb --wandb-project=robust-hw --mean $MEAN --std $STD --model=$m --adv-training=$a --attack-steps=$s $TRAIN_BATCH_SIZE_CONFIG $ema $synthetic_data" $OUTPUT_DIR "$EXPERIMENT_DIR" train 0 &