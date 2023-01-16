#!/bin/bash

# Arguments: sh chainer.sh TRAINSCRIPT TRAINCONFIG
# or Arguments: sh chainer.sh TRAINSCRIPT TRAINCONFIG repeat

# max number of jobs to chain
export MAX=36
# number of nodes, we need to specify here instead of in the submit.sh job
export NNODES=1
# train script, should just be a filename. `python $TRAINSCRIPT`
export TRAINSCRIPT=$1
# train config, should be a string as used in the python cmd line argument like "--arg1=5 --arg2=6"
export TRAINCONFIG="$2"
export EXP_OUT_DIR="$3"
export EXP_NAME="$4"
export TRAIN_OR_AA="$5"
# name of the chain: for logging files and job names
export CHAINNAME=chain_${TRAIN_OR_AA}_config_and_iter_${EXP_NAME}
# script with specific environment settings for the job
export STARTSCRIPT=scripts/chainer_recursion.sh
# bank to use for the job allocation
export BBANK=asyncmlb
# time limit for the job
export BTIME=12:00
# outdir of the logs
export BOUTDIR=logs

repeat=$6

if [ $# -lt 2 ]; then
  echo "Number of arguments not expected. Exiting.."
  exit 0
fi

if [ $# -gt 6 ]; then
  echo "Number of arguments not expected. Exiting.."
  exit 0
fi

if [ $# -eq 6 ]; then
  # Arguments: sh chainer.sh TRAINSCRIPT TRAINCONFIG repeat
  # if there's an argument to chainer.sh script: sh chainer arg0
  # meaning, there was a job running before this one
  lastcount=$repeat
  currcount=$(expr $lastcount + 1)
  if [ "$currcount" -gt $MAX ]; then
    echo "Chained more than $MAX jobs. Stop chaining..."
    exit 0
  fi

  name="${CHAINNAME}_${currcount}"
  dep_name="${CHAINNAME}_${lastcount}"
  outputname="${name}_%J"
  # Add the checkpoint to resume to the training config if we are at the first repeat
  if [ "$TRAIN_OR_AA" = "train" ] && [ $currcount -eq 1 ]; then
    TRAINCONFIG="$TRAINCONFIG --resume $EXP_OUT_DIR/$EXP_NAME/last.pth.tar"
  fi

  CMD="bsub -nnodes $NNODES -alloc_flags ipisolate -W $BTIME -G $BBANK -J $name -outdir $BOUTDIR -oo ${BOUTDIR}/${outputname}.out -w ended(${dep_name})"
fi

if [ $# -eq 5 ]; then
  # Arguments: sh chainer.sh TRAINSCRIPT TRAINCONFIG
  # if there's not an argument to chainer.sh script: sh chainer
  currcount=0
  name="${CHAINNAME}_${currcount}"
  outputname="${name}_%J"
  CMD="bsub -nnodes $NNODES -alloc_flags ipisolate -W $BTIME -G $BBANK -J $name -outdir $BOUTDIR -oo ${BOUTDIR}/${outputname}.out"
fi

echo "chainer main: $CMD sh $STARTSCRIPT \"$TRAINSCRIPT\" \"$TRAINCONFIG\" $EXP_OUT_DIR $EXP_NAME $TRAIN_OR_AA $currcount"
$CMD sh $STARTSCRIPT "$TRAINSCRIPT" "$TRAINCONFIG" $EXP_OUT_DIR $EXP_NAME $TRAIN_OR_AA $currcount
