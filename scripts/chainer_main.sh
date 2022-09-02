#!/bin/bash

# Arguments: sh chainer.sh TRAINSCRIPT TRAINCONFIG
# or Arguments: sh chainer.sh TRAINSCRIPT TRAINCONFIG repeat

# max number of jobs to chain
export MAX=3
# number of nodes, we need to specify here instead of in the submit.sh job
export NNODES=1
# train script, should just be a filename. `python $TRAINSCRIPT`
#export TRAINSCRIPT=chainer_test.py
export TRAINSCRIPT=$1
# train config, should be a string as used in the python cmd line argument like "--arg1=5 --arg2=6"
export TRAINCONFIG="$2"
# name of the chain: for logging files and job names
export CHAINNAME=chain_${TRAINSCRIPT}_config_and_iter_${TRAINCONFIG//[^[:alnum:]]/}
# script with specific environment settings for the job
export STARTSCRIPT=chainer_recursion.sh
# bank to use for the job allocation
export BBANK=safeml
# time limit for the job
#export BTIME=12:00
export BTIME=00:01
# outdir of the logs
export BOUTDIR=logs

repeat=$3

if [ $# -lt 1 ]; then
  echo "Number of arguments not expected. Exiting.."
  exit 0
fi

if [ $# -gt 3 ]; then
  echo "Number of arguments not expected. Exiting.."
  exit 0
fi

if [ $# -eq 3 ]; then
  # Arguments: sh chainer.sh TRAINSCRIPT TRAINCONFIG repeat
  # if there's an argument to chainer.sh script: sh chainer arg0
  # meaning, there was a job running before this one
  lastcount=$repeat
  currcount=$(expr $lastcount + 1)
  if [ "$currcount" -gt "$MAX" ]; then
    echo "Chained more than $MAX jobs. Stop chaining..."
    exit 0
  fi

  name="${CHAINNAME}_${currcount}"
  dep_name="${CHAINNAME}_${lastcount}"
  outputname="${name}_%J"
  CMD="bsub -nnodes $NNODES -alloc_flags ipisolate -W $BTIME -G $BBANK -J $name -outdir $BOUTDIR -oo ${BOUTDIR}/${outputname}.out -w ended(${dep_name})"
fi

if [ $# -eq 2 ]; then
  # Arguments: sh chainer.sh TRAINSCRIPT TRAINCONFIG
  # if there's not an argument to chainer.sh script: sh chainer
  currcount=0
  name="${CHAINNAME}_${currcount}"
  outputname="${name}_%J"
  CMD="bsub -nnodes $NNODES -alloc_flags ipisolate -W $BTIME -G $BBANK -J $name -outdir $BOUTDIR -oo ${BOUTDIR}/${outputname}.out"
fi

echo "chainer main: $CMD sh $STARTSCRIPT $TRAINSCRIPT $2 $currcount"
$CMD sh $STARTSCRIPT $TRAINSCRIPT "$2" $currcount
