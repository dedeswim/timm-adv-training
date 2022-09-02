#!/bin/sh

# Chaining bsub requests
# see chainer_main.sh for what the arguments are
#arr_2=($2)
echo "chainer recursion: sh chainer_main.sh $1 $2 $3 &"
sh chainer_main.sh $1 "$2" $3 &

echo "chainer job: python $1 $2"
#jpython $1 ${arr_2[@]}
python $1 $2
