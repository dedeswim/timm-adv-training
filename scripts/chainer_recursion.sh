#!/bin/sh

# Chaining bsub requests
# see chainer_main.sh for what the arguments are
#arr_2=($2)
echo "chainer recursion: sh scripts/chainer_main.sh "$1" "$2" $3 $4 $5 $6 &"
sh scripts/chainer_main.sh "$1" "$2" $3 $4 $5 $6 &

echo "chainer job: python $1 $2"
#jpython $1 ${arr_2[@]}
python $1 $2
