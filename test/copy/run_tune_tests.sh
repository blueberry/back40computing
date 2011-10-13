#!/bin/bash

#for i in 110 130 200; do time make tune tunearch=$i tunesize=8; done

OPTIONS="--i=4 --device=1"

if [ $# -eq 0 ]
then
	echo "Usage: $0 <ARCH>"
	exit 1
fi

ARCH=$1

mkdir -p eval/sm${ARCH}


for WORD in 1 2 4 8
do
	# 32K..32M
	for (( N=1024*32; N<=1024*1024*32; N*=4 ))
	do

		CMD="./bin/tune_copy_4.0_i386_sm${ARCH}_u${WORD}B ${OPTIONS} --n=${N}"
		echo "$CMD > eval/sm${ARCH}/${ARCH}_${WORD}B_${N}.txt"
		$CMD > eval/sm${ARCH}/${ARCH}_${WORD}B_${N}.txt

	done 
done
