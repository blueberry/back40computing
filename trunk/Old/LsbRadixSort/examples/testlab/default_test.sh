#!/bin/bash

for DEVICE in 0 
do
	for BIN in test_radix_sort_simple_3.0_i386 test_radix_sort_simple_3.0_x86_64 test_radix_sort_simple_3.1_i386 test_radix_sort_simple_3.1_x86_64
	do
		for N in 25 2048 33554432
		do
			echo "../bin/$BIN --device=$DEVICE --i=50 --n=$N --keys-only"
			../bin/$BIN --device=$DEVICE --i=50 --n=$N --keys-only

			echo "../bin/$BIN --device=$DEVICE --i=50 --n=$N"
			../bin/$BIN --device=$DEVICE --i=50 --n=$N
		done
	done
done
