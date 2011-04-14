#!/bin/bash

export CUDA_PROFILE=1
export CUDA_PROFILE_CSV=1
export CUDA_PROFILE_CONFIG=profile_config.dat.fermi

for KEYBYTES in 1 2 4 8
do
	for VALUEBYTES in 0 4 8 16
	do

		export CUDA_PROFILE_LOG=output/$HOSTNAME.Cuda30.${KEYBYTES}ByteKeys.${VALUEBYTES}ByteValues.MaxSize268M.profile.a

		echo ../bin/test_radix_sort_advanced_3.0 --noverify --i=4 --n-input=input/problem_sizes_32_268M.dat --key-bytes=$KEYBYTES --value-bytes=$VALUEBYTES 
		../bin/test_radix_sort_advanced_3.0 --noverify --i=4 --n-input=input/problem_sizes_32_268M.dat --key-bytes=$KEYBYTES --value-bytes=$VALUEBYTES 

	done
done


