#!/bin/bash

export CUDA_PROFILE=1
export CUDA_PROFILE_CSV=1

for KEYBYTES in 4 8 1 2
do
	for VALUEBYTES in 4 0 8 16
	do

		export CUDA_PROFILE_LOG=output/$HOSTNAME.Cuda30.${KEYBYTES}ByteKeys.${VALUEBYTES}ByteValues.MaxSize268M.profile.a
		export CUDA_PROFILE_CONFIG=profile_config.dat.tesla.a
		echo ../bin/test_radix_sort_advanced_3.0 --noverify --i=4 --n-input=input/problem_sizes_32_268M.dat --key-bytes=$KEYBYTES --value-bytes=$VALUEBYTES 
		../bin/test_radix_sort_advanced_3.0 --noverify --i=4 --n-input=input/problem_sizes_32_268M.dat --key-bytes=$KEYBYTES --value-bytes=$VALUEBYTES 

		export CUDA_PROFILE_LOG=output/$HOSTNAME.Cuda30.${KEYBYTES}ByteKeys.${VALUEBYTES}ByteValues.MaxSize268M.profile.b
		export CUDA_PROFILE_CONFIG=profile_config.dat.tesla.b
		echo ../bin/test_radix_sort_advanced_3.0 --noverify --i=4 --n-input=input/problem_sizes_32_268M.dat --key-bytes=$KEYBYTES --value-bytes=$VALUEBYTES 
		../bin/test_radix_sort_advanced_3.0 --noverify --i=4 --n-input=input/problem_sizes_32_268M.dat --key-bytes=$KEYBYTES --value-bytes=$VALUEBYTES 

	done
done


