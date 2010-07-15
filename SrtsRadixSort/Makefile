#-------------------------------------------------------------------------------
#
# Copyright 2010 Duane Merrill
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. 
#
# 
#  
#  AUTHORS' REQUEST: 
#  
#  		If you use|reference|benchmark this code, please cite our Technical 
#  		Report (http://www.cs.virginia.edu/~dgm4d/papers/RadixSortTR.pdf):
#  
#  		Duane Merrill and Andrew Grimshaw, "Revisiting Sorting for GPGPU 
#  		Stream Architectures," University of Virginia, Department of 
#  		Computer Science, Charlottesville, VA, USA, Technical Report 
#  		CS2010-03, 2010.
#  
#  For more information, see our Google Code project site: 
#  http://code.google.com/p/back40computing/
#  
#  Thanks!
#-------------------------------------------------------------------------------
 
#-------------------------------------------------------------------------------
# Build script for project
#
# IMPORTANT NOTE: We need to compile the kernels to use 32-bit device 
# pointers.  Otherwise our kernel register counts will skyrocket and we won't be 
# able to meet our targeted occupancy.
#
#-------------------------------------------------------------------------------

NVCC = $(shell which nvcc) 
CUDA_LIB32 = $(shell dirname $(NVCC))/../lib32
CUDA_INC = $(shell dirname $(NVCC))/../inc
ARCH = -m32

NVCCFLAGS = -Xptxas -v

ifeq ($(keep), 1)
    NVCCFLAGS += -keep
endif

bin/test_radix_sort : obj/test_radix_sort.cu_sm.o
	mkdir -p bin
	g++ -fPIC $(ARCH) -o bin/test_radix_sort obj/test_radix_sort.cu_sm.o -L./lib -L$(CUDA_LIB32) -lcudart -lcutil_i386 -lshrutil_i386
	
obj/test_radix_sort.cu_sm.o : test_radix_sort.cu kernel/srts_radixsort_kernel_common.cu kernel/srts_reduction_kernel.cu kernel/srts_scanscatter_kernel.cu kernel/srts_spine_kernel.cu srts_verifier.cu
	mkdir -p obj
	$(NVCC) -gencode=arch=compute_20,code=\"sm_20,compute_20\" -gencode=arch=compute_13,code=\"sm_13,compute_13\" -gencode=arch=compute_10,code=\"sm_10,compute_10\" -o obj/test_radix_sort.cu_sm.o -c test_radix_sort.cu $(NVCCFLAGS) $(ARCH) --compiler-options -fno-strict-aliasing  -I. -I$(CUDA_INC) -DUNIX -O3

clean :
	rm -f obj/test_radix_sort.cu_sm.o
	rm -f bin/test_radix_sort




