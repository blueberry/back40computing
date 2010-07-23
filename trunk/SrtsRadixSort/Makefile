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
#		@TechReport{ Merrill:Sorting:2010,
#        	author = "Duane Merrill and Andrew Grimshaw",
#        	title = "Revisiting Sorting for GPGPU Stream Architectures",
#        	year = "2010",
#        	institution = "University of Virginia, Department of Computer Science",
#        	address = "Charlottesville, VA, USA",
#        	number = "CS2010-03"
#		}
#  
#  For more information, see our Google Code project site: 
#  http://code.google.com/p/back40computing/
#  
#  Thanks!
#-------------------------------------------------------------------------------
 
#-------------------------------------------------------------------------------
# Build script for project
#
# IMPORTANT NOTES: 
#	We compile the kernels with the following options:
#
# 		- 32-bit device pointers, regardless of whether we're on a 64-bit 
# 		  machine or not.  64-bit device pointers incur ~10-15% slowdown 
#		  because it causes fairly increased kernel register counts 
#		  which prevents us from being able to meet our targeted occupancies.
#
#		- Without the newly-introduced-for-3.1 ABI (that gives programmers a 
#		  program stack).  Even though this code does not utilize a program 
#		  stack for sorting, the ABI incurs extra register pressure regardless 
#		  and results in a ~1-5% slowdown.   
#
#-------------------------------------------------------------------------------

NVCC = $(shell which nvcc)
NVCC_VERSION = $(strip $(shell nvcc --version | grep release | sed 's/.*release //' |  sed 's/,.*//'))
CUDA_LIB32 = -L$(shell dirname $(NVCC))/../lib32 -L$(shell dirname $(NVCC))/../lib -L$(shell dirname $(NVCC))/../lib64
CUDA_INC = $(shell dirname $(NVCC))/../include

ifneq ($(force64), 1)
	# Compile with 32-bit device pointers by default
	ARCH_SUFFIX = i386
	ARCH = -m32
else
	ARCH_SUFFIX = x86_64
	ARCH = -m64
endif

GEN_SM20 = -gencode=arch=compute_20,code=\"sm_20,compute_20\" 
GEN_SM13 = -gencode=arch=compute_13,code=\"sm_13,compute_13\" 
GEN_SM10 = -gencode=arch=compute_10,code=\"sm_10,compute_10\" 
SM_TARGETS = $(GEN_SM20) $(GEN_SM13) $(GEN_SM10)

NVCCFLAGS = -Xptxas -v

ifneq (,$(findstring 3.1, $(NVCC_VERSION)))
ifneq ($(abi), 1)
	# Disable the ABI by default for 3.1
	NVCCFLAGS += -Xptxas -abi=no
endif
endif

ifeq ($(keep), 1)
    NVCCFLAGS += -keep
endif

ifdef maxregisters
    NVCCFLAGS += -maxrregcount $(maxregisters)
endif


#
# Simple test driver
#

simple: bin/test_radix_sort_simple_$(NVCC_VERSION)

bin/test_radix_sort_simple_$(NVCC_VERSION) : obj/test_radix_sort_simple.cu.sm.$(NVCC_VERSION).o
	mkdir -p bin
	g++ -fPIC $(ARCH) -o bin/test_radix_sort_simple_$(NVCC_VERSION) obj/test_radix_sort_simple.cu.sm.$(NVCC_VERSION).o -L./lib $(CUDA_LIB32) -lcudart -lcutil_$(ARCH_SUFFIX) -lshrutil_$(ARCH_SUFFIX)
	
obj/test_radix_sort_simple.cu.sm.$(NVCC_VERSION).o : test_radix_sort_simple.cu srts_radix_sort.cu kernel/srts_radixsort_kernel_common.cu kernel/srts_reduction_kernel.cu kernel/srts_scanscatter_kernel.cu kernel/srts_spine_kernel.cu test_radix_sort_utils.cu
	mkdir -p obj
	$(NVCC) $(SM_TARGETS) -o obj/test_radix_sort_simple.cu.sm.$(NVCC_VERSION).o -c test_radix_sort_simple.cu $(NVCCFLAGS) $(ARCH) --compiler-options -fno-strict-aliasing  -I. -I$(CUDA_INC) -DUNIX -O3


#
# Advanced test driver (this can take quite some time to compile as it generates
# kernels for a wide variety of key and value type combinations
#

advanced: bin/test_radix_sort_advanced_$(NVCC_VERSION)

bin/test_radix_sort_advanced_$(NVCC_VERSION) : obj/test_radix_sort_advanced.cu.sm.$(NVCC_VERSION).o
	mkdir -p bin
	g++ -fPIC $(ARCH) -o bin/test_radix_sort_advanced_$(NVCC_VERSION) obj/test_radix_sort_advanced.cu.sm.$(NVCC_VERSION).o -L./lib $(CUDA_LIB32) -lcudart -lcutil_$(ARCH_SUFFIX) -lshrutil_$(ARCH_SUFFIX)

obj/test_radix_sort_advanced.cu.sm.$(NVCC_VERSION).o : test_radix_sort_advanced.cu srts_radix_sort.cu kernel/srts_radixsort_kernel_common.cu kernel/srts_reduction_kernel.cu kernel/srts_scanscatter_kernel.cu kernel/srts_spine_kernel.cu test_radix_sort_utils.cu
	mkdir -p obj
	$(NVCC) $(SM_TARGETS) -o obj/test_radix_sort_advanced.cu.sm.$(NVCC_VERSION).o -c test_radix_sort_advanced.cu $(NVCCFLAGS) $(ARCH) --compiler-options -fno-strict-aliasing  -I. -I$(CUDA_INC) -DUNIX -O3


#
# Clean
#

clean :
	rm -f obj/test_radix_sort_simple.cu.sm.$(NVCC_VERSION).o obj/test_radix_sort_advanced.cu.sm.$(NVCC_VERSION).o
	rm -f bin/test_radix_sort_simple_$(NVCC_VERSION) bin/test_radix_sort_advanced_$(NVCC_VERSION)
	rm -f *.i* *.cubin *.cu.c *.cudafe* *.fatbin.c *.ptx *.hash *.cu.cpp




