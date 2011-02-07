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
#	For our evaluation purposes, we compile the kernels with the following options:
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

NVCC = "$(shell which nvcc)"
NVCC_VERSION = $(strip $(shell nvcc --version | grep release | sed 's/.*release //' |  sed 's/,.*//'))

KERNELS = 

#-------------------------------------------------------------------------------
# Gen targets
#-------------------------------------------------------------------------------

GEN_SM20 = -gencode=arch=compute_20,code=\"sm_20,compute_20\" 
GEN_SM13 = -gencode=arch=compute_13,code=\"sm_13,compute_13\" 
GEN_SM10 = -gencode=arch=compute_10,code=\"sm_10,compute_10\" 
#SM_TARGETS = $(GEN_SM20) $(GEN_SM13) $(GEN_SM10)
SM_TARGETS = $(GEN_SM20) 

#-------------------------------------------------------------------------------
# Libs
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# Includes
#-------------------------------------------------------------------------------

CUDA_INC = "$(shell dirname $(NVCC))/../include"
INC = -I. -I$(CUDA_INC) -I.. -I../kernel -I../../Common -I../../ExamplesInclude 

#-------------------------------------------------------------------------------
# Defines
#-------------------------------------------------------------------------------

DEFINES = 

ifeq ($(fermi-ecc), 1)
	# Define if you're planning on using Fermi with ECC. (This enables logic 
	# designed for SM1.0 in order to programmatically coalesce unaligned 
	# scatters becuase they are so prohibitively expensive with ECC) 
	DEFINES += -DFERMI_ECC		
endif

#-------------------------------------------------------------------------------
# Compiler Flags
#-------------------------------------------------------------------------------

ifneq ($(force64), 1)
	# Compile with 32-bit device pointers by default
	ARCH_SUFFIX = i386
	ARCH = -m32
else
	ARCH_SUFFIX = x86_64
	ARCH = -m64
endif

NVCCFLAGS = -Xptxas -v

ifeq (,$(findstring 3.0, $(NVCC_VERSION)))
ifneq ($(abi), 1)
	# Disable the ABI by default for 3.1+
	NVCCFLAGS += -Xptxas -abi=no
endif
endif

ifeq ($(keep), 1)
    NVCCFLAGS += -keep
endif

ifdef maxregisters
    NVCCFLAGS += -maxrregcount $(maxregisters)
endif

#-------------------------------------------------------------------------------
# Dependency Lists
#-------------------------------------------------------------------------------

SORTING_DEPS = 	../radixsort_base.cu \
				../radixsort_early_exit.cu \
				../radixsort_multi_cta.cu \
				../radixsort_single_grid.cu \
				../kernel/radixsort_kernel_common.cu \
				../kernel/radixsort_key_conversion.cu \
				../kernel/radixsort_reduction_kernel.cu \
				../kernel/radixsort_scanscatter_kernel.cu \
				../kernel/radixsort_spine_kernel.cu \
				../kernel/radixsort_single_grid_kernel.cu \
				../../Common/b40c_cuda_properties.cu \
				../../Common/b40c_error_synchronize.cu \
				../../Common/b40c_kernel_data_movement.cu \
				../../Common/b40c_kernel_utils.cu \
				../../Common/b40c_vector_types.cu 
				 


#-------------------------------------------------------------------------------
# (make large) Simple test driver for large-problem sorting
#-------------------------------------------------------------------------------

large: bin/test_large_problem_sorting_$(NVCC_VERSION)_$(ARCH_SUFFIX)

bin/test_large_problem_sorting_$(NVCC_VERSION)_$(ARCH_SUFFIX) : test_large_problem_sorting.cu test_utils.cu $(SORTING_DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/test_large_problem_sorting_$(NVCC_VERSION)_$(ARCH_SUFFIX) test_large_problem_sorting.cu $(NVCCFLAGS) $(ARCH) $(INC) -DUNIX -O3  
	

#-------------------------------------------------------------------------------
# (make small) Illustrates the small-problem sorting enactor that explicitly 
# only sorts 17-bits, and does so using a single kernel launch.  
#-------------------------------------------------------------------------------

small: bin/test_small_problem_sorting_$(NVCC_VERSION)_$(ARCH_SUFFIX)

bin/test_small_problem_sorting_$(NVCC_VERSION)_$(ARCH_SUFFIX) : test_small_problem_sorting.cu test_utils.cu $(SORTING_DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/test_small_problem_sorting_$(NVCC_VERSION)_$(ARCH_SUFFIX) test_small_problem_sorting.cu $(NVCCFLAGS) $(ARCH) $(INC) -DUNIX -O3 

#-------------------------------------------------------------------------------
# (make advanced) Advanced test driver (this can take quite some time to compile as it generates
# kernels for a wide variety of key and value type combinations
#-------------------------------------------------------------------------------

advanced: bin/test_radix_sort_advanced_$(NVCC_VERSION)_$(ARCH_SUFFIX)

bin/test_radix_sort_advanced_$(NVCC_VERSION)_$(ARCH_SUFFIX) : test_radix_sort_advanced.cu test_utils.cu $(SORTING_DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/test_radix_sort_advanced_$(NVCC_VERSION)_$(ARCH_SUFFIX) test_radix_sort_advanced.cu $(NVCCFLAGS) $(ARCH) $(INC) -DUNIX -O3 



#
# Clean
#

clean :
	rm -f bin/test_large_problem_sorting_$(NVCC_VERSION)_$(ARCH_SUFFIX) bin/test_radix_sort_advanced_$(NVCC_VERSION)_$(ARCH_SUFFIX) bin/test_small_problem_sorting_$(NVCC_VERSION)_$(ARCH_SUFFIX)
	rm -f *.i* *.cubin *.cu.c *.cudafe* *.fatbin.c *.ptx *.hash *.cu.cpp *.o




