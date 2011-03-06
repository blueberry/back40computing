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
#  For more information, see our Google Code project site: 
#  http://code.google.com/p/back40computing/
#  
#  Thanks!
#-------------------------------------------------------------------------------
 
#-------------------------------------------------------------------------------
# Build script for project
#-------------------------------------------------------------------------------

NVCC = $(shell which nvcc)
NVCC_VERSION = $(strip $(shell nvcc --version | grep release | sed 's/.*release //' |  sed 's/,.*//'))

#-------------------------------------------------------------------------------
# Gen targets
#-------------------------------------------------------------------------------

GEN_SM20 = -gencode=arch=compute_20,code=\"sm_20,compute_20\" 
GEN_SM13 = -gencode=arch=compute_13,code=\"sm_13,compute_13\" 
GEN_SM10 = -gencode=arch=compute_10,code=\"sm_10,compute_10\" 
SM_TARGETS = $(GEN_SM20) $(GEN_SM13) 

#-------------------------------------------------------------------------------
# Libs
#-------------------------------------------------------------------------------

CUDA_LIB = -L$(shell dirname $(NVCC))/../lib32 -L$(shell dirname $(NVCC))/../lib -L$(shell dirname $(NVCC))/../lib64

#-------------------------------------------------------------------------------
# Includes
#-------------------------------------------------------------------------------

CUDA_INC = $(shell dirname $(NVCC))/../include
INC = -I. -I$(CUDA_INC) -I.. -I../kernel -I../../Common -I../../ExamplesInclude

#-------------------------------------------------------------------------------
# Defines
#-------------------------------------------------------------------------------

DEFINES = 


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

BFS_DEPS = 		../../Common/b40c_cuda_properties.cu \
				../../Common/b40c_error_synchronize.cu \
				../../Common/b40c_kernel_data_movement.cu \
				../../Common/b40c_kernel_utils.cu \
				../../Common/b40c_vector_types.cu \
				../kernel/bfs_kernel_common.cu \
				../kernel/bfs_single_grid_kernel.cu \
				../bfs_base.cu \
				../bfs_single_grid.cu
				 


#-------------------------------------------------------------------------------
# Simple test driver 
#-------------------------------------------------------------------------------

bfs: bin/test_bfs_$(NVCC_VERSION)_$(ARCH_SUFFIX)

bin/test_bfs_$(NVCC_VERSION)_$(ARCH_SUFFIX) : test_bfs.cu test_utils.cu $(BFS_DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/test_bfs_$(NVCC_VERSION)_$(ARCH_SUFFIX) test_bfs.cu $(NVCCFLAGS) $(ARCH) $(INC) -DUNIX -O3  
	



#-------------------------------------------------------------------------------
# Clean
#-------------------------------------------------------------------------------

clean :
	rm -f bin/test_bfs_$(NVCC_VERSION)_$(ARCH_SUFFIX) 
	rm -f *.i* *.cubin *.cu.c *.cudafe* *.fatbin.c *.ptx *.hash *.cu.cpp *.o




