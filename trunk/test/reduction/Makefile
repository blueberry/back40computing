#/******************************************************************************
# Copyright (c) 2010-2011, Duane Merrill.  All rights reserved.
# Copyright (c) 2011-2013, NVIDIA CORPORATION.  All rights reserved.
#  
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the NVIDIA CORPORATION nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 
# ******************************************************************************/
 
#-------------------------------------------------------------------------------
# Build script for project
#-------------------------------------------------------------------------------

NVCC = "$(shell which nvcc)"
NVCC_VERSION = $(strip $(shell nvcc --version | grep release | sed 's/.*release //' |  sed 's/,.*//'))

KERNELS = 

# detect OS
OSUPPER = $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])

#-------------------------------------------------------------------------------
# Gen targets
#-------------------------------------------------------------------------------

GEN_SM35 = -gencode=arch=compute_35,code=\"sm_35,compute_35\" 
GEN_SM30 = -gencode=arch=compute_30,code=\"sm_30,compute_30\" 
GEN_SM20 = -gencode=arch=compute_20,code=\"sm_20,compute_20\" 
GEN_SM13 = -gencode=arch=compute_13,code=\"sm_13,compute_13\" 
GEN_SM11 = -gencode=arch=compute_11,code=\"sm_11,compute_11\" 
GEN_SM10 = -gencode=arch=compute_10,code=\"sm_10,compute_10\" 
#SM_TARGETS = $(GEN_SM20) $(GEN_SM13) $(GEN_SM11)
SM_TARGETS = $(GEN_SM20) 

#-------------------------------------------------------------------------------
# Tune arch
#-------------------------------------------------------------------------------

ifdef tunearch
    TUNE_ARCH = $(tunearch)
else 
	TUNE_ARCH = 200
endif

ifeq ($(TUNE_ARCH), 350) 
	TUNE_SM_TARGETS = $(GEN_SM35)
endif
ifeq ($(TUNE_ARCH), 300) 
	TUNE_SM_TARGETS = $(GEN_SM30)
endif
ifeq ($(TUNE_ARCH), 200) 
	TUNE_SM_TARGETS = $(GEN_SM20)
endif
ifeq ($(TUNE_ARCH), 130) 
	TUNE_SM_TARGETS = $(GEN_SM13)
endif
ifeq ($(TUNE_ARCH), 110) 
	TUNE_SM_TARGETS = $(GEN_SM11)
endif
ifeq ($(TUNE_ARCH), 100) 
	TUNE_SM_TARGETS = $(GEN_SM10)
endif


#-------------------------------------------------------------------------------
# Tune size
#-------------------------------------------------------------------------------

ifdef tunesize
    TUNE_SIZE = $(tunesize)
else 
	TUNE_SIZE = 0
endif



#-------------------------------------------------------------------------------
# Includes
#-------------------------------------------------------------------------------

CUDA_INC = "$(shell dirname $(NVCC))/../include"
INC = -I$(CUDA_INC) -I.. -I../.. 

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

NVCCFLAGS = -Xptxas -v -Xopencc -OPT:Olimit=0 -Xcudafe -\#
TUNEFLAGS =

ifeq (WIN_NT, $(findstring WIN_NT, $(OSUPPER)))
	NVCCFLAGS += -Xcompiler /bigobj -Xcompiler /Zm500

	# Tuning requires compiling for small host code footprint (lest we blow out the compiler)
	TUNEFLAGS += -Xcompiler /Os
endif

ifeq (,$(findstring 3.0, $(NVCC_VERSION)))
ifneq ($(abi), 1)
	# Disable the ABI by default for 3.1+
	NVCCFLAGS += -Xptxas -abi=no 
endif
endif

ifeq ($(keep), 1)
    NVCCFLAGS += -keep
endif

ifeq ($(verbose), 1)
    NVCCFLAGS += -v
endif

ifdef maxregisters
    NVCCFLAGS += -maxrregcount $(maxregisters)
endif

ifeq (4.1, $(findstring 4.1, $(NVCC_VERSION)))
	# Disable nvvm for 4.1 
	NVCCFLAGS += -open64
endif 

#-------------------------------------------------------------------------------
# Dependency Lists
#-------------------------------------------------------------------------------

DEPS = 			./Makefile \
				./test_reduction.h \
				../b40c_test_util.h \
				$(wildcard ../../b40c/util/*.cuh) \
				$(wildcard ../../b40c/util/**/*.cuh) \
				$(wildcard ../../b40c/reduction/*.cuh) \
				$(wildcard ../../b40c/reduction/**/*.cuh) 
				 


#-------------------------------------------------------------------------------
# (make simple) Simple example driver for reduction
#-------------------------------------------------------------------------------

simple: bin/simple_reduction_$(NVCC_VERSION)_$(ARCH_SUFFIX)

bin/simple_reduction_$(NVCC_VERSION)_$(ARCH_SUFFIX) : simple_reduction.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/simple_reduction_$(NVCC_VERSION)_$(ARCH_SUFFIX) simple_reduction.cu $(NVCCFLAGS) $(ARCH) $(INC) -O3  



#-------------------------------------------------------------------------------
# (make test) Test driver for reduction
#-------------------------------------------------------------------------------

test: bin/test_reduction_$(NVCC_VERSION)_$(ARCH_SUFFIX)

bin/test_reduction_$(NVCC_VERSION)_$(ARCH_SUFFIX) : test_reduction.cu $(DEPS) test_reduction.h
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/test_reduction_$(NVCC_VERSION)_$(ARCH_SUFFIX) test_reduction.cu $(NVCCFLAGS) $(ARCH) $(INC) -O3  

	
#-------------------------------------------------------------------------------
# (make tune) Evaluates the space of reduction configuration types
#-------------------------------------------------------------------------------

tune: bin/tune_reduction_$(NVCC_VERSION)_$(ARCH_SUFFIX)_sm$(TUNE_ARCH)_u$(TUNE_SIZE)B

bin/tune_reduction_$(NVCC_VERSION)_$(ARCH_SUFFIX)_sm$(TUNE_ARCH)_u$(TUNE_SIZE)B : tune_reduction.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(TUNE_SM_TARGETS) -o bin/tune_reduction_$(NVCC_VERSION)_$(ARCH_SUFFIX)_sm$(TUNE_ARCH)_u$(TUNE_SIZE)B tune_reduction.cu $(NVCCFLAGS) $(ARCH) $(INC) $(TUNEFLAGS) -DTUNE_ARCH=$(TUNE_ARCH) -DTUNE_SIZE=$(TUNE_SIZE)  


#-------------------------------------------------------------------------------
# (make compare) Compares reduction performance against other libraries (e.g., Thrust)
#-------------------------------------------------------------------------------

compare: bin/compare_reduction_$(NVCC_VERSION)_$(ARCH_SUFFIX)

bin/compare_reduction_$(NVCC_VERSION)_$(ARCH_SUFFIX) : compare_reduction.cu $(DEPS) test_reduction.h
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/compare_reduction_$(NVCC_VERSION)_$(ARCH_SUFFIX) compare_reduction.cu $(NVCCFLAGS) $(ARCH) $(INC) -O3  


#
# Clean
#

clean :
	rm -f bin/*_$(NVCC_VERSION)_$(ARCH_SUFFIX)* 
	rm -f *.i* *.cubin *.cu.c *.cudafe* *.fatbin.c *.ptx *.hash *.cu.cpp *.o




