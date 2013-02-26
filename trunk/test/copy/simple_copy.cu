/******************************************************************************
 * Copyright (c) 2010-2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2013, NVIDIA CORPORATION.  All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/


/******************************************************************************
 * Simple test driver program for copy.
 ******************************************************************************/

#include <stdio.h> 
#include <b40c/copy/enactor.cuh>

// Test utils
#include "b40c_test_util.h"

#pragma warning(disable : 4344)

using namespace b40c;


/******************************************************************************
 * Utility Routines
 ******************************************************************************/


/**
 * Example showing syntax for invoking templated member functions from 
 * a templated function
 */
template <typename T, b40c::copy::ProbSizeGenre PROBLEM_SIZE_GENRE>
void TemplatedSubroutineCopy(
	b40c::copy::Enactor &copy_enactor,
	T *d_dest, 
	T *d_src,
	int num_elements)
{
	copy_enactor.template Copy<PROBLEM_SIZE_GENRE>(d_dest, d_src, num_elements * sizeof(T));
}


/******************************************************************************
 * Main
 ******************************************************************************/

int main(int argc, char** argv)
{
	CommandLineArgs args(argc, argv);

	// Usage/help
    if (args.CheckCmdLineFlag("help") || args.CheckCmdLineFlag("h")) {
    	printf("\nsimple_copy [--device=<device index>]\n");
    	return 0;
    }

	DeviceInit(args);

	typedef unsigned int T;
	const int NUM_ELEMENTS = 10;

	// Allocate and initialize host data
	T h_src[NUM_ELEMENTS];
	for (int i = 0; i < NUM_ELEMENTS; i++) {
		h_src[i] = i;
	}
	
	// Allocate and initialize device data
	T *d_src, *d_dest;
	cudaMalloc((void**) &d_src, sizeof(T) * NUM_ELEMENTS);
	cudaMalloc((void**) &d_dest, sizeof(T) * NUM_ELEMENTS);
	cudaMemcpy(d_src, h_src, sizeof(T) * NUM_ELEMENTS, cudaMemcpyHostToDevice);
	
	// Create a copy enactor
	b40c::copy::Enactor copy_enactor;
	
	//
	// Example 1: Enact simple copy using internal tuning heuristics
	//
	copy_enactor.Copy(d_dest, d_src, NUM_ELEMENTS * sizeof(T));
	
	printf("Simple copy: "); CompareDeviceResults(h_src, d_dest, NUM_ELEMENTS); printf("\n");
	
	
	//
	// Example 2: Enact simple copy using "large problem" tuning configuration
	//
	copy_enactor.Copy<b40c::copy::LARGE_SIZE>(d_dest, d_src, NUM_ELEMENTS * sizeof(T));

	printf("Large-tuned copy: "); CompareDeviceResults(h_src, d_dest, NUM_ELEMENTS); printf("\n");

	
	//
	// Example 3: Enact simple copy using "small problem" tuning configuration
	//
	copy_enactor.Copy<b40c::copy::SMALL_SIZE>(d_dest, d_src, NUM_ELEMENTS * sizeof(T));
	
	printf("Small-tuned copy: "); CompareDeviceResults(h_src, d_dest, NUM_ELEMENTS); printf("\n");

	
	//
	// Example 4: Enact simple copy using a templated subroutine function
	//
	TemplatedSubroutineCopy<T, b40c::copy::UNKNOWN_SIZE>(copy_enactor, d_dest, d_src, NUM_ELEMENTS);
	
	printf("Templated subroutine copy: "); CompareDeviceResults(h_src, d_dest, NUM_ELEMENTS); printf("\n");

	
	//
	// Example 5: Enact simple copy using custom tuning configuration (base copy enactor)
	//
	typedef b40c::copy::Policy<
		T, 
		unsigned long long,
		b40c::copy::SM20, 
		8, 8, 7, 1, 0,
		b40c::util::io::ld::cg,
		b40c::util::io::st::cs,
		true, 
		false> CustomPolicy;
	
	copy_enactor.Copy<CustomPolicy>(d_dest, d_src, NUM_ELEMENTS);

	printf("Custom copy: "); CompareDeviceResults(h_src, d_dest, NUM_ELEMENTS); printf("\n");

	return 0;
}

