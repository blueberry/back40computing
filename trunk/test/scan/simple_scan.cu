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
 * Simple test driver program for scan.
 ******************************************************************************/

#include <stdio.h> 
#include <b40c/scan/enactor.cuh>

// Test utils
#include "b40c_test_util.h"


/******************************************************************************
 * Utility Routines
 ******************************************************************************/


/**
 * Max binary scan operator
 */
template <typename T>
struct Max
{
	// Associative reduction operator
	__host__ __device__ __forceinline__ T operator()(const T &a, const T &b)
	{
		return (a > b) ? a : b;
	}

	// Identity operator
	__host__ __device__ __forceinline__ T operator()()
	{
		return 0;
	}
};


/**
 * Example showing syntax for invoking templated member functions from 
 * a templated function
 */
template <
	bool EXCLUSIVE_SCAN,
	bool COMMUTATIVE,
	typename T,
	typename SizeT,
	typename ReductionOp,
	typename IdentityOp>
void TemplatedSubroutineScan(
	b40c::scan::Enactor &scan_enactor,
	T *d_dest, 
	T *d_src,
	SizeT num_elements,
	ReductionOp scan_op,
	IdentityOp identity_op)
{
	scan_enactor.template Scan<EXCLUSIVE_SCAN, COMMUTATIVE>(
		d_dest, d_src, num_elements, scan_op, identity_op);
}


/******************************************************************************
 * Main
 ******************************************************************************/

int main(int argc, char** argv)
{
	b40c::CommandLineArgs args(argc, argv);

	// Usage/help
    if (args.CheckCmdLineFlag("help") || args.CheckCmdLineFlag("h")) {
    	printf("\nsimple_scan [--device=<device index>]\n");
    	return 0;
    }

    b40c::DeviceInit(args);
    bool verbose = args.CheckCmdLineFlag("v");

    // Define our problem type
	typedef unsigned int T;
	const int NUM_ELEMENTS = 564;
	const bool EXCLUSIVE_SCAN = true;

	// Allocate and initialize host problem data and host reference solution
	T h_src[NUM_ELEMENTS];
	T h_reference[NUM_ELEMENTS];
	Max<T> max_op;
	const bool IS_COMMUTATIVE = true;		// the maximum operator is commutative

	for (size_t i = 0; i < NUM_ELEMENTS; ++i) {
		h_src[i] = i;
		if (EXCLUSIVE_SCAN)
		{
			h_reference[i] = (i == 0) ?
				max_op() :									// identity
				max_op(h_reference[i - 1], h_src[i - 1]);
		} else {
			h_reference[i] = (i == 0) ?
				h_src[i] :
				max_op(h_reference[i - 1], h_src[i]);
		}
	}

	
	// Allocate and initialize device data
	T *d_src, *d_dest;
	cudaMalloc((void**) &d_src, sizeof(T) * NUM_ELEMENTS);
	cudaMalloc((void**) &d_dest, sizeof(T) * NUM_ELEMENTS);
	cudaMemcpy(d_src, h_src, sizeof(T) * NUM_ELEMENTS, cudaMemcpyHostToDevice);


	// Create a scan enactor
	b40c::scan::Enactor scan_enactor;
	scan_enactor.ENACTOR_DEBUG = verbose;

	//
	// Example 1: Enact simple exclusive scan using internal tuning heuristics
	//
	scan_enactor.Scan<EXCLUSIVE_SCAN, IS_COMMUTATIVE>(
		d_dest, d_src, NUM_ELEMENTS, max_op, max_op);
	
	printf("Simple scan: "); b40c::CompareDeviceResults(h_reference, d_dest, NUM_ELEMENTS); printf("\n");
	

	//
	// Example 2: Enact simple exclusive scan using "large problem" tuning configuration
	//
	scan_enactor.Scan<EXCLUSIVE_SCAN, IS_COMMUTATIVE, b40c::scan::LARGE_SIZE>(
		d_dest, d_src, NUM_ELEMENTS, max_op, max_op);

	printf("Large-tuned scan: "); b40c::CompareDeviceResults(h_reference, d_dest, NUM_ELEMENTS); printf("\n");


	//
	// Example 3: Enact simple exclusive scan using "small problem" tuning configuration
	//
	scan_enactor.Scan<EXCLUSIVE_SCAN, IS_COMMUTATIVE, b40c::scan::SMALL_SIZE>(
		d_dest, d_src, NUM_ELEMENTS, max_op, max_op);
	
	printf("Small-tuned scan: "); b40c::CompareDeviceResults(h_reference, d_dest, NUM_ELEMENTS); printf("\n");


	//
	// Example 4: Enact simple exclusive scan using a templated subroutine function
	//
	TemplatedSubroutineScan<EXCLUSIVE_SCAN, IS_COMMUTATIVE>(
		scan_enactor, d_dest, d_src, NUM_ELEMENTS, max_op, max_op);
	
	printf("Templated subroutine scan: "); b40c::CompareDeviceResults(h_reference, d_dest, NUM_ELEMENTS); printf("\n");


	//
	// Example 5: Enact simple exclusive scan using custom tuning configuration (base scan enactor)
	//
	typedef Max<T> ReductionOp;
	typedef Max<T> IdentityOp;
	typedef b40c::scan::ProblemType<T, int, ReductionOp, IdentityOp, EXCLUSIVE_SCAN, IS_COMMUTATIVE> ProblemType;
	typedef b40c::scan::Policy<
		ProblemType,
		b40c::scan::SM20,
		b40c::util::io::ld::cg,
		b40c::util::io::st::cg,
		false, 
		false,
		false,
		8,
		1, 7, 1, 0, 5,
		8, 0, 1, 5,
		1, 7, 1, 0, 5> CustomPolicy;
	
	scan_enactor.Scan<CustomPolicy>(d_dest, d_src, NUM_ELEMENTS, max_op, max_op);

	printf("Custom scan: "); b40c::CompareDeviceResults(h_reference, d_dest, NUM_ELEMENTS); printf("\n");


	//
	// Example 6: Enact simple exclusive scan with misaligned inputs
	//
	scan_enactor.Scan<EXCLUSIVE_SCAN, IS_COMMUTATIVE>(
		d_dest + 1, d_src + 1, NUM_ELEMENTS - 1, max_op, max_op);

	printf("Misaligned scan: "); b40c::CompareDeviceResults(
		h_reference + 1,
		d_dest + 1,
		NUM_ELEMENTS - 1,
		verbose,
		verbose); printf("\n");


	return 0;
}

