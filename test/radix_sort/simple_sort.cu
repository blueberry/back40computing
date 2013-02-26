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
 * Simple test driver program for radix sort.
 ******************************************************************************/

#include <stdio.h> 
#include <algorithm>

// Sorting includes
#include <b40c/radix_sort/enactor.cuh>
#include <b40c/util/multiple_buffering.cuh>

// Test utils
#include "b40c_test_util.h"

/******************************************************************************
 * Constants
 ******************************************************************************/

const int LOWER_BITS = 17;


/******************************************************************************
 * Main
 ******************************************************************************/

int main(int argc, char** argv)
{
	b40c::CommandLineArgs args(argc, argv);

	// Usage/help
    if (args.CheckCmdLineFlag("help") || args.CheckCmdLineFlag("h")) {
    	printf("\nsimple_sort [--device=<device index>] [--v] [--n=<elements>] [--keys-only]\n");
    	return 0;
    }

    b40c::DeviceInit(args);
    unsigned int num_elements = 77;
    bool verbose = args.CheckCmdLineFlag("v");
    bool keys_only = args.CheckCmdLineFlag("keys-only");
    args.GetCmdLineArgument("n", num_elements);

	// Allocate and initialize host problem data and host reference solution
	int *h_keys = new int[num_elements];
	int *h_values = new int[num_elements];
	int *h_reference_keys = new int[num_elements];
	int *h_reference_values = new int[num_elements];

	for (size_t i = 0; i < num_elements; ++i) {
		b40c::util::RandomBits(h_keys[i], 0, LOWER_BITS);
		h_values[i] = i;
		h_reference_keys[i] = h_keys[i];
	}

    // Compute reference solution
	std::sort(h_reference_keys, h_reference_keys + num_elements);

	// Allocate device data. (We will let the sorting enactor create
	// the "pong" storage if/when necessary.)
	int *d_keys;
	int *d_values;
	cudaMalloc((void**) &d_keys, sizeof(int) * num_elements);
	cudaMalloc((void**) &d_values, sizeof(int) * num_elements);

	// Create a scan enactor
	b40c::radix_sort::Enactor enactor;

	if (keys_only) {

		//
		// Keys-only sorting
		//

		// Create ping-pong storage wrapper.
		b40c::util::DoubleBuffer<int> sort_storage(d_keys);

		//
		// Example 1: simple sort.  Uses heuristics to select
		// appropriate problem-size-tuning granularity. (Using this
		// method causes the compiler to generate several tuning variants,
		// which can increase compilation times)
		//

		cudaMemcpy(sort_storage.d_keys[sort_storage.selector], h_keys, sizeof(int) * num_elements, cudaMemcpyHostToDevice);
		enactor.Sort(sort_storage, num_elements);

		printf("Simple keys-only sort: "); b40c::CompareDeviceResults(
			h_reference_keys, sort_storage.d_keys[sort_storage.selector], num_elements, verbose, verbose); printf("\n");

		//
		// Example 2: Small-problem-tuned sort.  Tuned for < 1M elements
		//

		cudaMemcpy(sort_storage.d_keys[sort_storage.selector], h_keys, sizeof(int) * num_elements, cudaMemcpyHostToDevice);
		enactor.Sort<b40c::radix_sort::SMALL_SIZE>(sort_storage, num_elements);

		printf("Small-problem keys-only sort: "); b40c::CompareDeviceResults(
			h_reference_keys, sort_storage.d_keys[sort_storage.selector], num_elements, verbose, verbose); printf("\n");

		//
		// Example 3: small-problem-tuned sort over specific bit-range
		//

		cudaMemcpy(sort_storage.d_keys[sort_storage.selector], h_keys, sizeof(int) * num_elements, cudaMemcpyHostToDevice);
		enactor.Sort<0, LOWER_BITS, b40c::radix_sort::SMALL_SIZE>(sort_storage, num_elements);

		printf("Small-problem restricted-range keys-only sort: "); b40c::CompareDeviceResults(
			h_reference_keys, sort_storage.d_keys[sort_storage.selector], num_elements, verbose, verbose); printf("\n");

		// Cleanup any "pong" storage allocated by the enactor
		if (sort_storage.d_keys[1]) cudaFree(sort_storage.d_keys[1]);

	} else {

		//
		// Key-value sorting
		//

		// Create ping-pong storage wrapper.
		b40c::util::DoubleBuffer<int, int> sort_storage(d_keys, d_values);

		//
		// Example 1: simple sort.  Uses heuristics to select
		// appropriate problem-size-tuning granularity. (Using this
		// method causes the compiler to generate several tuning variants,
		// which can increase compilation times)
		//

		cudaMemcpy(sort_storage.d_keys[sort_storage.selector], h_keys, sizeof(int) * num_elements, cudaMemcpyHostToDevice);
		cudaMemcpy(sort_storage.d_values[sort_storage.selector], h_values, sizeof(int) * num_elements, cudaMemcpyHostToDevice);
		enactor.Sort(sort_storage, num_elements);

		printf("Simple key-value sort: "); b40c::CompareDeviceResults(
			h_reference_keys, sort_storage.d_keys[sort_storage.selector], num_elements, verbose, verbose); printf("\n");

		//
		// Example 2: Small-problem-tuned sort.  Tuned for < 1M elements
		//

		cudaMemcpy(sort_storage.d_keys[sort_storage.selector], h_keys, sizeof(int) * num_elements, cudaMemcpyHostToDevice);
		cudaMemcpy(sort_storage.d_values[sort_storage.selector], h_values, sizeof(int) * num_elements, cudaMemcpyHostToDevice);
		enactor.Sort<b40c::radix_sort::SMALL_SIZE>(sort_storage, num_elements);

		printf("Small-problem key-value sort: "); b40c::CompareDeviceResults(
			h_reference_keys, sort_storage.d_keys[sort_storage.selector], num_elements, verbose, verbose); printf("\n");


		//
		// Example 3: small-problem-tuned sort over specific bit-range
		//

		cudaMemcpy(sort_storage.d_keys[sort_storage.selector], h_keys, sizeof(int) * num_elements, cudaMemcpyHostToDevice);
		cudaMemcpy(sort_storage.d_values[sort_storage.selector], h_values, sizeof(int) * num_elements, cudaMemcpyHostToDevice);

		enactor.Sort<0, LOWER_BITS, b40c::radix_sort::SMALL_SIZE>(sort_storage, num_elements);

		printf("Small-problem restricted-range key-value sort: "); b40c::CompareDeviceResults(
			h_reference_keys, sort_storage.d_keys[sort_storage.selector], num_elements, verbose, verbose); printf("\n");


		// Cleanup any "pong" storage allocated by the enactor
		if (sort_storage.d_keys[1]) cudaFree(sort_storage.d_keys[1]);
		if (sort_storage.d_values[1]) cudaFree(sort_storage.d_values[1]);

	}
	
	delete h_keys;
	delete h_reference_keys;
	delete h_values;
	delete h_reference_values;

	return 0;
}

