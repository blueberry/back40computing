/******************************************************************************
 *
 * Copyright 2010-2011 Duane Merrill
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a scan of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. 
 * 
 * For more information, see our Google Code project site: 
 * http://code.google.com/p/back40computing/
 * 
 ******************************************************************************/


/******************************************************************************
 * Simple test driver program for radix sort.
 ******************************************************************************/

#include <stdio.h> 
#include <algorithm>

// Enable printing of cuda errors to stderr
#define CUB_STDERR 1

// Sorting includes
#include <cub/cub.cuh>
//#include <back40/radix_sort/enactor.cuh>

// Test utils
#include "test_util.h"



/******************************************************************************
 * Simple utilities for working with cub::NullType values
 ******************************************************************************/

template <typename T, typename S>
void Cast(T &t, const S &s)
{
	t = (T) s;
}

template <typename T>
void Cast(T &t, const cub::NullType &s)
{
}

template <typename S>
void Cast(cub::NullType &t, const S &s)
{
}


/******************************************************************************
 * Main
 ******************************************************************************/

int main(int argc, char** argv)
{
//	typedef unsigned long long		KeyType;
	typedef unsigned int 			KeyType;
	typedef cub::NullType 			ValueType;
//	typedef unsigned long long 		ValueType;
//	typedef unsigned int			ValueType;

	const int 		START_BIT			= 0;
	const int 		KEY_BITS 			= sizeof(KeyType) * 8;
	const bool 		KEYS_ONLY			= cub::Equals<ValueType, cub::NullType>::VALUE;
    int 			num_elements 		= 1024 * 1024 * 8;			// 8 million pairs
    unsigned int 	max_ctas 			= 0;						// default: let the enactor decide how many CTAs to launch based upon device properties
    int 			iterations 			= 0;
    int				entropy_reduction 	= 0;
    int 			effective_bits 		= KEY_BITS;

    // Initialize command line
    back40::CommandLineArgs args(argc, argv);
    back40::DeviceInit(args);

	// Usage/help
    if (args.CheckCmdLineFlag("help") || args.CheckCmdLineFlag("h"))
    {
    	printf("\nlars_demo [--device=<device index>] [--v] [--n=<elements>] "
    			"[--max-ctas=<max-thread-blocks>] [--i=<iterations>] "
    			"[--zeros | --regular] [--entropy-reduction=<random &'ing rounds>\n");
    	return 0;
    }

    // Parse commandline args
    bool verbose = args.CheckCmdLineFlag("v");
    bool zeros = args.CheckCmdLineFlag("zeros");
    bool regular = args.CheckCmdLineFlag("regular");
    bool schmoo = args.CheckCmdLineFlag("schmoo");
    args.GetCmdLineArgument("n", num_elements);
    args.GetCmdLineArgument("i", iterations);
    args.GetCmdLineArgument("max-ctas", max_ctas);
    args.GetCmdLineArgument("entropy-reduction", entropy_reduction);
    args.GetCmdLineArgument("bits", effective_bits);

    // Print header
    if (zeros) {
    	printf("Zeros\n");
    } else if (regular) {
    	printf("%d-bit mod-%llu\n", KEY_BITS, 1ull << effective_bits);
    } else {
    	printf("%d-bit random\n", KEY_BITS);
    }
    fflush(stdout);

	// Allocate host problem data
    KeyType 	*h_keys 				= new KeyType[num_elements];
	KeyType 	*h_reference_keys 		= new KeyType[num_elements];
    ValueType 	*h_values 				= new ValueType[num_elements];

    // Initialize host problem data
	if (verbose) printf("Original: ");
	for (size_t i = 0; i < num_elements; ++i)
	{
		if (regular) {
			h_keys[i] = i & ((1ull << effective_bits) - 1);
		} else if (zeros) {
			h_keys[i] = 0;
		} else {
			back40::RandomBits(h_keys[i], entropy_reduction, KEY_BITS);
		}
		h_keys[i] <<= START_BIT;

		h_reference_keys[i] = h_keys[i];
		Cast(h_values[i], i);

		if (verbose)
		{
			printf("%d, ", h_keys[i]);
			if ((i & 255) == 255) printf("\n\n");
		}
	}
	if (verbose) printf("\n");

    // Compute reference solution
	std::sort(h_reference_keys, h_reference_keys + num_elements);

	// Allocate device data
	cub::DoubleBuffer<KeyType, ValueType> double_buffer;
	cudaMalloc((void**) &double_buffer.d_keys[0], sizeof(KeyType) * num_elements);
	cudaMalloc((void**) &double_buffer.d_keys[1], sizeof(KeyType) * num_elements);
	if (!KEYS_ONLY)
	{
		cudaMalloc((void**) &double_buffer.d_values[0], sizeof(ValueType) * num_elements);
		cudaMalloc((void**) &double_buffer.d_values[1], sizeof(ValueType) * num_elements);
	}
/*
	// Create sorting enactor
	back40::radix_sort::Enactor enactor;
*/
	// Perform one sorting pass (starting at bit zero and covering RADIX_BITS bits)
	cudaMemcpy(
		double_buffer.d_keys[double_buffer.selector],
		h_keys,
		sizeof(KeyType) * num_elements,
		cudaMemcpyHostToDevice);
	if (!KEYS_ONLY)
	{
		cudaMemcpy(
			double_buffer.d_values[double_buffer.selector],
			h_values,
			sizeof(ValueType) * num_elements,
			cudaMemcpyHostToDevice);
	}

	printf("\nRestricted-range %s sort (selector %d): ",
		(KEYS_ONLY) ? "keys-only" : "key-value",
		double_buffer.selector);
	fflush(stdout);
/*
	// Sort
	enactor.Sort<back40::radix_sort::LARGE_PROBLEM, KEY_BITS, START_BIT>(
		double_buffer, num_elements, 0, max_ctas, true);
*/
	// Force any stdout from kernels
	cudaThreadSynchronize();

	// Check answer
	back40::CompareDeviceResults(
		h_reference_keys,
		double_buffer.d_keys[double_buffer.selector],
		num_elements,
		true,
		verbose); printf("\n");
	if (!KEYS_ONLY)
	{
		// Copy out values
		cudaMemcpy(
			h_values,
			double_buffer.d_values[double_buffer.selector],
			sizeof(ValueType) * num_elements,
			cudaMemcpyDeviceToHost);

		// Check that values correspond to the sorting permutation
		bool correct = true;
		for (size_t i = 0; i < num_elements; ++i)
		{
			int permute_index;
			Cast(permute_index, h_values[i]);
			if (h_keys[permute_index] != h_reference_keys[i])
			{
				printf("Incorrect: [%d]: %d != %d\n",
					i,
					h_keys[permute_index],
					h_reference_keys[i]);
				correct = false;
				break;
			}
		}
		if (correct) printf("Correct\n\n");
	}

	// Evaluate performance iterations
	if (schmoo)
	{
		printf("iteration, elements, elapsed (ms), throughput (MKeys/s)\n");
	}

	back40::GpuTimer gpu_timer;
	double max_exponent 		= log2(double(num_elements)) - 5.0;
	unsigned int max_int 		= (unsigned int) -1;
	float elapsed 				= 0;

	for (int i = 0; i < iterations; i++)
	{
		// Reset problem
		double_buffer.selector = 0;
		cudaMemcpy(
			double_buffer.d_keys[double_buffer.selector],
			h_keys,
			sizeof(KeyType) * num_elements,
			cudaMemcpyHostToDevice);
		if (!KEYS_ONLY)
		{
			cudaMemcpy(
				double_buffer.d_values[double_buffer.selector],
				h_values,
				sizeof(ValueType) * num_elements,
				cudaMemcpyHostToDevice);
		}

		if (schmoo)
		{
			// Sample a problem size in the range [1,num_elements]
			unsigned int sample;
			back40::RandomBits(sample);
			double scale = double(sample) / max_int;
			int elements = (i < iterations / 2) ?
				pow(2.0, (max_exponent * scale) + 5.0) :		// log bias
				elements = scale * num_elements;						// uniform bias

			gpu_timer.Start();
/*
			// Sort
			enactor.Sort<back40::radix_sort::LARGE_PROBLEM, KEY_BITS, START_BIT>(
				double_buffer, num_elements, 0, max_ctas);
*/
			gpu_timer.Stop();

			float millis = gpu_timer.ElapsedMillis();
			printf("%d, %d, %.3f, %.2f\n",
				i,
				elements,
				millis,
				float(elements) / millis / 1000.f);
			fflush(stdout);
		}
		else
		{
			// Regular iteration
			gpu_timer.Start();
/*
			// Sort
			enactor.Sort<back40::radix_sort::LARGE_PROBLEM, KEY_BITS, START_BIT>(
				double_buffer, num_elements, 0, max_ctas);
*/
			gpu_timer.Stop();

			elapsed += gpu_timer.ElapsedMillis();
		}
	}

	// Display output
	if ((!schmoo) && (iterations > 0))
	{
		float avg_elapsed = elapsed / float(iterations);
		printf("Elapsed millis: %f, avg elapsed: %f, throughput: %.2f Mkeys/s\n",
			elapsed,
			avg_elapsed,
			float(num_elements) / avg_elapsed / 1000.f);
	}

	// Cleanup device storage
	if (double_buffer.d_keys[0]) cudaFree(double_buffer.d_keys[0]);
	if (double_buffer.d_keys[1]) cudaFree(double_buffer.d_keys[1]);
	if (double_buffer.d_values[0]) cudaFree(double_buffer.d_values[0]);
	if (double_buffer.d_values[1]) cudaFree(double_buffer.d_values[1]);

	// Cleanup other
	delete h_keys;
	delete h_reference_keys;
	delete h_values;

	return 0;
}

