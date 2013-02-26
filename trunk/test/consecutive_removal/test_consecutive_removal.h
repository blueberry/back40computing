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
 * Simple test utilities for consecutive removal
 ******************************************************************************/

#pragma once

#include <stdio.h> 

// ConsecutiveRemoval includes
#include <b40c/consecutive_removal/enactor.cuh>

// Test utils
#include "b40c_test_util.h"


/******************************************************************************
 * Test wrappers for binary, associative operations
 ******************************************************************************/

template <typename T>
struct Equality
{
	__host__ __device__ __forceinline__ bool operator()(const T &a, const T &b)
	{
		return a == b;
	}
};


/******************************************************************************
 * Utility Routines
 ******************************************************************************/

/**
 * Timed consecutive removal.  Uses the GPU to copy the specified vector of elements for the given
 * number of iterations, displaying runtime information.
 */
template <
	b40c::consecutive_removal::ProbSizeGenre PROB_SIZE_GENRE,
	typename DoubleBuffer,
	typename SizeT,
	typename EqualityOp>
double TimedConsecutiveRemoval(
	DoubleBuffer &h_problem_storage,			// host problem storage (selector points to input, but output contains reference result)
	SizeT num_elements,
	SizeT num_compacted,						// number of elements in reference result
	EqualityOp equality_op,
	int max_ctas,
	bool verbose,
	int iterations)
{
	using namespace b40c;

	typedef typename DoubleBuffer::KeyType 		KeyType;
	typedef typename DoubleBuffer::ValueType 	ValueType;

	const bool KEYS_ONLY = util::Equals<ValueType, util::NullType>::VALUE;

	// Allocate device storage
	DoubleBuffer 	d_problem_storage;
	SizeT				*d_num_compacted;

	if (util::B40CPerror(cudaMalloc((void**) &d_problem_storage.d_keys[0], sizeof(KeyType) * num_elements),
		"TimedConsecutiveReduction cudaMalloc d_keys failed: ", __FILE__, __LINE__)) exit(1);
	if (util::B40CPerror(cudaMalloc((void**) &d_problem_storage.d_keys[1], sizeof(KeyType) * num_elements),
		"TimedConsecutiveReduction cudaMalloc d_keys failed: ", __FILE__, __LINE__)) exit(1);
	if (!KEYS_ONLY) {
		if (util::B40CPerror(cudaMalloc((void**) &d_problem_storage.d_values[0], sizeof(ValueType) * num_elements),
			"TimedConsecutiveReduction cudaMalloc d_values failed: ", __FILE__, __LINE__)) exit(1);
		if (util::B40CPerror(cudaMalloc((void**) &d_problem_storage.d_values[1], sizeof(ValueType) * num_elements),
			"TimedConsecutiveReduction cudaMalloc d_values failed: ", __FILE__, __LINE__)) exit(1);
	}

	if (util::B40CPerror(cudaMalloc((void**) &d_num_compacted, sizeof(SizeT) * 1),
		"TimedConsecutiveReduction cudaMalloc d_num_compacted failed: ", __FILE__, __LINE__)) exit(1);

	// Create enactor
	consecutive_removal::Enactor enactor;
	SizeT h_num_compacted;

	// Move a fresh copy of the problem into device storage
	if (util::B40CPerror(cudaMemcpy(
			d_problem_storage.d_keys[0],
			h_problem_storage.d_keys[0],
			sizeof(KeyType) * num_elements,
			cudaMemcpyHostToDevice),
		"TimedConsecutiveReduction cudaMemcpy d_keys failed: ", __FILE__, __LINE__)) exit(1);
	if (!KEYS_ONLY) {
		if (util::B40CPerror(cudaMemcpy(
				d_problem_storage.d_values[0],
				h_problem_storage.d_values[0],
				sizeof(ValueType) * num_elements,
				cudaMemcpyHostToDevice),
			"TimedConsecutiveReduction cudaMemcpy d_values failed: ", __FILE__, __LINE__)) exit(1);
	}

	// Marker kernel in profiling stream
	util::FlushKernel<void><<<1,1>>>();

	// Perform a single iteration to allocate any memory if needed, prime code caches, etc.
	printf("\n");
	enactor.ENACTOR_DEBUG = true;
	enactor.template Trim<PROB_SIZE_GENRE>(
		d_problem_storage, num_elements, &h_num_compacted, d_num_compacted, equality_op, max_ctas);
	enactor.ENACTOR_DEBUG = false;

	// Perform the timed number of iterations
	GpuTimer timer;

	double elapsed = 0;
	for (int i = 0; i < iterations; i++) {

		// Marker kernel in profiling stream
		util::FlushKernel<void><<<1,1>>>();

		// Start timing record
		timer.Start();

		// Call the consecutive removal API routine
		enactor.template Trim<PROB_SIZE_GENRE>(
			d_problem_storage, num_elements, (SizeT *) NULL, d_num_compacted, equality_op, max_ctas);

		// End timing record
		timer.Stop();
		elapsed += (double) timer.ElapsedMillis();
	}

	// Display timing information
	double avg_runtime = elapsed / iterations;
	double throughput = ((double) num_elements) / avg_runtime / 1000.0 / 1000.0;
	long long bytes = (KEYS_ONLY) ?
		((num_elements * 2) + num_compacted) * sizeof(KeyType) :
		(((num_elements * 2) + num_compacted) * sizeof(KeyType)) +
			((num_elements + num_compacted) * sizeof(ValueType));
	double bandwidth = bytes / avg_runtime / 1000.0 / 1000.0;

	printf("\nB40C %s consecutive removal: %d iterations, %lu elements -> %lu compacted, ",
		(KEYS_ONLY) ? "keys-only" : "key-value", iterations, (unsigned long) num_elements, (unsigned long) num_compacted);
    printf("%f GPU ms, %f x10^9 elts/sec, %f x10^9 B/sec, ",
		avg_runtime, throughput, bandwidth);

	// Check and display results
	printf("\nCompacted keys: ");
	CompareDeviceResults(h_problem_storage.d_keys[1], d_problem_storage.d_keys[1], num_compacted, verbose, verbose);
	if (!KEYS_ONLY) {
		printf("\nCompacted values: ");
		CompareDeviceResults(h_problem_storage.d_values[1], d_problem_storage.d_values[1], num_compacted, verbose, verbose);
	}
	printf("\nCompacted size: ");
	CompareDeviceResults(&num_compacted, d_num_compacted, 1, verbose, verbose);
	printf("\nCompacted size reported to host: %s\n", (num_compacted == h_num_compacted) ? "CORRECT" : "INCORRECT");
	printf("\n");
	fflush(stdout);

	// Free allocated memory
    if (d_problem_storage.d_keys[0]) cudaFree(d_problem_storage.d_keys[0]);
    if (d_problem_storage.d_keys[1]) cudaFree(d_problem_storage.d_keys[1]);
    if (d_problem_storage.d_values[0]) cudaFree(d_problem_storage.d_values[0]);
    if (d_problem_storage.d_values[1]) cudaFree(d_problem_storage.d_values[1]);
    if (d_num_compacted) cudaFree(d_num_compacted);

	// Flushes any stdio from the GPU
	cudaThreadSynchronize();

	return throughput;
}


