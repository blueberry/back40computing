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
 * Simple test utilities for segmented scan
 ******************************************************************************/

#pragma once

#include <stdio.h> 

// SegmentedScan includes
#include <b40c/segmented_scan/enactor.cuh>

// Test utils
#include "b40c_test_util.h"

/******************************************************************************
 * Test wrappers for binary, associative operations
 ******************************************************************************/

template <typename T>
struct Sum
{
	// Binary reduction
	__host__ __device__ __forceinline__ T operator()(const T &a, const T &b)
	{
		return a + b;
	}

	// Identity
	__host__ __device__ __forceinline__ T operator()()
	{
		return 0;
	}
};

template <typename T>
struct Max
{
	// Binary reduction
	__host__ __device__ __forceinline__ T Op(const T &a, const T &b)
	{
		return (a > b) ? a : b;
	}

	// Identity
	__host__ __device__ __forceinline__ T operator()()
	{
		return 0;
	}
};


/******************************************************************************
 * Utility Routines
 ******************************************************************************/

/**
 * Timed segmented scan.  Uses the GPU to copy the specified vector of elements for the given
 * number of iterations, displaying runtime information.
 */
template <
	bool EXCLUSIVE,
	b40c::segmented_scan::ProbSizeGenre PROB_SIZE_GENRE,
	typename T,
	typename Flag,
	typename SizeT,
	typename ReductionOp,
	typename IdentityOp>
double TimedSegmentedScan(
	T *h_data,
	Flag *h_flag_data,
	T *h_reference,
	SizeT num_elements,
	ReductionOp scan_op,
	IdentityOp identity_op,
	int max_ctas,
	bool verbose,
	int iterations)
{
	using namespace b40c;

	// Allocate device storage
	T *d_src, *d_dest;
	Flag *d_flag_src;
	if (util::B40CPerror(cudaMalloc((void**) &d_src, sizeof(T) * num_elements),
		"TimedSegmentedScan cudaMalloc d_src failed: ", __FILE__, __LINE__)) exit(1);
	if (util::B40CPerror(cudaMalloc((void**) &d_dest, sizeof(T) * num_elements),
		"TimedSegmentedScan cudaMalloc d_dest failed: ", __FILE__, __LINE__)) exit(1);
	if (util::B40CPerror(cudaMalloc((void**) &d_flag_src, sizeof(Flag) * num_elements),
		"TimedSegmentedScan cudaMalloc d_flag_src failed: ", __FILE__, __LINE__)) exit(1);

	// Create enactor
	segmented_scan::Enactor enactor;

	// Move a fresh copy of the problem into device storage
	if (util::B40CPerror(cudaMemcpy(d_src, h_data, sizeof(T) * num_elements, cudaMemcpyHostToDevice),
		"TimedSegmentedScan cudaMemcpy d_src failed: ", __FILE__, __LINE__)) exit(1);
	if (util::B40CPerror(cudaMemcpy(d_flag_src, h_flag_data, sizeof(Flag) * num_elements, cudaMemcpyHostToDevice),
		"TimedSegmentedScan cudaMemcpy d_flag_src failed: ", __FILE__, __LINE__)) exit(1);

	// Marker kernel in profiling stream
	util::FlushKernel<void><<<1,1>>>();

	// Perform a single iteration to allocate any memory if needed, prime code caches, etc.
	printf("\n");
	enactor.ENACTOR_DEBUG = true;
	enactor.template Scan<PROB_SIZE_GENRE, EXCLUSIVE>(
		d_dest,
		d_src,
		d_flag_src,
		num_elements,
		scan_op,
		identity_op,
		max_ctas);
	enactor.ENACTOR_DEBUG = false;

	// Perform the timed number of iterations
	GpuTimer timer;

	double elapsed = 0;
	for (int i = 0; i < iterations; i++) {

		// Marker kernel in profiling stream
		util::FlushKernel<void><<<1,1>>>();

		// Start timing record
		timer.Start();

		// Call the segmented scan API routine
		enactor.template Scan<PROB_SIZE_GENRE, EXCLUSIVE>(
			d_dest, d_src, d_flag_src, num_elements, scan_op, identity_op, max_ctas);

		// End timing record
		timer.Stop();
		elapsed += (double) timer.ElapsedMillis();
	}

	// Display timing information
	double avg_runtime = elapsed / iterations;
	double throughput = ((double) num_elements) / avg_runtime / 1000.0 / 1000.0;
	printf("\nB40C %s segmented scan: %d iterations, %lu elements, ",
		EXCLUSIVE ? "exclusive" : "inclusive", iterations, (unsigned long) num_elements);
    printf("%f GPU ms, %f x10^9 elts/sec, %f x10^9 B/sec, ",
		avg_runtime, throughput, throughput * ((sizeof(T) * 3) + (sizeof(Flag) * 2)));

	// Flushes any stdio from the GPU
	cudaThreadSynchronize();

    // Verify solution
	CompareDeviceResults(h_reference, d_dest, num_elements, verbose, verbose);
	printf("\n");
	fflush(stdout);

    // Free allocated memory
    if (d_src) cudaFree(d_src);
    if (d_dest) cudaFree(d_dest);
    if (d_flag_src) cudaFree(d_flag_src);

	return throughput;
}


