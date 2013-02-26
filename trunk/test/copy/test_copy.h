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
 * Simple test utilities for copy
 ******************************************************************************/

#pragma once

#include <stdio.h> 

// Copy includes
#include <b40c/copy/enactor.cuh>

// Test utils
#include "b40c_test_util.h"

/******************************************************************************
 * Utility Routines
 ******************************************************************************/

/**
 * Timed copy.
 */
template <
	b40c::copy::ProbSizeGenre PROB_SIZE_GENRE,
	typename T,
	typename SizeT>
double TimedCopy(
	T *h_reference,
	T *d_src,
	T *d_dest,
	SizeT num_elements,
	int max_ctas,
	bool verbose,
	int iterations,
	bool same_device = true,
	bool warmup = true)
{
	using namespace b40c;

	// Create enactor
	copy::Enactor copy_enactor;

	// Perform a single iteration to allocate any memory if needed, prime code caches, etc.
	if (warmup) {
		copy_enactor.ENACTOR_DEBUG = true;
		copy_enactor.template Copy<PROB_SIZE_GENRE>(
			d_dest, d_src, num_elements * sizeof(T), max_ctas);
		copy_enactor.ENACTOR_DEBUG = false;
	}

	// Perform the timed number of iterations
	GpuTimer timer;

	double elapsed = 0;
	for (int i = 0; i < iterations; i++) {

		// Start timing record
		timer.Start();

		// Call the copy API routine
		copy_enactor.template Copy<PROB_SIZE_GENRE>(
			d_dest, d_src, num_elements * sizeof(T), max_ctas);

		// End timing record
		timer.Stop();
		elapsed += (double) timer.ElapsedMillis();
	}

	// Display timing information
	double avg_runtime = elapsed / iterations;
	double throughput = ((double) num_elements) / avg_runtime / 1000.0 / 1000.0;
	int bytes_per_element = (same_device) ? sizeof(T) * 2 : sizeof(T);

	printf("%lu, %lu, %d, %.3f, %.2f, ",
		(unsigned long) num_elements,
		(unsigned long) num_elements * sizeof(T),
		iterations,
		avg_runtime,
		throughput * bytes_per_element);

	// Flushes any stdio from the GPU
	cudaThreadSynchronize();

	// Check for correctness
	b40c::CompareDeviceResults(
		h_reference,
		d_dest,
		num_elements,
		true,
		verbose);
	printf("\n");
	fflush(stdout);

	return throughput;
}



