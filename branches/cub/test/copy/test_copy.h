/******************************************************************************
 * 
 * Copyright 2010-2012 Duane Merrill
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
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
	bool same_device = true)
{
	using namespace b40c;

	// Create enactor
	copy::Enactor copy_enactor;

	// Perform a single iteration to allocate any memory if needed, prime code caches, etc.
	copy_enactor.ENACTOR_DEBUG = true;
	copy_enactor.template Copy<PROB_SIZE_GENRE>(
		d_dest, d_src, num_elements * sizeof(T), max_ctas);
	copy_enactor.ENACTOR_DEBUG = false;

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



