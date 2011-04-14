/******************************************************************************
 * 
 * Copyright 2010-2011 Duane Merrill
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
 * Simple test utilities for consecutive removal
 ******************************************************************************/

#pragma once

#include <stdio.h> 

// ConsecutiveRemoval includes
#include <b40c/consecutive_removal_enactor.cuh>

// Test utils
#include "b40c_test_util.h"



/******************************************************************************
 * Utility Routines
 ******************************************************************************/

/**
 * Timed consecutive removal.  Uses the GPU to copy the specified vector of elements for the given
 * number of iterations, displaying runtime information.
 */
template <
	b40c::consecutive_removal::ProbSizeGenre PROB_SIZE_GENRE,
	typename T,
	typename SizeT>
double TimedConsecutiveRemoval(
	T *h_data,
	T *h_reference,
	SizeT num_elements,
	SizeT compacted_elements,
	int max_ctas,
	bool verbose,
	int iterations)
{
	using namespace b40c;

	// Allocate device storage
	T *d_src, *d_dest;
	SizeT *d_num_compacted;
	if (util::B40CPerror(cudaMalloc((void**) &d_src, sizeof(T) * num_elements),
		"TimedConsecutiveRemoval cudaMalloc d_src failed: ", __FILE__, __LINE__)) exit(1);
	if (util::B40CPerror(cudaMalloc((void**) &d_dest, sizeof(T) * num_elements),
		"TimedConsecutiveRemoval cudaMalloc d_dest failed: ", __FILE__, __LINE__)) exit(1);
	if (util::B40CPerror(cudaMalloc((void**) &d_num_compacted, sizeof(SizeT) * 1),
		"TimedConsecutiveRemoval cudaMalloc d_num_compacted failed: ", __FILE__, __LINE__)) exit(1);

	// Create enactor
	ConsecutiveRemovalEnactor consecutive_removal_enactor;

	// Move a fresh copy of the problem into device storage
	if (util::B40CPerror(cudaMemcpy(d_src, h_data, sizeof(T) * num_elements, cudaMemcpyHostToDevice),
		"TimedConsecutiveRemoval cudaMemcpy d_src failed: ", __FILE__, __LINE__)) exit(1);

	// Perform a single iteration to allocate any memory if needed, prime code caches, etc.
	printf("\n");
	consecutive_removal_enactor.DEBUG = true;
	consecutive_removal_enactor.template Enact<PROB_SIZE_GENRE, T>(
		d_dest, d_num_compacted, d_src, num_elements, max_ctas);
	consecutive_removal_enactor.DEBUG = false;

	// Perform the timed number of iterations

	cudaEvent_t start_event, stop_event;
	cudaEventCreate(&start_event);
	cudaEventCreate(&stop_event);

	double elapsed = 0;
	float duration = 0;
	for (int i = 0; i < iterations; i++) {

		// Start timing record
		cudaEventRecord(start_event, 0);

		// Call the consecutive removal API routine
		consecutive_removal_enactor.template Enact<PROB_SIZE_GENRE, T>(
			d_dest, d_num_compacted, d_src, num_elements, max_ctas);

		// End timing record
		cudaEventRecord(stop_event, 0);
		cudaEventSynchronize(stop_event);
		cudaEventElapsedTime(&duration, start_event, stop_event);
		elapsed += (double) duration;
	}

	// Display timing information
	double avg_runtime = elapsed / iterations;
	double throughput = ((double) num_elements) / avg_runtime / 1000.0 / 1000.0;
	double bandwidth = ((double) (num_elements * 2) + compacted_elements) * sizeof(T) / avg_runtime / 1000.0 / 1000.0;
	printf("\nB40C consecutive removal: %d iterations, %lu elements -> %lu compacted, ",
		iterations, (unsigned long) num_elements, (unsigned long) compacted_elements);
    printf("%f GPU ms, %f x10^9 elts/sec, %f x10^9 B/sec, ",
		avg_runtime, throughput, bandwidth);

    // Clean up events
	cudaEventDestroy(start_event);
	cudaEventDestroy(stop_event);

    // Copy out data
	T *h_dest = (T*) malloc(compacted_elements * sizeof(T));
    if (util::B40CPerror(cudaMemcpy(h_dest, d_dest, sizeof(T) * compacted_elements, cudaMemcpyDeviceToHost),
		"TimedConsecutiveRemoval cudaMemcpy d_dest failed: ", __FILE__, __LINE__)) exit(1);

    // Free allocated memory
    if (d_src) cudaFree(d_src);
    if (d_dest) cudaFree(d_dest);
    if (d_num_compacted) cudaFree(d_num_compacted);

	// Flushes any stdio from the GPU
	cudaThreadSynchronize();

	// Display copied data
	if (verbose) {
		printf("\n\nData:\n");
		for (int i = 0; i < compacted_elements; i++) {
			PrintValue<T>(h_dest[i]);
			printf(", ");
		}
		printf("\n\n");
	}

    // Verify solution
	CompareResults(h_dest, h_reference, compacted_elements, true);
	printf("\n");
	fflush(stdout);

	if (h_dest) free(h_dest);

	return throughput;
}


