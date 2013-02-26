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
#include <b40c/util/multiple_buffering.cuh>
#include <b40c/util/io/modified_load.cuh>
#include <b40c/util/io/modified_store.cuh>

#include <b40c/radix_sort/problem_type.cuh>
#include <b40c/radix_sort/policy.cuh>
#include <b40c/radix_sort/enactor.cuh>

// Test utils
#include "b40c_test_util.h"


/******************************************************************************
 * Problem / Tuning Policy Types
 ******************************************************************************/

/**
 * Sample sorting problem type (32-bit keys and 32-bit values)
 */
typedef b40c::radix_sort::ProblemType<
		unsigned int,						// Key type
		b40c::util::NullType,				// Value type (keys-only sorting)
		int> 								// SizeT (what type to use for counting)
	ProblemType;


/**
 * Sample radix sort tuning policy
 */
typedef b40c::radix_sort::Policy<
		ProblemType,				// Problem type

		// Common
		200,						// SM ARCH
		4,							// RADIX_BITS

		// Launch tuning policy
		10,							// LOG_SCHEDULE_GRANULARITY			The "grain" by which to divide up the problem input.  E.g., 7 implies a near-even distribution of 128-key chunks to each CTA.  Related to, but different from the upsweep/downswep tile sizes, which may be different from each other.
		b40c::util::io::ld::NONE,	// CACHE_MODIFIER					Load cache-modifier.  Valid values: NONE, ca, cg, cs
		b40c::util::io::st::NONE,	// CACHE_MODIFIER					Store cache-modifier.  Valid values: NONE, wb, cg, cs
		false,						// EARLY_EXIT						Whether or not to early-terminate a sorting pass if we detect all keys have the same digit in that pass's digit place
		false,						// UNIFORM_SMEM_ALLOCATION			Whether or not to pad the dynamic smem allocation to ensure that all three kernels (upsweep, spine, downsweep) have the same overall smem allocation
		true, 						// UNIFORM_GRID_SIZE				Whether or not to launch the spine kernel with one CTA (all that's needed), or pad it up to the same grid size as the upsweep/downsweep kernels
		true,						// OVERSUBSCRIBED_GRID_SIZE			Whether or not to oversubscribe the GPU with CTAs, up to a constant factor (usually 4x the resident occupancy)

		// Policy for upsweep kernel.
		// 		Reduces/counts all the different digit numerals for a given digit-place
		//
		8,							// UPSWEEP_CTA_OCCUPANCY			The targeted SM occupancy to feed PTXAS in order to influence how it does register allocation
		7,							// UPSWEEP_LOG_THREADS				The number of threads (log) to launch per CTA.  Valid range: 5-10
		0,							// UPSWEEP_LOG_LOAD_VEC_SIZE		The vector-load size (log) for each load (log).  Valid range: 0-2
		2,							// UPSWEEP_LOG_LOADS_PER_TILE		The number of loads (log) per tile.  Valid range: 0-2

		// Spine-scan kernel policy
		//		Prefix sum of upsweep histograms counted by each CTA.  Relatively insignificant in the grand scheme, not really worth tuning for large problems)
		//
		7,							// SPINE_LOG_THREADS				The number of threads (log) to launch per CTA.  Valid range: 5-10
		2,							// SPINE_LOG_LOAD_VEC_SIZE			The vector-load size (log) for each load (log).  Valid range: 0-2
		0,							// SPINE_LOG_LOADS_PER_TILE			The number of loads (log) per tile.  Valid range: 0-2
		5,							// SPINE_LOG_RAKING_THREADS			The number of raking threads (log) for local prefix sum.  Valid range: 5-SPINE_LOG_THREADS

		// Policy for downsweep kernel
		//		Given prefix counts, scans/scatters keys into appropriate bins
		// 		Note: a "cycle" is a tile sub-segment up to 256 keys
		//
		b40c::partition::downsweep::SCATTER_TWO_PHASE,						// DOWNSWEEP_TWO_PHASE_SCATTER		Whether or not to perform a two-phase scatter (scatter to smem first to recover some locality before scattering to global bins)
		8,							// DOWNSWEEP_CTA_OCCUPANCY			The targeted SM occupancy to feed PTXAS in order to influence how it does register allocation
		6,							// DOWNSWEEP_LOG_THREADS			The number of threads (log) to launch per CTA.  Valid range: 5-10, subject to constraints described above
		2,							// DOWNSWEEP_LOG_LOAD_VEC_SIZE		The vector-load size (log) for each load (log).  Valid range: 0-2, subject to constraints described above
		1,							// DOWNSWEEP_LOG_LOADS_PER_CYCLE	The number of loads (log) per cycle.  Valid range: 0-2, subject to constraints described above
		1, 							// DOWNSWEEP_LOG_CYCLES_PER_TILE	The number of cycles (log) per tile.  Valid range: 0-2
		6>							// DOWNSWEEP_LOG_RAKING_THREADS		The number of raking threads (log) for local prefix sum.  Valid range: 5-DOWNSWEEP_LOG_THREADS
	Policy;



/******************************************************************************
 * Main
 ******************************************************************************/

int main(int argc, char** argv)
{
    typedef typename ProblemType::OriginalKeyType 	KeyType;
    typedef typename Policy::ValueType 				ValueType;
    typedef typename Policy::SizeT 					SizeT;

    // Initialize command line
    b40c::CommandLineArgs args(argc, argv);
    b40c::DeviceInit(args);

	// Usage/help
    if (args.CheckCmdLineFlag("help") || args.CheckCmdLineFlag("h")) {
    	printf("\ngrid_size [--device=<device index>] [--v] [--n=<elements>] [--i=<samples>]\n");
    	return 0;
    }

    // Parse commandline args
    SizeT num_elements = 1024 * 1024 * 64;			// 64 million pairs
    int samples = 10;								// 1 sample

    bool verbose = args.CheckCmdLineFlag("v");
    args.GetCmdLineArgument("n", num_elements);
    args.GetCmdLineArgument("i", samples);

    // Allocate array of random grid sizes (1 - 65536)
    int *cta_sizes = new int[samples];
	for (int i = 0; i < samples; i++) {
		b40c::util::RandomBits(cta_sizes[i], 0, 16);
		if (cta_sizes[i] == 0) cta_sizes[i] = 1;
	}

	// Allocate and initialize host problem data and host reference solution
	KeyType *h_keys 				= new KeyType[num_elements];

	// Only use RADIX_BITS effective bits (remaining high order bits
	// are left zero): we only want to perform one sorting pass
	for (size_t i = 0; i < num_elements; ++i) {
		b40c::util::RandomBits(h_keys[i], 0, Policy::RADIX_BITS);
//		h_keys[i] = i & ((1 << Policy::RADIX_BITS) - 1);
	}

	// Allocate device data. (We will let the sorting enactor create
	// the "pong" storage if/when necessary.)
	KeyType *d_keys;
	cudaMalloc((void**) &d_keys, sizeof(KeyType) * num_elements);

	// Create ping-pong storage wrapper.
	b40c::util::DoubleBuffer<KeyType> sort_storage(d_keys);
	cudaMemcpy(
		sort_storage.d_keys[sort_storage.selector],
		h_keys,
		sizeof(KeyType) * num_elements,
		cudaMemcpyHostToDevice);

	// Create a scan enactor
	b40c::radix_sort::Enactor enactor;
	enactor.ENACTOR_DEBUG = verbose;

	// Perform the timed number of iterations
	b40c::GpuTimer timer;

	printf("Sample, Items, CTAs, Elapsed, Throughput\n");
	for (int i = 0; i < samples; i++) {

		timer.Start();
		enactor.Sort<
			0,
			Policy::RADIX_BITS,
			Policy>(sort_storage, num_elements, cta_sizes[i]);
		timer.Stop();

		float throughput = float(num_elements) / timer.ElapsedMillis() / 1000.0 / 1000.0;

		printf("%d, %d, %d, %f, %f\n",
			i,
			num_elements,
			cta_sizes[i],
			timer.ElapsedMillis(),
			throughput);
	}

	// Cleanup
	if (sort_storage.d_keys[0]) cudaFree(sort_storage.d_keys[0]);
	if (sort_storage.d_keys[1]) cudaFree(sort_storage.d_keys[1]);
	delete h_keys;
	delete cta_sizes;

	return 0;
}

