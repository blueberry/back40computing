/******************************************************************************
 * 
 * Copyright 2010 Duane Merrill
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
 * Thanks!
 * 
 ******************************************************************************/

/******************************************************************************
 * Work Management Datastructures
 ******************************************************************************/

#pragma once

namespace b40c {
namespace util {


/**
 * Description of work distribution amongst CTAs
 *
 * A given threadblock may receive one of three different amounts of 
 * work: "big", "normal", and "last".  The big workloads are one
 * grain greater than the normal, and the last workload 
 * does the extra work.
 */
template <typename SizeT> 		// Integer type for indexing into problem arrays (e.g., int, long long, etc.)
struct CtaWorkDistribution
{
	SizeT num_elements;		// Number of elements in the problem
	SizeT total_grains;		// Number of "grain" blocks to break the problem into (round up)
	SizeT grains_per_cta;	// Number of "grain" blocks per CTA
	SizeT extra_grains;		// Number of CTAs having one extra "grain block"
	int grid_size;			// Number of CTAs

	/**
	 * Constructor
	 */
	CtaWorkDistribution(
		SizeT num_elements,
		int schedule_granularity, 	// Problem granularity by which work is distributed amongst CTA threadblocks
		int grid_size) :
			num_elements(num_elements),
			total_grains((num_elements + schedule_granularity - 1) / schedule_granularity),		// round up
			grains_per_cta((grid_size > 0) ? total_grains / grid_size : 0),						// round down for the ks
			extra_grains(total_grains - (grains_per_cta * grid_size)), 							// the CTAs with +1 grains
			grid_size(grid_size)
	{}


	/**
	 * Computes work limits for the current CTA
	 */	
	template <
		int LOG_TILE_ELEMENTS,			// CTA tile size, i.e., granularity by which the CTA processes work
		int LOG_SCHEDULE_GRANULARITY>	// Problem granularity by which work is distributed amongst CTA threadblocks
	__device__ __forceinline__ void GetCtaWorkLimits(
		SizeT &cta_offset,				// Out param: Offset at which this CTA begins processing
		SizeT &cta_elements,			// Out param: Total number of elements for this CTA to process
		SizeT &guarded_offset, 			// Out param: Offset of final, partially-full tile (requires guarded loads)
		SizeT &cta_guarded_elements)	// Out param: Number of elements in partially-full tile
	{
		const int TILE_ELEMENTS 				= 1 << LOG_TILE_ELEMENTS;
		const int SCHEDULE_GRANULARITY 			= 1 << LOG_SCHEDULE_GRANULARITY;
		
		// Compute number of elements and offset at which to start tile processing
		if (blockIdx.x < extra_grains) {

			// This CTA gets grains_per_cta+1 grains
			cta_elements = (grains_per_cta + 1) << LOG_SCHEDULE_GRANULARITY;
			cta_offset = cta_elements * blockIdx.x;

		} else if (blockIdx.x < total_grains) {

			// This CTA gets grains_per_cta grains
			cta_elements = grains_per_cta << LOG_SCHEDULE_GRANULARITY;
			cta_offset = (cta_elements * blockIdx.x) + (extra_grains << LOG_SCHEDULE_GRANULARITY);

		} else {

			// This CTA gets no work (problem small enough that some CTAs don't even a single grain)
			cta_elements = 0;
			cta_offset = 0;
		}
		
		// The last CTA having work will have rounded its last grain up past the end
		if (cta_offset + cta_elements > num_elements) {
			cta_elements = cta_elements - SCHEDULE_GRANULARITY + 			// subtract grain size
				(num_elements & (SCHEDULE_GRANULARITY - 1));				// add delta to end of input
		}

		// The tile-aligned limit for full-tile processing
		cta_guarded_elements = cta_elements & (TILE_ELEMENTS - 1);

		// The number of extra guarded-load elements to process afterward (always
		// less than a full tile)
		guarded_offset = cta_offset + cta_elements - cta_guarded_elements;
	}


	/**
	 * Print to stdout
	 */
	__host__ __device__ __forceinline__ void Print()
	{
		printf("num_elements: %lu, total_grains: %lu, grains_per_cta: %lu, extra_grains: %lu, grid_size: %lu\n",
			(unsigned long) num_elements,
			(unsigned long) total_grains,
			(unsigned long) grains_per_cta,
			(unsigned long) extra_grains,
			(unsigned long) grid_size);

	}
};


} // namespace util
} // namespace b40c

