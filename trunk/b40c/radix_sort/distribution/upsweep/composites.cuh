
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
 * AUTHORS' REQUEST: 
 * 
 * 		If you use|reference|benchmark this code, please cite our Technical 
 * 		Report (http://www.cs.virginia.edu/~dgm4d/papers/RadixSortTR.pdf):
 * 
 *		@TechReport{ Merrill:Sorting:2010,
 *        	author = "Duane Merrill and Andrew Grimshaw",
 *        	title = "Revisiting Sorting for GPGPU Stream Architectures",
 *        	year = "2010",
 *        	institution = "University of Virginia, Department of Computer Science",
 *        	address = "Charlottesville, VA, USA",
 *        	number = "CS2010-03"
 *		}
 * 
 * For more information, see our Google Code project site: 
 * http://code.google.com/p/back40computing/
 * 
 * Thanks!
 * 
 ******************************************************************************/


/******************************************************************************
 * Upsweep composite-grid processing abstraction
 ******************************************************************************/

#pragma once

#include <b40c/util/cuda_properties.cuh>
#include <b40c/util/basic_utils.cuh>
#include <b40c/util/cta_work_distribution.cuh>
#include <b40c/util/io/modified_load.cuh>
#include <b40c/util/io/modified_store.cuh>
#include <b40c/util/io/load_tile.cuh>
#include <b40c/util/device_intrinsics.cuh>
#include <b40c/util/reduction/serial_reduce.cuh>

#include <b40c/radix_sort/distribution/upsweep/lanes.cuh>
#include <b40c/radix_sort/sort_utils.cuh>

namespace b40c {
namespace radix_sort {
namespace distribution {
namespace upsweep {


/**
 * Composites
 */
template <typename KernelConfig>
struct Composites
{
	enum {
		LANES_PER_WARP 						= KernelConfig::LANES_PER_WARP,
		COMPOSITES_PER_LANE_PER_THREAD 		= KernelConfig::COMPOSITES_PER_LANE_PER_THREAD,
		WARPS								= KernelConfig::WARPS,
		COMPOSITE_LANES						= KernelConfig::COMPOSITE_LANES,
	};


	//---------------------------------------------------------------------
	// Iteration Structures
	//---------------------------------------------------------------------

	/**
	 * Iterate next composite
	 */
	template <int WARP_LANE, int THREAD_COMPOSITE, int dummy = 0>
	struct Iterate
	{
		// ReduceComposites
		template <typename Cta>
		static __device__ __forceinline__ void ReduceComposites(Cta *cta)
		{
			int lane				= (WARP_LANE * WARPS) + cta->warp_id;
			int composite			= (THREAD_COMPOSITE * B40C_WARP_THREADS(__B40C_CUDA_ARCH__)) + cta->warp_idx;

			cta->local_counts[WARP_LANE][0] += cta->smem_storage.composite_counters.counters[lane][composite][0];
			cta->local_counts[WARP_LANE][1] += cta->smem_storage.composite_counters.counters[lane][composite][1];
			cta->local_counts[WARP_LANE][2] += cta->smem_storage.composite_counters.counters[lane][composite][2];
			cta->local_counts[WARP_LANE][3] += cta->smem_storage.composite_counters.counters[lane][composite][3];

			Iterate<WARP_LANE, THREAD_COMPOSITE + 1>::ReduceComposites(cta);
		}

		// PlacePartials
		template <typename Cta>
		static __device__ __forceinline__ void PlacePartials(Cta *cta)
		{
			Iterate<WARP_LANE, THREAD_COMPOSITE + 1>::PlacePartials(cta);
		}

		// ResetCounters
		template <typename Cta>
		static __device__ __forceinline__ void ResetCounters(Cta *cta)
		{
			Iterate<WARP_LANE, THREAD_COMPOSITE + 1>::ResetCounters(cta);
		}
	};


	/**
	 * Iterate next lane
	 */
	template <int WARP_LANE, int dummy>
	struct Iterate<WARP_LANE, COMPOSITES_PER_LANE_PER_THREAD, dummy>
	{
		// ReduceComposites
		template <typename Cta>
		static __device__ __forceinline__ void ReduceComposites(Cta *cta)
		{
			Iterate<WARP_LANE + 1, 0>::ReduceComposites(cta);
		}

		// PlacePartials
		template <typename Cta>
		static __device__ __forceinline__ void PlacePartials(Cta *cta)
		{
			int lane				= (WARP_LANE * WARPS) + cta->warp_id;
			int row 				= lane << 2;	// lane * 4;

			cta->smem_storage.aggregate[row + 0][cta->warp_idx] = cta->local_counts[WARP_LANE][0];
			cta->smem_storage.aggregate[row + 1][cta->warp_idx] = cta->local_counts[WARP_LANE][1];
			cta->smem_storage.aggregate[row + 2][cta->warp_idx] = cta->local_counts[WARP_LANE][2];
			cta->smem_storage.aggregate[row + 3][cta->warp_idx] = cta->local_counts[WARP_LANE][3];

			Iterate<WARP_LANE + 1, 0>::PlacePartials(cta);
		}

		// ResetCounters
		template <typename Cta>
		static __device__ __forceinline__ void ResetCounters(Cta *cta)
		{
			cta->local_counts[WARP_LANE][0] = 0;
			cta->local_counts[WARP_LANE][1] = 0;
			cta->local_counts[WARP_LANE][2] = 0;
			cta->local_counts[WARP_LANE][3] = 0;

			Iterate<WARP_LANE + 1, 0>::ResetCounters(cta);
		}
	};

	/**
	 * Terminate iteration
	 */
	template <int dummy>
	struct Iterate<LANES_PER_WARP, 0, dummy>
	{
		// ReduceComposites
		template <typename Cta>
		static __device__ __forceinline__ void ReduceComposites(Cta *cta) {}

		// PlacePartials
		template <typename Cta>
		static __device__ __forceinline__ void PlacePartials(Cta *cta) {}

		// ResetCounters
		template <typename Cta>
		static __device__ __forceinline__ void ResetCounters(Cta *cta) {}
	};

	//---------------------------------------------------------------------
	// Interface
	//---------------------------------------------------------------------

	/**
	 * ReduceComposites
	 */
	template <typename Cta>
	static __device__ __forceinline__ void ReduceComposites(Cta *cta)
	{
		if (cta->warp_id < COMPOSITE_LANES) {
			Iterate<0, 0>::ReduceComposites(cta);
		}
	}

	/**
	 * ReduceComposites
	 */
	template <typename Cta>
	static __device__ __forceinline__ void PlacePartials(Cta *cta)
	{
		if (cta->warp_id < COMPOSITE_LANES) {
			Iterate<0, 0>::PlacePartials(cta);
		}
	}

	/**
	 * ResetCounters
	 */
	template <typename Cta>
	static __device__ __forceinline__ void ResetCounters(Cta *cta)
	{
		Iterate<0, 0>::ResetCounters(cta);
	}
};



} // namespace upsweep
} // namespace distribution
} // namespace radix_sort
} // namespace b40c

