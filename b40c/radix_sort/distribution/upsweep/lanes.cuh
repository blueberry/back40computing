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
 * Upsweep lane processing abstraction
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

#include <b40c/radix_sort/sort_utils.cuh>

namespace b40c {
namespace radix_sort {
namespace distribution {
namespace upsweep {


/**
 * Lanes
 */
template <typename KernelConfig>
struct Lanes
{
	enum {
		COMPOSITE_LANES = KernelConfig::COMPOSITE_LANES,
	};


	//---------------------------------------------------------------------
	// Iteration Structures
	//---------------------------------------------------------------------

	/**
	 * Iterate lane
	 */
	template <int LANE, int dummy = 0>
	struct Iterate
	{
		// ResetCompositeCounters
		template <typename Cta>
		static __device__ __forceinline__ void ResetCompositeCounters(Cta *cta)
		{
			cta->smem_storage.composite_counters.words[LANE][threadIdx.x] = 0;
			Iterate<LANE + 1>::ResetCompositeCounters(cta);
		}
	};

	/**
	 * Terminate iteration
	 */
	template <int dummy>
	struct Iterate<COMPOSITE_LANES, dummy>
	{
		// ResetCompositeCounters
		template <typename Cta>
		static __device__ __forceinline__ void ResetCompositeCounters(Cta *cta) {}
	};

	//---------------------------------------------------------------------
	// Interface
	//---------------------------------------------------------------------

	/**
	 * ResetCompositeCounters
	 */
	template <typename Cta>
	static __device__ __forceinline__ void ResetCompositeCounters(Cta *cta)
	{
		Iterate<0>::ResetCompositeCounters(cta);
	}
};


} // namespace upsweep
} // namespace distribution
} // namespace radix_sort
} // namespace b40c

