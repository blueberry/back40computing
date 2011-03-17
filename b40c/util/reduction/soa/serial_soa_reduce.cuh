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
 * Thanks!
 * 
 ******************************************************************************/

/******************************************************************************
 * SerialSoaReduce
 ******************************************************************************/

#pragma once

#include <b40c/util/soa_tuple.cuh>

namespace b40c {
namespace util {
namespace reduction {
namespace soa {


/**
 * Have each thread concurrently perform a serial sequence reduction
 * over its specified array segment
 */
template <
	typename Tuple,						// Tuple of partials
	typename RakingSoa,			// Tuple of SOA raking segments
	int NUM_ELEMENTS,					// Length of SOA array segment(s) to reduce
	Tuple ReductionOp(Tuple&, Tuple&)>
struct SerialSoaReduce
{
	// Iterate
	template <int COUNT, int TOTAL>
	struct Iterate
	{
		static __host__ __device__ __forceinline__ Tuple Invoke(
			RakingSoa raking_partials,
			Tuple exclusive_partial)
		{
			// Load current partial
			Tuple current_partial = raking_partials.template Get<Tuple>(COUNT);

			// Compute inclusive partial from exclusive and current partials
			Tuple inclusive_partial = ReductionOp(exclusive_partial, current_partial);

			// Recurse
			return Iterate<COUNT + 1, TOTAL>::Invoke(raking_partials, inclusive_partial);
		}
	};

	// Terminate
	template <int TOTAL>
	struct Iterate<TOTAL, TOTAL>
	{
		static __host__ __device__ __forceinline__ Tuple Invoke(
			RakingSoa raking_partials,
			Tuple exclusive_partial)
		{
			return exclusive_partial;
		}
	};

	// Interface
	static __host__ __device__ __forceinline__ Tuple Invoke(
		RakingSoa raking_partials)
	{
		// Get first partial
		Tuple inclusive_partial = raking_partials.template Get<Tuple>(0);
		return Iterate<1, NUM_ELEMENTS>::Invoke(raking_partials, inclusive_partial);
	}

	// Interface
	static __host__ __device__ __forceinline__ Tuple Invoke(
		RakingSoa raking_partials,
		Tuple exclusive_partial)
	{
		// Get first partial
		Tuple current_partial = raking_partials.template Get<Tuple>(0);
		Tuple inclusive_partial = ReductionOp(exclusive_partial, current_partial);
		return Iterate<1, NUM_ELEMENTS>::Invoke(raking_partials, inclusive_partial);
	}
};


/**
 * Have each thread concurrently perform a serial sequence reduction
 * over its specified array segment
 */
template <
	typename Tuple,						// Tuple of partials
	typename RakingSoa,					// Tuple of SOA raking segments
	int LANE,							// Lane segment in 2D array to serially reduce
	int NUM_ELEMENTS,					// Length of SOA array segment(s) to reduce
	Tuple ReductionOp(Tuple&, Tuple&)>
struct SerialSoaReduceLane
{
	// Iterate
	template <int COUNT, int TOTAL>
	struct Iterate
	{
		static __host__ __device__ __forceinline__ Tuple Invoke(
			RakingSoa raking_partials,
			Tuple exclusive_partial)
		{
			// Load current partial
			Tuple current_partial =	raking_partials.template Get<LANE, Tuple>(COUNT);

			// Compute inclusive partial from exclusive and current partials
			Tuple inclusive_partial = ReductionOp(exclusive_partial, current_partial);

			// Recurse
			return Iterate<COUNT + 1, TOTAL>::Invoke(
				raking_partials, inclusive_partial);
		}
	};

	// Terminate
	template <int TOTAL>
	struct Iterate<TOTAL, TOTAL>
	{
		static __host__ __device__ __forceinline__ Tuple Invoke(
			RakingSoa raking_partials,
			Tuple exclusive_partial)
		{
			return exclusive_partial;
		}
	};

	// Interface
	static __host__ __device__ __forceinline__ Tuple Invoke(
		RakingSoa raking_partials)
	{
		// Get first partial
		Tuple inclusive_partial = raking_partials.template Get<LANE, Tuple>(0);

		return Iterate<1, NUM_ELEMENTS>::Invoke(raking_partials, inclusive_partial);
	}

	// Interface
	static __host__ __device__ __forceinline__ Tuple Invoke(
		RakingSoa raking_partials,
		Tuple exclusive_partial)
	{
		// Get first partial
		Tuple current_partial = raking_partials.template Get<LANE, Tuple>(0);
		Tuple inclusive_partial = ReductionOp(exclusive_partial, current_partial);
		return Iterate<1, NUM_ELEMENTS>::Invoke(raking_partials, inclusive_partial);
	}

};


} // namespace soa
} // namespace reduction
} // namespace util
} // namespace b40c

