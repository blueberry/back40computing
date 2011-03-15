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
 * SerialSoaScan
 ******************************************************************************/

#pragma once

#include <b40c/util/soa_tuple.cuh>

namespace b40c {
namespace util {
namespace scan {
namespace soa {


/**
 * Have each thread concurrently perform a serial scan over its
 * specified tuple segment (in place).  Returns the inclusive total reduction.
 */
template <
	typename Tuple,						// Tuple of partials
	typename RakingSoa,			// Tuple of SOA raking segments
	int NUM_ELEMENTS,					// Length of SOA array segment(s) to scan
	bool EXCLUSIVE,
	Tuple ScanOp(Tuple&, Tuple&)>
struct SerialSoaScan;


/**
 * Inclusive serial scan
 */
template <
	typename Tuple,						// Tuple of partials
	typename RakingSoa,			// Tuple of SOA raking segments
	int NUM_ELEMENTS,
	Tuple ScanOp(Tuple&, Tuple&)>
struct SerialSoaScan <Tuple, RakingSoa, NUM_ELEMENTS, false, ScanOp>
{
	// Iterate
	template <int COUNT, int TOTAL>
	struct Iterate
	{
		static __host__ __device__ __forceinline__ Tuple Invoke(
			RakingSoa raking_partials,
			RakingSoa raking_results,
			Tuple exclusive_partial)
		{
			// Load current partial
			Tuple current_partial = raking_partials.template Get<Tuple>(COUNT);

			// Compute inclusive partial from exclusive and current partials
			Tuple inclusive_partial = ScanOp(exclusive_partial, current_partial);

			// Store inclusive partial because this is an inclusive scan
			raking_results.Set(inclusive_partial, COUNT);

			// Recurse
			return Iterate<COUNT + 1, TOTAL>::Invoke(
				raking_partials, raking_results, inclusive_partial);
		}
	};

	// Terminate
	template <int TOTAL>
	struct Iterate<TOTAL, TOTAL>
	{
		static __host__ __device__ __forceinline__ Tuple Invoke(
			RakingSoa raking_partials,
			RakingSoa raking_results,
			Tuple exclusive_partial)
		{
			return exclusive_partial;
		}
	};

	// Interface
	static __host__ __device__ __forceinline__ Tuple Invoke(
		RakingSoa raking_partials,
		Tuple exclusive_partial)			// Exclusive partial to seed with
	{
		return Iterate<0, NUM_ELEMENTS>::Invoke(raking_partials, raking_partials, exclusive_partial);
	}

	// Interface
	static __host__ __device__ __forceinline__ Tuple Invoke(
		RakingSoa raking_partials,
		RakingSoa raking_results,
		Tuple exclusive_partial)			// Exclusive partial to seed with
	{
		return Iterate<0, NUM_ELEMENTS>::Invoke(raking_partials, raking_results, exclusive_partial);
	}
};


/**
 * Exclusive serial scan
 */
template <
	typename Tuple,						// Tuple of partials
	typename RakingSoa,			// Tuple of SOA raking segments
	int NUM_ELEMENTS,
	Tuple ScanOp(Tuple&, Tuple&)>
struct SerialSoaScan <Tuple, RakingSoa, NUM_ELEMENTS, true, ScanOp>
{
	// Iterate
	template <int COUNT, int TOTAL>
	struct Iterate
	{
		static __host__ __device__ __forceinline__ Tuple Invoke(
			RakingSoa raking_partials,
			RakingSoa raking_results,
			Tuple exclusive_partial)
		{
			// Load current partial
			Tuple current_partial = raking_partials.template Get<Tuple>(COUNT);

			// Compute inclusive partial from exclusive and current partials
			Tuple inclusive_partial = ScanOp(exclusive_partial, current_partial);

			// Store exclusive partial because this is an exclusive scan
			raking_results.Set(exclusive_partial, COUNT);

			// Recurse
			return Iterate<COUNT + 1, TOTAL>::Invoke(
				raking_partials, raking_results, inclusive_partial);
		}
	};

	// Terminate
	template <int TOTAL>
	struct Iterate<TOTAL, TOTAL>
	{
		static __host__ __device__ __forceinline__ Tuple Invoke(
			RakingSoa raking_partials,
			RakingSoa raking_results,
			Tuple exclusive_partial)
		{
			return exclusive_partial;
		}
	};

	// Interface
	static __host__ __device__ __forceinline__ Tuple Invoke(
		RakingSoa raking_partials,
		Tuple exclusive_partial)			// Exclusive partial to seed with
	{
		return Iterate<0, NUM_ELEMENTS>::Invoke(raking_partials, raking_partials, exclusive_partial);
	}

	// Interface
	static __host__ __device__ __forceinline__ Tuple Invoke(
		RakingSoa raking_partials,
		RakingSoa raking_results,
		Tuple exclusive_partial)			// Exclusive partial to seed with
	{
		return Iterate<0, NUM_ELEMENTS>::Invoke(raking_partials, raking_results, exclusive_partial);
	}
};



/**
 * Have each thread concurrently perform a serial scan over its
 * specified tuple segment (in place).  Returns the inclusive total reduction.
 */
template <
	typename Tuple,						// Tuple of partials
	typename RakingSoa,			// Tuple of SOA raking segments
	int LANE,							// Lane segment in 2D array to serially reduce
	int NUM_ELEMENTS,					// Length of SOA array segment(s) to scan
	bool EXCLUSIVE,
	Tuple ScanOp(Tuple&, Tuple&)>
struct SerialSoaScanLane;


/**
 * Inclusive serial scan
 */
template <
	typename Tuple,						// Tuple of partials
	typename RakingSoa,			// Tuple of SOA raking segments
	int LANE,							// Lane segment in 2D array to serially reduce
	int NUM_ELEMENTS,
	Tuple ScanOp(Tuple&, Tuple&)>
struct SerialSoaScanLane <Tuple, RakingSoa, LANE, NUM_ELEMENTS, false, ScanOp>
{
	// Iterate
	template <int COUNT, int TOTAL>
	struct Iterate
	{
		static __host__ __device__ __forceinline__ Tuple Invoke(
			RakingSoa raking_partials,
			RakingSoa raking_results,
			Tuple exclusive_partial)
		{
			// Load current partial
			Tuple current_partial = raking_partials.template Get<LANE, Tuple>(COUNT);

			// Compute inclusive partial from exclusive and current partials
			Tuple inclusive_partial = ScanOp(exclusive_partial, current_partial);

			// Store inclusive partial because this is an inclusive scan
			raking_results.template Set<LANE>(inclusive_partial, COUNT);

			// Recurse
			return Iterate<COUNT + 1, TOTAL>::Invoke(
				raking_partials, raking_results, inclusive_partial);
		}
	};

	// Terminate
	template <int TOTAL>
	struct Iterate<TOTAL, TOTAL>
	{
		static __host__ __device__ __forceinline__ Tuple Invoke(
			RakingSoa raking_partials,
			RakingSoa raking_results,
			Tuple exclusive_partial)
		{
			return exclusive_partial;
		}
	};

	// Interface
	static __host__ __device__ __forceinline__ Tuple Invoke(
		RakingSoa raking_partials,
		Tuple exclusive_partial)			// Exclusive partial to seed with
	{
		return Iterate<0, NUM_ELEMENTS>::Invoke(raking_partials, raking_partials, exclusive_partial);
	}

	// Interface
	static __host__ __device__ __forceinline__ Tuple Invoke(
		RakingSoa raking_partials,
		RakingSoa raking_results,
		Tuple exclusive_partial)			// Exclusive partial to seed with
	{
		return Iterate<0, NUM_ELEMENTS>::Invoke(raking_partials, raking_results, exclusive_partial);
	}
};


/**
 * Exclusive serial scan
 */
template <
	typename Tuple,						// Tuple of partials
	typename RakingSoa,			// Tuple of SOA raking segments
	int LANE,							// Lane segment in 2D array to serially reduce
	int NUM_ELEMENTS,
	Tuple ScanOp(Tuple&, Tuple&)>
struct SerialSoaScanLane <Tuple, RakingSoa, LANE, NUM_ELEMENTS, true, ScanOp>
{
	// Iterate
	template <int COUNT, int TOTAL>
	struct Iterate
	{
		static __host__ __device__ __forceinline__ Tuple Invoke(
			RakingSoa raking_partials,
			RakingSoa raking_results,
			Tuple exclusive_partial)
		{
			// Load current partial
			Tuple current_partial = raking_partials.template Get<LANE, Tuple>(COUNT);

			// Compute inclusive partial from exclusive and current partials
			Tuple inclusive_partial = ScanOp(exclusive_partial, current_partial);

			// Store exclusive partial because this is an exclusive scan
			raking_results.template Set<LANE>(exclusive_partial, COUNT);

			// Recurse
			return Iterate<COUNT + 1, TOTAL>::Invoke(raking_partials, raking_results, inclusive_partial);
		}
	};

	// Terminate
	template <int TOTAL>
	struct Iterate<TOTAL, TOTAL>
	{
		static __host__ __device__ __forceinline__ Tuple Invoke(
			RakingSoa raking_partials,
			RakingSoa raking_results,
			Tuple exclusive_partial)
		{
			return exclusive_partial;
		}
	};

	// Interface
	static __host__ __device__ __forceinline__ Tuple Invoke(
		RakingSoa raking_partials,
		Tuple exclusive_partial)			// Exclusive partial to seed with
	{
		return Iterate<0, NUM_ELEMENTS>::Invoke(raking_partials, raking_partials, exclusive_partial);
	}

	// Interface
	static __host__ __device__ __forceinline__ Tuple Invoke(
		RakingSoa raking_partials,
		RakingSoa raking_results,
		Tuple exclusive_partial)			// Exclusive partial to seed with
	{
		return Iterate<0, NUM_ELEMENTS>::Invoke(raking_partials, raking_results, exclusive_partial);
	}
};

} // namespace soa
} // namespace scan
} // namespace util
} // namespace b40c

