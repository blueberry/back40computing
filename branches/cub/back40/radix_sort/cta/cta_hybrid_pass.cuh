/******************************************************************************
 * 
 * Copyright (c) 2010-2012, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2012, NVIDIA CORPORATION.  All rights reserved.
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
 ******************************************************************************/

/******************************************************************************
 * "Hybrid" CTA abstraction for locally sorting small blocks or performing
 * global distribution passes over large blocks
 ******************************************************************************/

#pragma once

#include "../../util/basic_utils.cuh"
#include "../../util/io/modified_load.cuh"
#include "../../util/io/modified_store.cuh"
#include "../../util/ns_wrapper.cuh"

#include "../../radix_sort/sort_utils.cuh"
#include "../../radix_sort/cta/cta_single_tile.cuh"
#include "../../radix_sort/cta/cta_upsweep_pass.cuh"
#include "../../radix_sort/cta/cta_downsweep_pass.cuh"

BACK40_NS_PREFIX
namespace back40 {
namespace radix_sort {
namespace cta {


/**
 * Hybrid CTA tuning policy.
 */
template <
	typename CtaSingleTilePolicy,
	typename CtaUpsweepPassPolicy,
	typename CtaDownsweepPassPolicy>
struct CtaHybridPolicy
{
	typename _CtaSingleTilePolicy 		CtaSingleTilePolicy;
	typename _CtaUpsweepPassPolicy 		CtaUpsweepPassPolicy;
	typename _CtaDownsweepPassPolicy 	CtaDownsweepPassPolicy;
};



/**
 * "Hybrid" CTA abstraction for locally sorting small blocks or performing
 * global distribution passes over large blocks
 */
template <
	typename CtaHybridPolicy,
	typename KeyType,
	typename ValueType,
	typename SizeT>
class CtaHybridPass
{
private:

	//---------------------------------------------------------------------
	// Type definitions and constants
	//---------------------------------------------------------------------

	// Single-tile CTA abstraction
	typedef CtaSingle<
		typename CtaHybridPolicy::Upsweep,
		KeyType,
		ValueType> SingleCta;

	// Upsweep CTA abstraction
	typedef CtaUpsweep<
		typename CtaHybridPolicy::Upsweep,
		KeyType,
		SizeT> UpsweepCta;

	// Downsweep CTA abstraction
	typedef CtaDownsweep<
		typename CtaHybridPolicy::Downsweep,
		KeyType,
		ValueType,
		SizeT> DownsweepCta;

public:

	/**
	 * Shared memory storage layout
	 */
	struct SmemStorage
	{
		union
		{
			typename BlockCta::SmemStorage 		block_storage;
			typename UpsweepCta::SmemStorage 	upsweep_storage;
			typename DownsweepCta::SmemStorage 	downsweep_storage;
			volatile SizeT						warpscan[2][PASS_RADIX_DIGITS];
		};
	};


	//---------------------------------------------------------------------
	// Interface
	//---------------------------------------------------------------------

	/**
	 * Process work range
	 */
	static __device__ __forceinline__ void Sort(
		SmemStorage 	&cta_smem_storage,
		KeyType 		*d_keys_in,
		KeyType 		*d_keys_out,
		KeyType 		*d_keys_final,
		ValueType 		*d_values_in,
		ValueType 		*d_values_out,
		ValueType 		*d_values_final,
		int				current_bit,
		int				low_bit,
		SizeT			num_elements,
		int				&bin_count,			// The digit count for tid'th bin (output param, valid in the first RADIX_DIGITS threads)
		int				&bin_prefix)		// The base offset for each digit (valid in the first RADIX_DIGITS threads)
	{
		bin_count = 0;
		bin_prefix = 0;

		// Choose whether to block-sort or pass-sort
		if (num_elements < TILE_ELEMENTS)
		{
			// Perform block sort
			BlockCta::Sort(
				cta_smem_storage.block_storage,
				d_keys_in,
				d_keys_final,
				d_values_in,
				d_values_final,
				low_bit,
				cta_smem_storage.partition.current_bit - low_bit,
				cta_smem_storage.partition.offset,
				cta_smem_storage.partition.num_elements);
		}
		else
		{
			// Compute bin-count for each radix digit (valid in tid < RADIX_DIGITS)
			SizeT bin_count;
			UpsweepCta::Upsweep(
				cta_smem_storage.upsweep_storage,
				d_keys_in,
				current_bit,
				num_elements,
				bin_count);

			__syncthreads();

			// Exclusive scan across bin counts
			SizeT bin_prefix;
			if (threadIdx.x < PASS_RADIX_DIGITS)
			{
				// Initialize warpscan identity regions
				warpscan[0][threadIdx.x] = 0;

				// Warpscan
				SizeT partial = bin_count;
				warpscan[1][threadIdx.x] = partial;

				#pragma unroll
				for (int STEP = 0; STEP < LOG_WARP_THREADS; STEP++)
				{
					partial += warpscan[1][threadIdx.x - (1 << STEP)];
					warpscan[1][threadIdx.x] = partial;
				}

				bin_prefix = partial - bin_count;
			}

			// Note: no syncthreads() necessary

			// Distribute keys
			DownsweepCta::Downsweep(
				cta_smem_storage.downsweep_storage,
				bin_prefix,
				d_keys_in,
				d_keys_out,
				d_values_in,
				d_values_out,
				current_bit,
				num_elements);
		}
	}
};






} // namespace cta
} // namespace radix_sort
} // namespace back40
BACK40_NS_POSTFIX
