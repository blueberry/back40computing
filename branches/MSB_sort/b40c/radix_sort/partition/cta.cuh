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
 *
 ******************************************************************************/

#pragma once

#include "../../util/basic_utils.cuh"
#include "../../util/io/modified_load.cuh"
#include "../../util/io/modified_store.cuh"
#include "../../util/ns_umbrella.cuh"

#include "../../radix_sort/sort_utils.cuh"
#include "../../radix_sort/cta_radix_sort.cuh"
#include "../../radix_sort/upsweep/cta.cuh"

B40C_NS_PREFIX
namespace b40c {
namespace radix_sort {
namespace partition {


/**
 *
 */
template <
	typename KernelPolicy,
	typename SizeT,
	typename KeyType,
	typename ValueType>
struct Cta
{
	//---------------------------------------------------------------------
	// Type definitions and constants
	//---------------------------------------------------------------------

	// Block CTA abstraction
	typedef block::Cta<
		typename KernelPolicy::Upsweep,
		KeyType,
		ValueType> BlockCta;

	// Upsweep CTA abstraction
	typedef upsweep::Cta<
		typename KernelPolicy::Upsweep,
		SizeT,
		KeyType> UpsweepCta;

	// Downsweep CTA abstraction
	typedef downsweep::Cta<
		typename KernelPolicy::Downsweep,
		SizeT,
		KeyType,
		ValueType> DownsweepCta;


	/**
	 * Shared memory storage layout
	 */
	struct SmemStorage
	{
		Partition								partition;

		union
		{
			typename BlockCta::SmemStorage 		block_storage;
			typename UpsweepCta::SmemStorage 	upsweep_storage;
			typename DownsweepCta::SmemStorage 	downsweep_storage;
		};
	};


	//---------------------------------------------------------------------
	// Interface
	//---------------------------------------------------------------------

	/**
	 * Process work range
	 */
	static __device__ __forceinline__ void ProcessWorkRange(
		Partition		*d_partitions_in,
		Partition		*d_partitions_out,
		KeyType 		*d_keys_in,
		KeyType 		*d_keys_out,
		KeyType 		*d_keys_final,
		ValueType 		*d_values_in,
		ValueType 		*d_values_out,
		ValueType 		*d_values_final,
		int				low_bit)
	{
		// Retrieve work
		if (threadIdx.x == 0)
		{
			smem_storage.partition = d_partitions_in[blockIdx.x];
/*
			printf("\tCTA %d loaded partition (low bit %d, current bit %d) of %d elements at offset %d\n",
				blockIdx.x,
				low_bit,
				smem_storage.partition.current_bit,
				smem_storage.partition.num_elements,
				smem_storage.partition.offset);
*/
		}

		__syncthreads();

		// Quit if there is no work
		if (smem_storage.partition.num_elements == 0) return;

		// Choose whether to block-sort or pass-sort
		if (smem_storage.partition.num_elements < TILE_ELEMENTS)
		{
			// Construct CTA abstraction
			Cta cta(
				smem_storage,
				d_partitions_out,
				d_keys_in,
				d_keys_out,
				d_keys_final,
				d_values_in,
				d_values_out,
				d_values_final,
				low_bit);

			// Block sort the remainder of the radix bits
			cta.BlockSort();
		}
		else
		{
			// Distribution sort a single radix pass
			cta.DistributionSort();
		}
	}

};



/**
 * Kernel entry point
 */
template <
	typename KernelPolicy,
	typename SizeT,
	typename KeyType,
	typename ValueType>
__launch_bounds__ (KernelPolicy::CTA_THREADS, KernelPolicy::MIN_CTA_OCCUPANCY)
__global__
void Kernel(
	Partition							*d_partitions_in,
	Partition							*d_partitions_out,
	KeyType 							*d_keys_in,
	KeyType 							*d_keys_out,
	KeyType 							*d_keys_final,
	ValueType 							*d_values_in,
	ValueType 							*d_values_out,
	ValueType 							*d_values_final,
	int									low_bit)
{
	// CTA abstraction type
	typedef Cta<KernelPolicy, SizeT, KeyType, ValueType> Cta;

	// Shared memory pool
	__shared__ typename Cta::SmemStorage smem_storage;

	Cta::ProcessWorkRange(
		smem_storage,
		d_partitions_in,
		d_partitions_out,
		d_keys_in,
		d_keys_out,
		d_keys_final,
		d_values_in,
		d_values_out,
		d_values_final,
		low_bit);
}




} // namespace partition
} // namespace radix_sort
} // namespace b40c
B40C_NS_POSTFIX
