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

#include "../../cub/cub.cuh"
#include "../ns_wrapper.cuh"

#include "sort_utils.cuh"
#include "cta_single_tile.cuh"
#include "cta_downsweep_pass.cuh"
#include "cta_upsweep_pass.cuh"


BACK40_NS_PREFIX
namespace back40 {
namespace radix_sort {

//---------------------------------------------------------------------
// Tuning policy types
//---------------------------------------------------------------------


/**
 * Hybrid CTA tuning policy
 */
template <
	int 						RADIX_BITS,						// The number of radix bits, i.e., log2(bins)
	int 						_CTA_THREADS,					// The number of threads per CTA

	int 						UPSWEEP_THREAD_ITEMS,			// The number of consecutive upsweep items to load per thread per tile
	int 						DOWNSWEEP_THREAD_ITEMS,			// The number of consecutive downsweep items to load per thread per tile
	ScatterStrategy 			DOWNSWEEP_SCATTER_STRATEGY,		// Downsweep strategy
	int 						SINGLE_TILE_THREAD_ITEMS,		// The number of consecutive single-tile items to load per thread per tile

	cub::LoadModifier 			LOAD_MODIFIER,					// Load cache-modifier
	cub::StoreModifier			STORE_MODIFIER,					// Store cache-modifier
	cudaSharedMemConfig			_SMEM_CONFIG>					// Shared memory bank size
struct CtaHybridPassPolicy
{
	enum
	{
		CTA_THREADS = _CTA_THREADS,
	};

	static const cudaSharedMemConfig SMEM_CONFIG = _SMEM_CONFIG;

	// Upsweep pass policy
	typedef CtaUpsweepPassPolicy<
		RADIX_BITS,
		CTA_THREADS,
		UPSWEEP_THREAD_ITEMS,
		LOAD_MODIFIER,
		STORE_MODIFIER,
		SMEM_CONFIG> CtaUpsweepPassPolicyT;

	// Downsweep pass policy
	typedef CtaDownsweepPassPolicy<
		RADIX_BITS,
		CTA_THREADS,
		DOWNSWEEP_THREAD_ITEMS,
		DOWNSWEEP_SCATTER_STRATEGY,
		LOAD_MODIFIER,
		STORE_MODIFIER,
		SMEM_CONFIG> CtaDownsweepPassPolicyT;

	// Single tile policy
	typedef CtaSingleTilePolicy<
		RADIX_BITS,
		CTA_THREADS,
		SINGLE_TILE_THREAD_ITEMS,
		LOAD_MODIFIER,
		STORE_MODIFIER,
		SMEM_CONFIG> CtaSingleTilePolicyT;
};

//---------------------------------------------------------------------
// CTA-wide abstractions
//---------------------------------------------------------------------

/**
 * Kernel entry point
 */
template <
	typename 	CtaHybridPassPolicy,
	int 		MIN_CTA_OCCUPANCY,
	typename 	SizeT,
	typename 	KeyType,
	typename 	ValueType>
__launch_bounds__ (
	CtaHybridPassPolicy::CTA_THREADS,
	MIN_CTA_OCCUPANCY)
__global__
void HybridKernel(
	BinDescriptor						*d_bins_in,
	BinDescriptor						*d_bins_out,
	KeyType 							*d_keys_in,
	KeyType 							*d_keys_out,
	KeyType 							*d_keys_final,
	ValueType 							*d_values_in,
	ValueType 							*d_values_out,
	ValueType 							*d_values_final,
	int									low_bit)
{
	enum
	{
		CTA_THREADS				= CtaHybridPassPolicy::CtaUpsweepPassPolicyT::CTA_THREADS,

		SWEEP_RADIX_BITS 		= CtaHybridPassPolicy::CtaUpsweepPassPolicyT::RADIX_BITS,
		SWEEP_RADIX_DIGITS		= 1 << SWEEP_RADIX_BITS,

		WARP_THREADS			= cub::DeviceProps::WARP_THREADS,

		SINGLE_TILE_ITEMS		= CtaHybridPassPolicy::CtaSingleTilePolicyT::TILE_ITEMS,
	};

	// CTA upsweep abstraction
	typedef CtaUpsweepPass<
		typename CtaHybridPassPolicy::CtaUpsweepPassPolicyT,
		KeyType,
		SizeT> CtaUpsweepPassT;

	// CTA downsweep abstraction
	typedef CtaDownsweepPass<
		typename CtaHybridPassPolicy::CtaDownsweepPassPolicyT,
		KeyType,
		ValueType,
		SizeT> CtaDownsweepPassT;

	// CTA single-tile abstraction
	typedef CtaSingleTile<
		typename CtaHybridPassPolicy::CtaSingleTilePolicyT,
		KeyType,
		ValueType> CtaSingleTileT;

	// Warp scan abstraction
	typedef cub::WarpScan<
		SizeT,
		1,
		SWEEP_RADIX_DIGITS> WarpScanT;

	// CTA scan abstraction
	typedef cub::CtaScan<
		SizeT,
		CTA_THREADS> CtaScanT;

	/**
	 * Shared memory storage layout
	 */
	struct SmemStorage
	{
		union
		{
			typename CtaUpsweepPassT::SmemStorage 		upsweep_storage;
			typename CtaDownsweepPassT::SmemStorage 	downsweep_storage;
			typename CtaSingleTileT::SmemStorage 		single_storage;
			typename WarpScanT::SmemStorage				warp_scan_storage;
			typename CtaScanT::SmemStorage				cta_scan_storage;
		};
	};

	// Shared data structures
	__shared__ SmemStorage 								smem_storage;
	__shared__ SizeT									bin_count[SWEEP_RADIX_DIGITS];
	__shared__ unsigned int 							current_bit;
	__shared__ unsigned int 							bits_remaining;
	__shared__ SizeT									num_elements;
	__shared__ SizeT									cta_offset;

	// Retrieve work
	if (threadIdx.x == 0)
	{
		BinDescriptor input_bin 	= d_bins_in[blockIdx.x];
		current_bit 				= input_bin.current_bit - SWEEP_RADIX_BITS;
		bits_remaining 				= input_bin.current_bit - low_bit;
		num_elements 				= input_bin.num_elements;
		cta_offset					= input_bin.offset;

		// Reset current partition descriptor
		d_bins_in[blockIdx.x].num_elements = 0;
	}

	__syncthreads();

	if (num_elements <= SINGLE_TILE_ITEMS)
	{
		// Sort input tile
		CtaSingleTileT::Sort(
			smem_storage.single_storage,
			d_keys_in + cta_offset,
			d_keys_final + cta_offset,
			d_values_in + cta_offset,
			d_values_final + cta_offset,
			low_bit,
			bits_remaining,
			num_elements);
	}
	else
	{
		// Compute bin-count for each radix digit (valid in tid < RADIX_DIGITS)
		CtaUpsweepPassT::UpsweepPass(
			smem_storage.upsweep_storage,
			d_keys_in + cta_offset,
			current_bit,
			num_elements,
			bin_count[threadIdx.x]);

		// Warp prefix sum
		SizeT bin_prefix;
		if (threadIdx.x < SWEEP_RADIX_DIGITS)
		{
			WarpScanT::ExclusiveSum(
				smem_storage.warp_scan_storage,
				bin_count[threadIdx.x],
				bin_prefix);
		}

		// Distribute keys
		CtaDownsweepPassT::DownsweepPass(
			smem_storage.downsweep_storage,
			bin_prefix,
			d_keys_in + cta_offset,
			d_keys_out + cta_offset,
			d_values_in + cta_offset,
			d_values_out + cta_offset,
			current_bit,
			num_elements);

		// Output bin
		if ((threadIdx.x < SWEEP_RADIX_DIGITS) && (bin_count > 0))
		{
			BinDescriptor bin(
				cta_offset + bin_prefix,
				bin_count[threadIdx.x],
				current_bit);

			d_bins_out[(blockIdx.x * SWEEP_RADIX_DIGITS) + threadIdx.x] = bin;
		}

	}
}


/**
 * Hybrid kernel props
 */
template <
	typename KeyType,
	typename ValueType,
	typename SizeT>
struct HybridKernelProps : cub::KernelProps
{
	// Kernel function type
	typedef void (*KernelFunc)(
		BinDescriptor*,
		BinDescriptor*,
		KeyType*,
		KeyType*,
		KeyType*,
		ValueType*,
		ValueType*,
		ValueType*,
		int);

	// Fields
	KernelFunc 					kernel_func;
	cudaSharedMemConfig 		sm_bank_config;

	/**
	 * Initializer
	 */
	template <
		typename CtaHybridPassPolicy,
		typename OpaqueCtaHybridPassPolicy,
		int MIN_CTA_OCCUPANCY>
	cudaError_t Init(const cub::CudaProps &cuda_props)	// CUDA properties for a specific device
	{
		// Initialize fields
		kernel_func 			= HybridKernel<OpaqueCtaHybridPassPolicy, MIN_CTA_OCCUPANCY, SizeT>;
		sm_bank_config 			= CtaHybridPassPolicy::SMEM_CONFIG;

		// Initialize super class
		return cub::KernelProps::Init(
			kernel_func,
			CtaHybridPassPolicy::CTA_THREADS,
			cuda_props);
	}

	/**
	 * Initializer
	 */
	template <typename CtaHybridPassPolicy>
	cudaError_t Init(const cub::CudaProps &cuda_props)	// CUDA properties for a specific device
	{
		return Init<CtaHybridPassPolicy, CtaHybridPassPolicy>(cuda_props);
	}

};


} // namespace radix_sort
} // namespace back40
BACK40_NS_POSTFIX
