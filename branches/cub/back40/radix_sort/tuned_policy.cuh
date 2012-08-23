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
 * Tuned derivatives of radix sorting policy
 ******************************************************************************/

#pragma once

#include "cub/cub.cuh"

#include "../ns_wrapper.cuh"
#include "../radix_sort/cta/cta_downsweep_pass.cuh"
#include "../radix_sort/cta/cta_hybrid_pass.cuh"
#include "../radix_sort/cta/cta_scan_pass.cuh"
#include "../radix_sort/cta/cta_single_tile.cuh"
#include "../radix_sort/cta/cta_upsweep_pass.cuh"

BACK40_NS_PREFIX
namespace back40 {
namespace radix_sort {


/**
 * Problem size enumerations
 */
enum ProblemSize
{
	LARGE_PROBLEM,		// > 64K elements
	SMALL_PROBLEM		// <= 64K elements
};


/**
 * Tuned pass policy specializations
 */
template <
	int 			TUNE_ARCH,
	typename 		KeyType,
	typename 		ValueType,
	typename 		SizeT,
	ProblemSize 	PROBLEM_SIZE>
struct TunedPassPolicy;


/**
 * SM20
 */
template <typename KeyType, typename ValueType, typename SizeT, ProblemSize PROBLEM_SIZE>
struct TunedPassPolicy<200, KeyType, ValueType, SizeT, PROBLEM_SIZE>
{
	enum
	{
		RADIX_BITS			= 5,
		TUNE_ARCH			= 200,
		KEYS_ONLY 			= cub::Equals<ValueType, cub::NullType>::VALUE,
		LARGE_DATA			= (sizeof(typename ProblemInstance::KeyType) > 4) || (sizeof(typename ProblemInstance::ValueType) > 4),
		EARLY_EXIT			= false,
	};

	// Dispatch policy
	typedef DispatchPolicy <
		DYNAMIC_SMEM_NONE, 						// UNIFORM_SMEM_ALLOCATION
		true> 									// UNIFORM_GRID_SIZE
			DispatchPolicy;

	// Upsweep kernel policy
	typedef upsweep::KernelPolicy<
		RADIX_BITS,								// RADIX_BITS
		8,										// MIN_CTA_OCCUPANCY
		7,										// LOG_CTA_THREADS
		17,										// ELEMENTS_PER_THREAD,
		back40::cub::io::ld::NONE,				// LOAD_MODIFIER
		back40::cub::io::st::NONE,				// STORE_MODIFIER
		cudaSharedMemBankSizeFourByte,			// SMEM_CONFIG
		EARLY_EXIT>								// EARLY_EXIT
			UpsweepPolicy;

	// Spine-scan kernel policy
	typedef spine::KernelPolicy<
		8,										// LOG_CTA_THREADS
		2,										// LOG_LOAD_VEC_SIZE
		2,										// LOG_LOADS_PER_TILE
		back40::cub::io::ld::NONE,				// LOAD_MODIFIER
		back40::cub::io::st::NONE,				// STORE_MODIFIER
		cudaSharedMemBankSizeFourByte>			// SMEM_CONFIG
			SpinePolicy;

	// Downsweep kernel policy
	typedef downsweep::KernelPolicy<
		RADIX_BITS,								// RADIX_BITS
		4, 										// MIN_CTA_OCCUPANCY
		7,										// LOG_CTA_THREADS
		17,										// THREAD_ELEMENTS
		back40::cub::io::ld::NONE, 				// LOAD_MODIFIER
		back40::cub::io::st::NONE,				// STORE_MODIFIER
		downsweep::SCATTER_TWO_PHASE,			// SCATTER_STRATEGY
		cudaSharedMemBankSizeFourByte,			// SMEM_CONFIG
		EARLY_EXIT>								// EARLY_EXIT
			DownsweepPolicy;

	// Tile kernel policy
	typedef single::KernelPolicy<
		RADIX_BITS,								// RADIX_BITS
		128,									// CTA_THREADS
		((KEYS_ONLY) ? 17 : 9), 				// THREAD_ELEMENTS
		back40::cub::io::ld::NONE, 				// LOAD_MODIFIER
		back40::cub::io::st::NONE,				// STORE_MODIFIER
		cudaSharedMemBankSizeFourByte>			// SMEM_CONFIG
			TilePolicy;

	// BinDescriptor kernel policy
	typedef block::KernelPolicy<
		RADIX_BITS,								// RADIX_BITS
		4, 										// MIN_CTA_OCCUPANCY
		128,									// CTA_THREADS
		((KEYS_ONLY) ? 17 : 9), 				// THREAD_ELEMENTS
		back40::cub::io::ld::NONE, 				// LOAD_MODIFIER
		back40::cub::io::st::NONE,				// STORE_MODIFIER
		cudaSharedMemBankSizeFourByte>			// SMEM_CONFIG
			BinDescriptorPolicy;
};





}// namespace radix_sort
}// namespace back40
BACK40_NS_POSTFIX
