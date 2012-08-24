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

#include "../../cub/cub.cuh"
#include "../ns_wrapper.cuh"

#include "dispatch_policy.cuh"
#include "cta_upsweep_pass.cuh"
#include "cta_downsweep_pass.cuh"
#include "cta_scan_pass.cuh"
#include "cta_single_tile.cuh"
#include "cta_hybrid_pass.cuh"


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
 * Tuned policy specializations
 */
template <
	int 			TUNE_ARCH,
	typename 		KeyType,
	typename 		ValueType,
	typename 		SizeT,
	ProblemSize 	PROBLEM_SIZE>
struct TunedPolicy;


/**
 * SM20
 */
template <
	typename 		KeyType,
	typename 		ValueType,
	typename 		SizeT,
	ProblemSize 	PROBLEM_SIZE>
struct TunedPolicy<200, KeyType, ValueType, SizeT, PROBLEM_SIZE>
{
	enum
	{
		RADIX_BITS					= 5,
		KEYS_ONLY 					= cub::Equals<ValueType, cub::NullType>::VALUE,
		LARGE_DATA					= (sizeof(KeyType) > 4) || (sizeof(ValueType) > 4),
	};

	// Dispatch policy
	typedef DispatchPolicy <
		8,										// UPSWEEP_MIN_CTA_OCCUPANCY
		4,										// DOWNSWEEP_MIN_CTA_OCCUPANCY
		4,										// HYBRID_MIN_CTA_OCCUPANCY
		DYNAMIC_SMEM_NONE, 						// UNIFORM_SMEM_ALLOCATION
		true> 									// UNIFORM_GRID_SIZE
			DispatchPolicyT;

	// Upsweep pass CTA policy
	typedef CtaUpsweepPassPolicy<
		RADIX_BITS,								// RADIX_BITS
		128,									// CTA_THREADS
		17,										// THREAD_ITEMS,
		cub::LOAD_NONE,							// LOAD_MODIFIER
		cub::STORE_NONE,						// STORE_MODIFIER
		cudaSharedMemBankSizeFourByte>			// SMEM_CONFIG
			CtaUpsweepPassPolicyT;

	// Spine-scan pass CTA policy
	typedef CtaScanPassPolicy<
		256,									// CTA_THREADS
		4,										// THREAD_STRIP_ITEMS
		4,										// TILE_STRIPS
		cub::LOAD_NONE,							// LOAD_MODIFIER
		cub::STORE_NONE,						// STORE_MODIFIER
		cudaSharedMemBankSizeFourByte>			// SMEM_CONFIG
			CtaScanPassPolicyT;

	// Downsweep pass CTA policy
	typedef CtaDownsweepPassPolicy<
		RADIX_BITS,								// RADIX_BITS
		128,									// CTA_THREADS
		17,										// THREAD_ITEMS
		SCATTER_TWO_PHASE,						// SCATTER_STRATEGY
		cub::LOAD_NONE, 						// LOAD_MODIFIER
		cub::STORE_NONE,						// STORE_MODIFIER
		cudaSharedMemBankSizeFourByte>			// SMEM_CONFIG
			CtaDownsweepPassPolicyT;

	// Single-tile CTA policy
	typedef CtaSingleTilePolicy<
		RADIX_BITS,								// RADIX_BITS
		128,									// CTA_THREADS
		((KEYS_ONLY) ? 17 : 9), 				// THREAD_ITEMS
		cub::LOAD_NONE, 						// LOAD_MODIFIER
		cub::STORE_NONE,						// STORE_MODIFIER
		cudaSharedMemBankSizeFourByte>			// SMEM_CONFIG
			CtaSingleTilePolicyT;

	// Hybrid pass CTA policy
	typedef CtaHybridPassPolicy<
		RADIX_BITS,								// RADIX_BITS
		128,									// CTA_THREADS
		17, 									// UPSWEEP_THREAD_ITEMS
		17, 									// DOWNSWEEP_THREAD_ITEMS
		SCATTER_TWO_PHASE,						// DOWNSWEEP_SCATTER_STRATEGY
		((KEYS_ONLY) ? 17 : 9), 				// SINGLE_TILE_THREAD_ITEMS
		cub::LOAD_NONE, 						// LOAD_MODIFIER
		cub::STORE_NONE,						// STORE_MODIFIER
		cudaSharedMemBankSizeFourByte>			// SMEM_CONFIG
			CtaHybridPassPolicyT;

};





}// namespace radix_sort
}// namespace back40
BACK40_NS_POSTFIX
