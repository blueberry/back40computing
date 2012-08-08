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
 * Radix sort policy
 ******************************************************************************/

#pragma once

#include "../util/basic_utils.cuh"
#include "../util/io/modified_load.cuh"
#include "../util/io/modified_store.cuh"
#include "../util/ns_umbrella.cuh"

#include "../radix_sort/pass_policy.cuh"
#include "../radix_sort/upsweep/kernel_policy.cuh"
#include "../radix_sort/spine/kernel_policy.cuh"
#include "../radix_sort/downsweep/kernel_policy.cuh"
#include "../radix_sort/partition/kernel_policy.cuh"
#include "../radix_sort/tile/kernel_policy.cuh"

B40C_NS_PREFIX
namespace b40c {
namespace radix_sort {


/******************************************************************************
 * Dispatch policy
 ******************************************************************************/

/**
 * Alternative policies for how much dynamic smem should be allocated to each kernel
 */
enum DynamicSmemConfig
{
	DYNAMIC_SMEM_NONE,			// No dynamic smem for kernels
	DYNAMIC_SMEM_UNIFORM,		// Uniform: pad with dynamic smem so all kernels get the same total smem allocation
	DYNAMIC_SMEM_LCM,			// Least-common-multiple: pad with dynamic smem so kernel occupancy a multiple of the lowest occupancy
};


/**
 * Dispatch policy
 */
template <
	DynamicSmemConfig 	_DYNAMIC_SMEM_CONFIG,
	bool 				_UNIFORM_GRID_SIZE>
struct DispatchPolicy
{
	enum
	{
		UNIFORM_GRID_SIZE = _UNIFORM_GRID_SIZE,
	};

	static const DynamicSmemConfig 	DYNAMIC_SMEM_CONFIG = _DYNAMIC_SMEM_CONFIG;
};


/******************************************************************************
 * Pass policy
 ******************************************************************************/

/**
 * Pass policy
 */
template <
	typename 	_UpsweepPolicy,
	typename 	_SpinePolicy,
	typename 	_DownsweepPolicy,
	typename 	_TilePolicy,
	typename 	_PartitionPolicy,
	typename 	_DispatchPolicy>
struct PassPolicy
{
	typedef _UpsweepPolicy			UpsweepPolicy;
	typedef _SpinePolicy 			SpinePolicy;
	typedef _DownsweepPolicy 		DownsweepPolicy;
	typedef _PartitionPolicy 		PartitionPolicy;
	typedef _DispatchPolicy 		DispatchPolicy;
	typedef _TilePolicy 			TilePolicy;
};


/******************************************************************************
 * Tuned pass policy specializations
 ******************************************************************************/

/**
 * Problem size enumerations
 */
enum ProblemSize
{
	LARGE_PROBLEM,		// > 32K elements
	SMALL_PROBLEM		// <= 32K elements
};


/**
 * Preferred radix digit bits policy
 */
template <int TUNE_ARCH>
struct PreferredDigitBits
{
	enum {
		PREFERRED_BITS = 5,		// All architectures currently prefer 5-bit passes
	};
};


/**
 * Tuned pass policy specializations
 */
template <
	int 			TUNE_ARCH,
	typename 		ProblemInstance,
	ProblemSize 	PROBLEM_SIZE,
	int				RADIX_BITS>
struct TunedPassPolicy;


/**
 * SM20
 */
template <typename ProblemInstance, ProblemSize PROBLEM_SIZE, int RADIX_BITS>
struct TunedPassPolicy<200, ProblemInstance, PROBLEM_SIZE, RADIX_BITS>
{
	enum
	{
		TUNE_ARCH			= 200,
		KEYS_ONLY 			= util::Equals<typename ProblemInstance::ValueType, util::NullType>::VALUE,
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
		b40c::util::io::ld::NONE,				// LOAD_MODIFIER
		b40c::util::io::st::NONE,				// STORE_MODIFIER
		cudaSharedMemBankSizeFourByte,			// SMEM_CONFIG
		EARLY_EXIT>								// EARLY_EXIT
			UpsweepPolicy;

	// Spine-scan kernel policy
	typedef spine::KernelPolicy<
		8,										// LOG_CTA_THREADS
		2,										// LOG_LOAD_VEC_SIZE
		2,										// LOG_LOADS_PER_TILE
		b40c::util::io::ld::NONE,				// LOAD_MODIFIER
		b40c::util::io::st::NONE,				// STORE_MODIFIER
		cudaSharedMemBankSizeFourByte>			// SMEM_CONFIG
			SpinePolicy;

	// Downsweep kernel policy
	typedef downsweep::KernelPolicy<
		RADIX_BITS,								// RADIX_BITS
		4, 										// MIN_CTA_OCCUPANCY
		7,										// LOG_CTA_THREADS
		17,										// THREAD_ELEMENTS
		b40c::util::io::ld::NONE, 				// LOAD_MODIFIER
		b40c::util::io::st::NONE,				// STORE_MODIFIER
		downsweep::SCATTER_TWO_PHASE,			// SCATTER_STRATEGY
		cudaSharedMemBankSizeFourByte,			// SMEM_CONFIG
		EARLY_EXIT>								// EARLY_EXIT
			DownsweepPolicy;

	// Tile kernel policy
	typedef single::KernelPolicy<
		RADIX_BITS,								// RADIX_BITS
		128,									// CTA_THREADS
		((KEYS_ONLY) ? 17 : 9), 				// THREAD_ELEMENTS
		b40c::util::io::ld::NONE, 				// LOAD_MODIFIER
		b40c::util::io::st::NONE,				// STORE_MODIFIER
		cudaSharedMemBankSizeFourByte>			// SMEM_CONFIG
			TilePolicy;

	// Partition kernel policy
	typedef block::KernelPolicy<
		RADIX_BITS,								// RADIX_BITS
		4, 										// MIN_CTA_OCCUPANCY
		128,									// CTA_THREADS
		((KEYS_ONLY) ? 17 : 9), 				// THREAD_ELEMENTS
		b40c::util::io::ld::NONE, 				// LOAD_MODIFIER
		b40c::util::io::st::NONE,				// STORE_MODIFIER
		cudaSharedMemBankSizeFourByte>			// SMEM_CONFIG
			PartitionPolicy;
};





}// namespace radix_sort
}// namespace b40c
B40C_NS_POSTFIX
