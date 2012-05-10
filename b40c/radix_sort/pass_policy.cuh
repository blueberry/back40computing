/******************************************************************************
 * 
 * Copyright 2010-2012 Duane Merrill
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
 ******************************************************************************/

/******************************************************************************
 * Radix sort policy
 ******************************************************************************/

#pragma once

#include <b40c/util/basic_utils.cuh>
#include <b40c/util/io/modified_load.cuh>
#include <b40c/util/io/modified_store.cuh>

#include <b40c/radix_sort/pass_policy.cuh>
#include <b40c/radix_sort/upsweep/kernel_policy.cuh>
#include <b40c/radix_sort/spine/kernel_policy.cuh>
#include <b40c/radix_sort/downsweep/kernel_policy.cuh>

namespace b40c {
namespace radix_sort {


/******************************************************************************
 * Dispatch policy
 ******************************************************************************/

/**
 * Dispatch policy
 */
template <
	int			_TUNE_ARCH,
	int 		_RADIX_BITS,
	bool 		_UNIFORM_SMEM_ALLOCATION,
	bool 		_UNIFORM_GRID_SIZE>
struct DispatchPolicy
{
	enum {
		TUNE_ARCH					= _TUNE_ARCH,
		RADIX_BITS					= _RADIX_BITS,
		UNIFORM_SMEM_ALLOCATION 	= _UNIFORM_SMEM_ALLOCATION,
		UNIFORM_GRID_SIZE 			= _UNIFORM_GRID_SIZE,
	};
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
	typename 	_DispatchPolicy>
struct PassPolicy
{
	typedef _UpsweepPolicy			UpsweepPolicy;
	typedef _SpinePolicy 			SpinePolicy;
	typedef _DownsweepPolicy 		DownsweepPolicy;
	typedef _DispatchPolicy 		DispatchPolicy;
};


/******************************************************************************
 * Tuned pass policy specializations
 ******************************************************************************/

/**
 * Problem size enumerations
 */
enum ProblemSize {
	LARGE_PROBLEM,		// > 32K elements
	SMALL_PROBLEM		// <= 32K elements
};


/**
 * Tuned pass policy specializations
 */
template <
	int 			TUNE_ARCH,
	typename 		ProblemInstance,
	ProblemSize 	PROBLEM_SIZE,
	int 			BITS_REMAINING,
	int 			CURRENT_BIT,
	int 			CURRENT_PASS>
struct TunedPassPolicy;


/**
 * SM20
 */
template <typename ProblemInstance, ProblemSize PROBLEM_SIZE, int BITS_REMAINING, int CURRENT_BIT, int CURRENT_PASS>
struct TunedPassPolicy<200, ProblemInstance, PROBLEM_SIZE, BITS_REMAINING, CURRENT_BIT, CURRENT_PASS>
{
	enum {
		TUNE_ARCH			= 200,
		KEYS_ONLY 			= util::Equals<typename ProblemInstance::ValueType, util::NullType>::VALUE,
		PREFERRED_BITS		= 5,
		RADIX_BITS 			= CUB_MIN(BITS_REMAINING, (BITS_REMAINING % PREFERRED_BITS == 0) ? PREFERRED_BITS : PREFERRED_BITS - 1),
		SMEM_8BYTE_BANKS	= false,
		EARLY_EXIT 			= false,
		LARGE_DATA			= (sizeof(typename ProblemInstance::KeyType) > 4) || (sizeof(typename ProblemInstance::ValueType) > 4),
	};

	// Dispatch policy
	typedef DispatchPolicy <
		TUNE_ARCH,									// TUNE_ARCH
		RADIX_BITS,									// RADIX_BITS
		false, 										// UNIFORM_SMEM_ALLOCATION
		true> 										// UNIFORM_GRID_SIZE
			DispatchPolicy;

	// Upsweep kernel policy
	typedef upsweep::KernelPolicy<
		RADIX_BITS,									// RADIX_BITS
		CURRENT_BIT,								// CURRENT_BIT
		CURRENT_PASS,								// CURRENT_PASS
		8,											// MIN_CTA_OCCUPANCY
		7,											// LOG_THREADS
		(!LARGE_DATA ? 2 : 1),						// LOG_LOAD_VEC_SIZE
		1,											// LOG_LOADS_PER_TILE
		b40c::util::io::ld::NONE,					// LOAD_MODIFIER
		b40c::util::io::st::NONE,					// STORE_MODIFIER
		SMEM_8BYTE_BANKS,							// SMEM_8BYTE_BANKS
		EARLY_EXIT>									// EARLY_EXIT
			UpsweepPolicy;

	// Spine-scan kernel policy
	typedef spine::KernelPolicy<
		8,											// LOG_THREADS
		2,											// LOG_LOAD_VEC_SIZE
		2,											// LOG_LOADS_PER_TILE
		b40c::util::io::ld::NONE,					// LOAD_MODIFIER
		b40c::util::io::st::NONE>					// STORE_MODIFIER
			SpinePolicy;

	// Downsweep kernel policy
	typedef downsweep::KernelPolicy<
			RADIX_BITS,								// RADIX_BITS
			CURRENT_BIT,							// CURRENT_BIT
			CURRENT_PASS,							// CURRENT_PASS
			(KEYS_ONLY && !LARGE_DATA ? 4 : 2),		// MIN_CTA_OCCUPANCY
			(KEYS_ONLY && !LARGE_DATA ? 7 : 8),		// LOG_THREADS
			(!LARGE_DATA ? 4 : 3),					// LOG_ELEMENTS_PER_TILE
			b40c::util::io::ld::NONE,				// LOAD_MODIFIER
			b40c::util::io::st::NONE,				// STORE_MODIFIER
			downsweep::SCATTER_TWO_PHASE,			// SCATTER_STRATEGY
			SMEM_8BYTE_BANKS,						// SMEM_8BYTE_BANKS
			EARLY_EXIT>								// EARLY_EXIT
				DownsweepPolicy;
};



}// namespace radix_sort
}// namespace b40c

