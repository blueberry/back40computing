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
 ******************************************************************************/

/******************************************************************************
 * "Metatypes" for guiding scan granularity configuration
 ******************************************************************************/

#pragma once

#include <b40c/util/cuda_properties.cuh>
#include <b40c/util/srts_grid.cuh>
#include <b40c/util/data_movement_load.cuh>
#include <b40c/util/data_movement_store.cuh>
#include <b40c/reduction/kernel_config.cuh>

namespace b40c {
namespace scan {


/**
 * Type of scan problem
 */
template <
	typename T,
	typename SizeT,
	T BinaryOp(const T&, const T&),
	T Identity()>
struct ScanProblemType : reduction::ReductionProblemType<T, SizeT, BinaryOp, Identity>
{
};



/**
 * Scan kernel granularity configuration meta-type.  Parameterizations of this
 * type encapsulate our kernel-tuning parameters (i.e., they are reflected via
 * the static fields).
 *
 * Kernels can be specialized for problem-type, SM-version, etc. by parameterizing
 * them with different performance-tuned parameterizations of this type.  By
 * incorporating this type into the kernel code itself, we guide the compiler in
 * expanding/unrolling the kernel code for specific architectures and problem
 * types.
 */
template <
	// ScanProblemType type parameters
	typename ScanProblemType,

	// Tunable parameters
	int _CTA_OCCUPANCY,
	int _LOG_THREADS,
	int _LOG_LOAD_VEC_SIZE,
	int _LOG_LOADS_PER_TILE,
	int _LOG_RAKING_THREADS,
	util::ld::CacheModifier _READ_MODIFIER,
	util::st::CacheModifier _WRITE_MODIFIER,
	int _LOG_SCHEDULE_GRANULARITY>

struct ScanKernelConfig : ScanProblemType
{
	typedef typename ScanProblemType::T T;

	static const int CTA_OCCUPANCY  						= _CTA_OCCUPANCY;
	static const util::ld::CacheModifier READ_MODIFIER 		= _READ_MODIFIER;
	static const util::st::CacheModifier WRITE_MODIFIER 	= _WRITE_MODIFIER;

	static const int LOG_THREADS 					= _LOG_THREADS;
	static const int THREADS						= 1 << LOG_THREADS;

	static const int LOG_LOAD_VEC_SIZE  			= _LOG_LOAD_VEC_SIZE;
	static const int LOAD_VEC_SIZE					= 1 << LOG_LOAD_VEC_SIZE;

	static const int LOG_LOADS_PER_TILE 			= _LOG_LOADS_PER_TILE;
	static const int LOADS_PER_TILE					= 1 << LOG_LOADS_PER_TILE;

	static const int LOG_RAKING_THREADS				= _LOG_RAKING_THREADS;
	static const int RAKING_THREADS					= 1 << LOG_RAKING_THREADS;

	static const int LOG_WARPS						= LOG_THREADS - B40C_LOG_WARP_THREADS(__B40C_CUDA_ARCH__);
	static const int WARPS							= 1 << LOG_WARPS;

	static const int LOG_TILE_ELEMENTS_PER_THREAD	= LOG_LOAD_VEC_SIZE + LOG_LOADS_PER_TILE;
	static const int TILE_ELEMENTS_PER_THREAD		= 1 << LOG_TILE_ELEMENTS_PER_THREAD;

	static const int LOG_TILE_ELEMENTS 				= LOG_TILE_ELEMENTS_PER_THREAD + LOG_THREADS;
	static const int TILE_ELEMENTS					= 1 << LOG_TILE_ELEMENTS;

	static const int LOG_SCHEDULE_GRANULARITY		= _LOG_SCHEDULE_GRANULARITY;
	static const int SCHEDULE_GRANULARITY			= 1 << LOG_SCHEDULE_GRANULARITY;

	//
	// We reduce the elements in registers, and then place that partial
	// scan into smem rows for further scan
	//
	// Because all lanes are dependent, we need a two-level grid if
	// (LOG_RAKING_THREADS > LOG_WARP_THREADS) in order cooperate between
	// multiple raking warps.
	//
	// (N.B.: Typically two-level grids are a losing performance proposition)
	//

	// SRTS grid type
	typedef util::SrtsGrid<
		T,										// Partial type
		LOG_THREADS,							// Depositing threads (the CTA size)
		LOG_LOADS_PER_TILE,						// Lanes (the number of loads)
		LOG_RAKING_THREADS,						// Raking threads
		typename util::If<(LOG_RAKING_THREADS > B40C_LOG_WARP_THREADS(__B40C_CUDA_ARCH__)),	// Secondary grid type
			util::SrtsGrid<										// Yes secondary grid
				T,													// Partial type
				LOG_RAKING_THREADS,									// Depositing threads (the primary raking threads)
				0,													// 1 lane (the primary raking threads only make one deposit)
				B40C_LOG_WARP_THREADS(__B40C_CUDA_ARCH__)>,			// Raking threads (1 warp)
			util::InvalidSrtsGrid>::Type>						// No secondary grid
		SrtsGrid;

	static const int SMEM_QUADS	= SrtsGrid::SMEM_QUADS;
};


} // namespace scan
} // namespace b40c

