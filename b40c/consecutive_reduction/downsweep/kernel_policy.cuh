/******************************************************************************
 * Copyright (c) 2010-2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2013, NVIDIA CORPORATION.  All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/******************************************************************************
 * Configuration policy for consecutive reduction downsweep scan kernels
 ******************************************************************************/

#pragma once

#include <b40c/util/cuda_properties.cuh>
#include <b40c/util/soa_tuple.cuh>
#include <b40c/util/srts_grid.cuh>
#include <b40c/util/srts_soa_details.cuh>
#include <b40c/util/io/modified_load.cuh>
#include <b40c/util/io/modified_store.cuh>

#include <b40c/consecutive_reduction/soa_scan_operator.cuh>

namespace b40c {
namespace consecutive_reduction {
namespace downsweep {


/**
 * A detailed kernel configuration policy type that specializes kernel
 * code for a specific consecutive reduction downsweep pass. It encapsulates our
 * kernel-tuning parameters (they are reflected via the static fields).
 *
 * The kernel is specialized for problem-type, SM-version, etc. by declaring
 * it with different performance-tuned parameterizations of this type.  By
 * incorporating this type into the kernel code itself, we guide the compiler in
 * expanding/unrolling the kernel code for specific architectures and problem
 * types.
 */
template <
	// ProblemType type parameters
	typename ProblemType,

	// Machine parameters
	int CUDA_ARCH,
	bool _CHECK_ALIGNMENT,

	// Tunable parameters
	int _MIN_CTA_OCCUPANCY,
	int _LOG_THREADS,
	int _LOG_LOAD_VEC_SIZE,
	int _LOG_LOADS_PER_TILE,
	int _LOG_RAKING_THREADS,
	util::io::ld::CacheModifier _READ_MODIFIER,
	util::io::st::CacheModifier _WRITE_MODIFIER,
	int _LOG_SCHEDULE_GRANULARITY,
	bool _TWO_PHASE_SCATTER,
	bool _CONSECUTIVE_SMEM_ASSIST>

struct KernelPolicy : ProblemType
{
	typedef typename ProblemType::KeyType 					KeyType;
	typedef typename ProblemType::ValueType 				ValueType;
	typedef typename ProblemType::SizeT						SizeT;
	typedef typename ProblemType::ReductionOp 				ReductionOp;

	typedef int 											LocalFlag;			// Local type for noting local discontinuities, (just needs to count up to TILE_ELEMENTS)

	typedef typename util::If<
		_TWO_PHASE_SCATTER,
		LocalFlag,
		SizeT>::Type 										RankType;			// Type for local raking prefix sum

	typedef util::Tuple<ValueType, RankType> 				TileTuple;			// Structure-of-array "slice" tuple for local raking scanning
	typedef util::Tuple<ValueType, SizeT> 					SpineSoaTuple;		// Structure-of-array "slice" tuple for spine scan
	typedef SoaScanOperator<ReductionOp, TileTuple> 		SoaScanOperator;	// Structure-of-array scan operator



	static const util::io::ld::CacheModifier READ_MODIFIER 		= _READ_MODIFIER;
	static const util::io::st::CacheModifier WRITE_MODIFIER 	= _WRITE_MODIFIER;

	enum {
		LOG_THREADS 					= _LOG_THREADS,
		THREADS							= 1 << LOG_THREADS,

		LOG_LOAD_VEC_SIZE  				= _LOG_LOAD_VEC_SIZE,
		LOAD_VEC_SIZE					= 1 << LOG_LOAD_VEC_SIZE,

		LOG_LOADS_PER_TILE 				= _LOG_LOADS_PER_TILE,
		LOADS_PER_TILE					= 1 << LOG_LOADS_PER_TILE,

		LOG_LOAD_STRIDE					= LOG_THREADS + LOG_LOAD_VEC_SIZE,
		LOAD_STRIDE						= 1 << LOG_LOAD_STRIDE,

		LOG_RAKING_THREADS				= _LOG_RAKING_THREADS,
		RAKING_THREADS					= 1 << LOG_RAKING_THREADS,

		LOG_WARPS						= LOG_THREADS - B40C_LOG_WARP_THREADS(CUDA_ARCH),
		WARPS							= 1 << LOG_WARPS,

		LOG_TILE_ELEMENTS_PER_THREAD	= LOG_LOAD_VEC_SIZE + LOG_LOADS_PER_TILE,
		TILE_ELEMENTS_PER_THREAD		= 1 << LOG_TILE_ELEMENTS_PER_THREAD,

		LOG_TILE_ELEMENTS 				= LOG_TILE_ELEMENTS_PER_THREAD + LOG_THREADS,
		TILE_ELEMENTS					= 1 << LOG_TILE_ELEMENTS,

		LOG_SCHEDULE_GRANULARITY		= _LOG_SCHEDULE_GRANULARITY,
		SCHEDULE_GRANULARITY			= 1 << LOG_SCHEDULE_GRANULARITY,

		TWO_PHASE_SCATTER				= _TWO_PHASE_SCATTER,
		CHECK_ALIGNMENT					= _CHECK_ALIGNMENT,
		CONSECUTIVE_SMEM_ASSIST			= _CONSECUTIVE_SMEM_ASSIST,
	};

	//
	// We reduce the elements in registers, and then place that partial
	// scan into smem rows for further scan
	//

	// Raking grid type for value partials
	typedef util::RakingGrid<
		CUDA_ARCH,
		ValueType,								// Partial type
		LOG_THREADS,							// Depositing threads (the CTA size)
		LOG_LOADS_PER_TILE,						// Lanes (the number of loads)
		LOG_RAKING_THREADS,						// Raking threads
		true>									// There are prefix dependences between lanes
			PartialsRakingGrid;

	// Raking grid type for ranks
	typedef util::RakingGrid<
		CUDA_ARCH,
		RankType,								// Partial type
		LOG_THREADS,							// Depositing threads (the CTA size)
		LOG_LOADS_PER_TILE,						// Lanes (the number of loads)
		LOG_RAKING_THREADS,						// Raking threads
		true>									// There are prefix dependences between lanes
			RanksRakingGrid;


	/**
	 * Shared memory structure
	 */
	struct SmemStorage
	{
		ValueType			partials_warpscan[2][B40C_WARP_THREADS(CUDA_ARCH)];
		RankType 			ranks_warpscan[2][B40C_WARP_THREADS(CUDA_ARCH)];

		union {
			struct {
				ValueType	partials_raking_elements[PartialsRakingGrid::TOTAL_RAKING_ELEMENTS];
				RankType 	ranks_raking_elements[RanksRakingGrid::TOTAL_RAKING_ELEMENTS];
			};
/*
			// Swap exchange arenas for two-phase scatter
			struct {
				KeyType 	key_exchange[TILE_ELEMENTS];
			};
			struct {
				ValueType 	value_exchange[TILE_ELEMENTS];
			};
*/
		};

		KeyType				assist_scratch[THREADS+ 1];
	};


	enum {
		THREAD_OCCUPANCY				= B40C_SM_THREADS(CUDA_ARCH) >> LOG_THREADS,
		SMEM_OCCUPANCY					= B40C_SMEM_BYTES(CUDA_ARCH) / sizeof(SmemStorage),

		MAX_CTA_OCCUPANCY  				= B40C_MIN(B40C_SM_CTAS(CUDA_ARCH), B40C_MIN(THREAD_OCCUPANCY, SMEM_OCCUPANCY)),
		MIN_CTA_OCCUPANCY 				= _MIN_CTA_OCCUPANCY,

		VALID							= (MAX_CTA_OCCUPANCY > 0)
	};


	// Tuple type of raking grid types
	typedef util::Tuple<
		PartialsRakingGrid,
		RanksRakingGrid> RakingGridTuple;


	// Operational details type for raking grid type
	typedef util::RakingSoaDetails<
		TileTuple,
		RakingGridTuple> RakingSoaDetails;

};


} // namespace downsweep
} // namespace consecutive_reduction
} // namespace b40c

