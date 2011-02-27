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
 * Reduction kernel
 ******************************************************************************/

#pragma once

#include "b40c_cuda_properties.cuh"
#include "b40c_kernel_utils.cuh"
#include "b40c_kernel_data_movement.cuh"

namespace b40c {
namespace reduction {


/******************************************************************************
 *  Granularity Configuration
 ******************************************************************************/

/**
 * Type of reduction problem
 */
template <
	typename _T,
	typename _SizeT,
	_T _BinaryOp(const _T&, const _T&),
	_T _Identity()>
struct ReductionProblem
{
	typedef _T								T;
	typedef _SizeT							SizeT;

	static __host__ __device__ __forceinline__ T BinaryOp(const T &a, const T &b)
	{
		return _BinaryOp(a, b);
	}

	static __host__ __device__ __forceinline__ T Identity()
	{
		return _Identity();
	}
};



/**
 * Reduction kernel granularity configuration meta-type.  Parameterizations of this
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
	// Problem type parameters
	typename ReductionProblem,

	// Tunable parameters
	int _CTA_OCCUPANCY,
	int _LOG_THREADS,
	int _LOG_LOAD_VEC_SIZE,
	int _LOG_LOADS_PER_TILE,
	int _LOG_RAKING_THREADS,
	CacheModifier _READ_MODIFIER,
	CacheModifier _WRITE_MODIFIER,
	bool _WORK_STEALING,
	int _LOG_SCHEDULE_GRANULARITY>

struct ReductionKernelConfig
{
	typedef ReductionProblem						Problem;

	static const int CTA_OCCUPANCY  				= _CTA_OCCUPANCY;
	static const CacheModifier READ_MODIFIER 		= _READ_MODIFIER;
	static const CacheModifier WRITE_MODIFIER 		= _WRITE_MODIFIER;
	static const bool WORK_STEALING					= _WORK_STEALING;

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


	// We reduce the elements in registers, and then place that partial
	// reduction into smem rows for further reduction

	// We need a two-level grid if (LOG_RAKING_THREADS > LOG_WARP_THREADS).  If so, we
	// back up the primary raking warps with a single warp of raking-threads.
	//
	// (N.B.: Typically two-level grids are a losing performance proposition)
	static const bool TWO_LEVEL_GRID				= (LOG_RAKING_THREADS > B40C_LOG_WARP_THREADS(__B40C_CUDA_ARCH__));

	// Primary smem SRTS grid type
	typedef SrtsGrid<
		typename Problem::T,					// Partial type
		LOG_THREADS,							// Depositing threads (the CTA size)
		0,										// 1 lane (CTA threads only make one deposit)
		LOG_RAKING_THREADS> PrimaryGrid;		// Raking threads

	// Secondary smem SRTS grid type
	typedef SrtsGrid<
		typename Problem::T,					// Partial type
		LOG_RAKING_THREADS,						// Depositing threads (the primary raking threads)
		0,										// 1 lane (the primary raking threads only make one deposit)
		B40C_LOG_WARP_THREADS(__B40C_CUDA_ARCH__)> SecondaryGrid;	// Raking threads (1 warp)


	static const int SMEM_QUADS	= (TWO_LEVEL_GRID) ?
		PrimaryGrid::SMEM_QUADS + SecondaryGrid::SMEM_QUADS :	// two-level smem SRTS
		PrimaryGrid::SMEM_QUADS;								// one-level smem SRTS


	static __device__ __forceinline__ void LoadTransform(typename Problem::T &val, bool in_bounds)
	{
		// Assigns identity value to out-of-bounds loads
		if (!in_bounds) val = Problem::Identity();
	}

};


/******************************************************************************
 * Reduction kernel subroutines
 ******************************************************************************/

/**
 * Collective reduction across all threads: One-level raking grid
 *
 * Used to collectively reduce each thread's aggregate after striding through
 * the input.  For when we have one warp or smaller of raking threads.
 */
template <typename Config, bool TWO_LEVEL_GRID>
struct CollectiveReduction
{
	typedef typename Config::Problem::T T;
	typedef typename Config::Problem::SizeT SizeT;

	static __device__ __forceinline__ void Invoke(
		T carry,
		T *d_out)
	{
		// Shared memory pool
		__shared__ uint4 smem_pool[Config::SMEM_QUADS];

		// Determine the deposit and raking pointers for SRTS grid
		T *primary_grid 			= reinterpret_cast<T*>(smem_pool);
		T *primary_base_partial 	= Config::PrimaryGrid::BasePartial(primary_grid);

		// Place partials in primary smem grid
		primary_base_partial[0] = carry;

		__syncthreads();

		// Primary rake and reduce (guaranteed one warp or fewer raking threads)
		if (threadIdx.x < Config::PrimaryGrid::RAKING_THREADS) {

			// Raking reduction
			T *primary_raking_seg = Config::PrimaryGrid::RakingSegment(primary_grid);
			T raking_partial = SerialReduce<T, Config::PrimaryGrid::PARTIALS_PER_SEG, Config::Problem::BinaryOp>::Invoke(primary_raking_seg);

			// WarpReduce
			T total = WarpReduce<T, Config::PrimaryGrid::LOG_RAKING_THREADS, Config::Problem::BinaryOp>::Invoke(
				raking_partial, primary_grid);

			// Write output
			if (threadIdx.x == 0) {
				ModifiedStore<T, Config::WRITE_MODIFIER>::St(total, d_out, 0);
			}
		}
	}
};


/**
 * Collective reduction across all threads: Two-level raking grid
 *
 * Used to collectively reduce each thread's aggregate after striding through
 * the input.  For when we have more than one warp of raking threads.
 */
template <typename Config>
struct CollectiveReduction <Config, true>
{
	typedef typename Config::Problem::T T;
	typedef typename Config::Problem::SizeT SizeT;

	static __device__ __forceinline__ void Invoke(
		T carry,
		T *d_out)
	{
		// Shared memory pool
		__shared__ uint4 smem_pool[Config::SMEM_QUADS];

		// Determine the deposit and raking pointers for SRTS grids
		T *primary_grid 			= reinterpret_cast<T*>(smem_pool);
		T *secondary_grid 			= reinterpret_cast<T*>(
										smem_pool + Config::PrimaryGrid::SMEM_QUADS);		// Offset by the primary grid
		T *primary_base_partial 	= Config::PrimaryGrid::BasePartial(primary_grid);

		// Place partials in primary smem grid
		primary_base_partial[0] = carry;

		__syncthreads();

		// Primary rake and reduce
		if (threadIdx.x < Config::PrimaryGrid::RAKING_THREADS) {

			// Raking reduction in primary grid
			T *primary_raking_seg = Config::PrimaryGrid::RakingSegment(primary_grid);
			T raking_partial = SerialReduce<T, Config::PrimaryGrid::PARTIALS_PER_SEG, Config::Problem::BinaryOp>::Invoke(primary_raking_seg);

			// Place raked partial in secondary grid
			T *secondary_base_partial = Config::SecondaryGrid::BasePartial(secondary_grid);
			secondary_base_partial[0] = raking_partial;
		}

		__syncthreads();

		// Secondary rake and reduce (guaranteed one warp or fewer raking threads)
		if (threadIdx.x < Config::SecondaryGrid::RAKING_THREADS) {

			// Raking reduction in secondary grid
			T *secondary_raking_seg = Config::SecondaryGrid::RakingSegment(secondary_grid);
			T raking_partial = SerialReduce<T, Config::SecondaryGrid::PARTIALS_PER_SEG, Config::Problem::BinaryOp>::Invoke(secondary_raking_seg);

			// WarpReduce
			T total = WarpReduce<T, Config::SecondaryGrid::LOG_RAKING_THREADS, Config::Problem::BinaryOp>::Invoke(
				raking_partial, secondary_grid);

			// Write output
			if (threadIdx.x == 0) {
				d_out[0] = total;
			}
		}
	}
};


/**
 * Process a single tile
 *
 * Each thread reduces only the strided values it loads.
 */
template <typename Config, bool UNGUARDED_IO>
__device__ __forceinline__ void ProcessTile(
	typename Config::Problem::T * __restrict d_in,
	typename Config::Problem::SizeT 	cta_offset,
	typename Config::Problem::SizeT 	out_of_bounds,
	typename Config::Problem::T &carry)
{
	typedef typename Config::Problem::T T;
	typedef typename Config::Problem::SizeT SizeT;

	T data[Config::LOADS_PER_TILE][Config::LOAD_VEC_SIZE];

	// Load tile
	LoadTile<
		T,
		SizeT,
		Config::LOG_LOADS_PER_TILE,
		Config::LOG_LOAD_VEC_SIZE,
		Config::THREADS,
		Config::READ_MODIFIER,
		UNGUARDED_IO,
		Config::LoadTransform>::Invoke(data, d_in, cta_offset, out_of_bounds);

	// Reduce into carry
	carry = Config::Problem::BinaryOp(carry,
		SerialReduce<T, Config::TILE_ELEMENTS_PER_THREAD, Config::Problem::BinaryOp>::Invoke(
			reinterpret_cast<T*>(data)));

	__syncthreads();
}


/**
 * Upsweep reduction pass (non-workstealing)
 */
template <typename Config, bool WORK_STEALING>
struct UpsweepReductionPass
{
	static __device__ __forceinline__ void Invoke(
		typename Config::Problem::T 			* __restrict &d_in,
		typename Config::Problem::T 			* __restrict &d_out,
		typename Config::Problem::SizeT 		* __restrict &d_work_progress,
		CtaWorkDistribution<typename Config::Problem::SizeT> &work_decomposition,
		int &progress_selector)
	{
		typedef typename Config::Problem::SizeT SizeT;

		typename Config::Problem::T carry = Config::Problem::Identity();		// The value we will accumulate

		// Determine our threadblock's work range
		SizeT cta_offset;			// Offset at which this CTA begins processing
		SizeT cta_elements;			// Total number of elements for this CTA to process
		SizeT guarded_offset; 		// Offset of final, partially-full tile (requires guarded loads)
		SizeT guarded_elements;		// Number of elements in partially-full tile

		work_decomposition.GetCtaWorkLimits<Config::LOG_TILE_ELEMENTS, Config::LOG_SCHEDULE_GRANULARITY>(
			cta_offset, cta_elements, guarded_offset, guarded_elements);

		SizeT out_of_bounds = cta_offset + cta_elements;

		// Process full tiles of tile_elements
		while (cta_offset < guarded_offset) {

			ProcessTile<Config, true>(d_in, cta_offset, out_of_bounds, carry);
			cta_offset += Config::TILE_ELEMENTS;
		}

		// Clean up last partial tile with guarded-io
		if (guarded_elements) {
			ProcessTile<Config, false>(d_in, cta_offset, out_of_bounds, carry);
		}

		// Collectively reduce accumulated carry from each thread
		CollectiveReduction<Config, Config::TWO_LEVEL_GRID>::Invoke(carry, d_out);
	}
};



/**
 * Upsweep reduction pass (workstealing)
 */
template <typename Config>
struct UpsweepReductionPass <Config, true>
{
	static __device__ __forceinline__ void Invoke(
		typename Config::Problem::T 			* __restrict &d_in,
		typename Config::Problem::T 			* __restrict &d_out,
		typename Config::Problem::SizeT 		* __restrict &d_work_progress,
		CtaWorkDistribution<typename Config::Problem::SizeT> &work_decomposition,
		int &progress_selector)
	{
		typedef typename Config::Problem::SizeT SizeT;
		typedef typename Config::Problem::T T;

		// The offset at which this CTA performs tile processing
		__shared__ SizeT cta_offset;

		// The value we will accumulate
		T carry = Config::Problem::Identity();

		// First CTA resets the work progress for the next pass
		if ((blockIdx.x == 0) && (threadIdx.x == 0)) {
			d_work_progress[progress_selector ^ 1] = 0;
		}

		// Steal full-tiles of work, incrementing progress counter
		SizeT unguarded_elements = work_decomposition.num_elements & (~(Config::TILE_ELEMENTS - 1));
		while (true) {

			// Thread zero atomically steals work from the progress counter
			if (threadIdx.x == 0) {
				cta_offset = atomicAdd(&d_work_progress[progress_selector], Config::TILE_ELEMENTS);
			}

			__syncthreads();		// Protect cta_offset

			if (cta_offset >= unguarded_elements) {
				// All done
				break;
			}

			ProcessTile<Config, true>(d_in, cta_offset, unguarded_elements, carry);
		}

		// Last CTA does any extra, guarded work
		if (blockIdx.x == gridDim.x - 1) {
			ProcessTile<Config, false>(d_in, unguarded_elements, work_decomposition.num_elements, carry);
		}

		// Collectively reduce accumulated carry from each thread
		CollectiveReduction<Config, Config::TWO_LEVEL_GRID>::Invoke(carry, d_out);
	}
};


/**
 * Spine reduction pass
 */
template <typename Config>
__device__ __forceinline__ void SpineReductionPass(
	typename Config::Problem::T 		* __restrict 	d_spine,
	typename Config::Problem::T 		* __restrict 	d_out,
	typename Config::Problem::SizeT 					spine_elements)
{
	typedef typename Config::Problem::SizeT SizeT;
	typedef typename Config::Problem::T T;

	// Exit if we're not the first CTA
	if (blockIdx.x > 0) return;

	// The value we will accumulate
	T carry = Config::Problem::Identity();

	// Number of elements in (the last) partially-full tile (requires guarded loads)
	SizeT cta_guarded_elements = spine_elements & (Config::TILE_ELEMENTS - 1);

	// Offset of final, partially-full tile (requires guarded loads)
	SizeT cta_guarded_offset = spine_elements - cta_guarded_elements;

	// Process full tiles of tile_elements
	SizeT cta_offset = 0;
	while (cta_offset < cta_guarded_offset) {

		ProcessTile<Config, true>(d_spine, cta_offset, cta_guarded_offset, carry);
		cta_offset += Config::TILE_ELEMENTS;
	}

	// Clean up last partial tile with guarded-io
	if (cta_guarded_elements) {
		ProcessTile<Config, false>(d_spine, cta_offset, spine_elements, carry);
	}

	// Collectively reduce accumulated carry from each thread
	CollectiveReduction<Config, Config::TWO_LEVEL_GRID>::Invoke(carry, d_out);
}



/******************************************************************************
 * Upsweep Reduction Kernel Entrypoint
 ******************************************************************************/

/**
 * Upsweep reduction kernel entry point
 */
template <typename Config>
__launch_bounds__ (Config::THREADS, Config::CTA_OCCUPANCY)
__global__
void UpsweepReductionKernel(
	typename Config::Problem::T 			* __restrict d_in,
	typename Config::Problem::T 			* __restrict d_spine,
	typename Config::Problem::SizeT 		* __restrict d_work_progress,
	CtaWorkDistribution<typename Config::Problem::SizeT> work_decomposition,
	int progress_selector)
{
	typename Config::Problem::T *d_spine_partial = d_spine + blockIdx.x;

	UpsweepReductionPass<Config, Config::WORK_STEALING>::Invoke(
		d_in,
		d_spine_partial,
		d_work_progress,
		work_decomposition,
		progress_selector);
}


/**
 * Wrapper stub for arbitrary types to quiet the linker
 */
template <typename Config>
void __wrapper__device_stub_UpsweepReductionKernel(
	typename Config::Problem::T 			* __restrict &,
	typename Config::Problem::T 			* __restrict &,
	typename Config::Problem::SizeT 		* __restrict &,
	CtaWorkDistribution<typename Config::Problem::SizeT> &,
	int &) {}



/******************************************************************************
 * Spine Reduction Kernel Entrypoint
 ******************************************************************************/

/**
 * Spine reduction kernel entry point
 */
template <typename Config>
__launch_bounds__ (Config::THREADS, Config::CTA_OCCUPANCY)
__global__ 
void SpineReductionKernel(
	typename Config::Problem::T 		* __restrict 	d_spine,
	typename Config::Problem::T 		* __restrict 	d_out,
	typename Config::Problem::SizeT 					spine_elements)
{
	SpineReductionPass<Config>(d_spine, d_out, spine_elements);
}


/**
 * Wrapper stub for arbitrary types to quiet the linker
 */
template <typename Config>
void __wrapper__device_stub_SpineReductionKernel(
		typename Config::Problem::T * __restrict &,
		typename Config::Problem::T * __restrict &,
		typename Config::Problem::SizeT&) {}




} // namespace reduction
} // namespace b40c

