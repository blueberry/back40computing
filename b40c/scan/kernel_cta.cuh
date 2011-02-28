/******************************************************************************
 * 
 * Copyright 2010 Duane Merrill
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
 * Thanks!
 * 
 ******************************************************************************/

/******************************************************************************
 * Derivation of ScanKernelConfig that encapsulates tile-processing routines
 ******************************************************************************/

#pragma once

#include <b40c/scan/scan_utils.cuh>

namespace b40c {
namespace scan {


/******************************************************************************
 * ScanCta Declaration
 ******************************************************************************/

/**
 * Derivation of ScanKernelConfig that encapsulates tile-processing
 * routines
 */
template <
	typename ScanKernelConfig,
	bool TWO_LEVEL_GRID = ScanKernelConfig::TWO_LEVEL_GRID>
struct ScanCta;


/**
 * Derivation of ScanKernelConfig that encapsulates tile-processing
 * routines (one-level SRTS grid)
 */
template <typename ScanKernelConfig>
struct ScanCta<ScanKernelConfig, false> : ScanKernelConfig
{
	typedef typename ScanKernelConfig::T T;
	typedef typename ScanKernelConfig::SizeT SizeT;

	// The value we will accumulate
	T carry;

	T *d_in;
	T *d_out;

	uint4 *smem_quads;
	T (*warpscan)[B40C_WARP_THREADS(__B40C_CUDA_ARCH__)];

	T *primary_grid;
	T *primary_base_partial;
	T *primary_raking_seg;


	/**
	 * Reduce each load in registers and place into smem
	 */
	struct ReduceVectors;


	/**
	 * Scan each load in registers, seeding from smem partials
	 */
	struct ScanVectors;


	/**
	 * Process a single tile
	 *
	 * Each thread scans only the strided values it loads.
	 */
	template <bool UNGUARDED_IO>
	__device__ __forceinline__ void ProcessTile(
		SizeT cta_offset,
		SizeT out_of_bounds);


	/**
	 * Constructor
	 */
	__device__ __forceinline__ ScanCta(T *d_in, T *d_out);

};


/**
 * Derivation of ScanKernelConfig that encapsulates tile-processing
 * routines (two-level SRTS grid)
 */
template <typename ScanKernelConfig>
struct ScanCta<ScanKernelConfig, true> : ScanCta<ScanKernelConfig, false>
{
	typedef typename ScanKernelConfig::T T;
	typedef typename ScanKernelConfig::SizeT SizeT;

	T *secondary_grid;
	T *secondary_base_partial = 0;
	T *secondary_raking_seg = 0;

	/**
	 * Process a single tile
	 *
	 * Each thread scans only the strided values it loads.
	 */
	template <bool UNGUARDED_IO>
	__device__ __forceinline__ void ProcessTile(
		SizeT cta_offset,
		SizeT out_of_bounds);


	/**
	 * Constructor
	 */
	__device__ __forceinline__ ScanCta(T *d_in, T *d_out);
};



/******************************************************************************
 * ScanCta Implementation (one-level SRTS grid)
 ******************************************************************************/

/**
 * Constructor
 */
template <typename ScanKernelConfig>
ScanCta<ScanKernelConfig, false>::ScanCta(
	T *d_in,
	T *d_out) :
		carry(Identity()),
		d_in(d_in),
		d_out(d_out),
		primary_raking_seg(NULL)
{
	__shared__ uint4 smem_pool[SMEM_QUADS];
	__shared__ T warpscan_storage[2][B40C_WARP_THREADS(__B40C_CUDA_ARCH__)];

	smem_quads = smem_pool;
	warpscan = warpscan_storage;
	primary_grid = reinterpret_cast<T*>(smem_pool);
	primary_base_partial = PrimaryGrid::BasePartial(primary_grid);

	if (threadIdx.x < PrimaryGrid::RAKING_THREADS) {

		primary_raking_seg = PrimaryGrid::RakingSegment(primary_grid);

		// Initialize warpscan
		if (threadIdx.x < B40C_WARP_THREADS(__B40C_CUDA_ARCH__)) {
			warpscan[0][threadIdx.x] = 0;
		}
	}
}


/**
 * Process a single tile
 */
template <typename ScanKernelConfig>
template <bool UNGUARDED_IO>
void ScanCta<ScanKernelConfig, false>::ProcessTile(
	SizeT cta_offset,
	SizeT out_of_bounds)
{
	// Tile of scan elements
	T data[LOADS_PER_TILE][LOAD_VEC_SIZE];

	// Load tile
	util::LoadTile<T, SizeT, LOG_LOADS_PER_TILE, LOG_LOAD_VEC_SIZE, THREADS, READ_MODIFIER, UNGUARDED_IO>::Invoke(
		data, d_in, cta_offset, out_of_bounds);

	// Reduce in registers, place partials in smem
	ReduceVectors::Invoke(data, primary_base_partial);

	__syncthreads();

	// Primary rake and scan (guaranteed one warp or fewer raking threads)
	WarpRakeAndScan<Config::PrimaryGrid>(primary_raking_seg, warpscan, carry);

	__syncthreads();

	// Extract partials from smem, scan in registers
	ScanVectors::Invoke(data, primary_base_partial);

	// Store tile
	util::StoreTile<T, SizeT, LOG_LOADS_PER_TILE, LOG_LOAD_VEC_SIZE, THREADS, WRITE_MODIFIER, UNGUARDED_IO>::Invoke(
		data, d_out, cta_offset, out_of_bounds);
}


/**
 * Reduce each load in registers and place into smem
 */
template <typename ScanKernelConfig>
struct ScanCta<ScanKernelConfig, false>::ReduceVectors
{
	// Next load
	template <int LOAD, int TOTAL_LOADS>
	struct Iterate {
		static __device__ __forceinline__ void Invoke(
			T data[LOADS_PER_TILE][LOAD_VEC_SIZE],
			T *base_partial)
		{
			// Store partial reduction into SRTS grid
			base_partial[LOAD * PrimaryGrid::LANE_STRIDE] =
				SerialReduce<T, LOAD_VEC_SIZE>::Invoke(data[LOAD]);

			// Next load
			Iterate<LOAD + 1, TOTAL_LOADS>::Invoke(data, base_partial);
		}
	};

	// Terminate
	template <int TOTAL_LOADS>
	struct Iterate<TOTAL_LOADS, TOTAL_LOADS> {
		static __device__ __forceinline__ void Invoke(
			T data[LOADS_PER_TILE][LOAD_VEC_SIZE],
			T *base_partial) {}
	};

	// Interface
	static __device__ __forceinline__ void Invoke(
		T data[LOADS_PER_TILE][LOAD_VEC_SIZE],
		T *base_partial)
	{
		Iterate<0, LOADS_PER_TILE>::Invoke(data, base_partial);
	}

};


/**
 * Scan each load in registers, seeding from smem partials
 */
template <typename ScanKernelConfig>
struct ScanCta<ScanKernelConfig, false>::ScanVectors
{
	// Next load
	template <int LOAD, int TOTAL_LOADS>
	struct Iterate {
		static __device__ __forceinline__ void Invoke(
			T data[LOADS_PER_TILE][LOAD_VEC_SIZE],
			T *base_partial)
		{
			T exclusive_partial = base_partial[LOAD * PrimaryGrid::LANE_STRIDE];
			SerialScan<T, LOAD_VEC_SIZE>::Invoke(data[LOAD], exclusive_partial);

			// Next load
			Iterate<LOAD + 1, TOTAL_LOADS>::Invoke(data, base_partial);
		}
	};

	// Terminate
	template <int TOTAL_LOADS>
	struct Iterate<TOTAL_LOADS, TOTAL_LOADS> {
		static __device__ __forceinline__ void Invoke(
			T data[LOADS_PER_TILE][LOAD_VEC_SIZE],
			T *base_partial) {}
	};

	// Interface
	static __device__ __forceinline__ void Invoke(
		T data[LOADS_PER_TILE][LOAD_VEC_SIZE],
		T *base_partial)
	{
		Iterate<0, LOADS_PER_TILE>::Invoke(data, base_partial);
	}
};



/******************************************************************************
 * ScanCta Implementation (two-level SRTS grid)
 ******************************************************************************/

/**
 * Constructor
 */
template <typename ScanKernelConfig>
ScanCta<ScanKernelConfig, true>::ScanCta(
	T *d_in,
	T *d_out) :
		ScanCta<ScanKernelConfig, false>(d_in, d_out),
		secondary_base_partial(NULL),
		secondary_raking_seg(NULL)
{
	secondary_grid = reinterpret_cast<T*>(smem_quads + PrimaryGrid::SMEM_QUADS);

	if (threadIdx.x < PrimaryGrid::RAKING_THREADS) {

		secondary_base_partial = SecondaryGrid::BasePartial(secondary_grid);

		if (threadIdx.x < SecondaryGrid::RAKING_THREADS) {
			secondary_raking_seg = SecondaryGrid::RakingSegment(secondary_grid);
		}
	}
}


/**
 * Process a single tile
 */
template <typename ScanKernelConfig>
template <bool UNGUARDED_IO>
void ScanCta<ScanKernelConfig, true>::ProcessTile(
	SizeT cta_offset,
	SizeT out_of_bounds)
{
	// Tile of scan elements
	T data[LOADS_PER_TILE][LOAD_VEC_SIZE];

	// Load tile
	LoadTile<T, SizeT, LOG_LOADS_PER_TILE, LOG_LOAD_VEC_SIZE, THREADS, READ_MODIFIER, UNGUARDED_IO>::Invoke(
		data, d_data, cta_offset, out_of_bounds);

	// Reduce in registers, place partials in smem
	ReduceVectors::Invoke(data, primary_base_partial);

	__syncthreads();

	// Raking reduction in primary grid, place result partial into secondary grid
	if (threadIdx.x < PrimaryGrid::RAKING_THREADS) {
		T partial = SerialReduce<T, PrimaryGrid::PARTIALS_PER_SEG>::Invoke(primary_raking_seg);
		secondary_base_partial[0] = partial;
	}

	__syncthreads();

	// Secondary rake and scan (guaranteed one warp or fewer raking threads)
	WarpRakeAndScan<SecondaryGrid>(secondary_raking_seg, warpscan, carry);

	__syncthreads();

	// Raking scan in primary grid seeded by partial from secondary grid
	if (threadIdx.x < PrimaryGrid::RAKING_THREADS) {
		T partial = secondary_base_partial[0];
		SerialScan<T, PrimaryGrid::PARTIALS_PER_SEG>::Invoke(primary_raking_seg, partial);
	}

	__syncthreads();

	// Extract partials from smem, scan in registers
	ScanVectors::Invoke(data, primary_base_partial);

	// Store tile
	StoreTile<T, SizeT, LOG_LOADS_PER_TILE, LOG_LOAD_VEC_SIZE, THREADS, WRITE_MODIFIER, UNGUARDED_IO>::Invoke(
		data, d_data, cta_offset, out_of_bounds);
}



} // namespace scan
} // namespace b40c

