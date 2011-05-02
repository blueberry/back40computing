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
 * AUTHORS' REQUEST: 
 * 
 * 		If you use|reference|benchmark this code, please cite our Technical 
 * 		Report (http://www.cs.virginia.edu/~dgm4d/papers/RadixSortTR.pdf):
 * 
 *		@TechReport{ Merrill:Sorting:2010,
 *        	author = "Duane Merrill and Andrew Grimshaw",
 *        	title = "Revisiting Sorting for GPGPU Stream Architectures",
 *        	year = "2010",
 *        	institution = "University of Virginia, Department of Computer Science",
 *        	address = "Charlottesville, VA, USA",
 *        	number = "CS2010-03"
 *		}
 * 
 * For more information, see our Google Code project site: 
 * http://code.google.com/p/back40computing/
 * 
 * Thanks!
 * 
 ******************************************************************************/


/******************************************************************************
 * Upsweep CTA processing abstraction
 ******************************************************************************/

#pragma once

#include <b40c/util/cuda_properties.cuh>
#include <b40c/util/basic_utils.cuh>
#include <b40c/util/cta_work_distribution.cuh>
#include <b40c/util/io/modified_load.cuh>
#include <b40c/util/io/modified_store.cuh>
#include <b40c/util/io/load_tile.cuh>
#include <b40c/util/device_intrinsics.cuh>
#include <b40c/util/reduction/serial_reduce.cuh>

#include <b40c/radix_sort/distribution/upsweep/lanes.cuh>
#include <b40c/radix_sort/distribution/upsweep/composites.cuh>
#include <b40c/radix_sort/distribution/upsweep/tile.cuh>
#include <b40c/radix_sort/sort_utils.cuh>

namespace b40c {
namespace radix_sort {
namespace distribution {
namespace upsweep {



/**
 * Derivation of KernelConfig that encapsulates tile-processing routines
 */
template <
	typename KernelConfig,
	typename SmemStorage,
	typename Derived = util::NullType>
struct Cta
{
	//---------------------------------------------------------------------
	// Typedefs and Constants
	//---------------------------------------------------------------------

	typedef typename KernelConfig::KeyType 	KeyType;
	typedef typename KernelConfig::SizeT 	SizeT;
	typedef typename util::If<util::Equals<Derived, util::NullType>::VALUE, Cta, Derived>::Type Dispatch;

	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Shared storage for this CTA
	typename KernelConfig::SmemStorage 	&smem_storage;

	// Each thread is responsible for aggregating an unencoded segment of composite counters
	SizeT 								local_counts[KernelConfig::LANES_PER_WARP][4];

	// Input and output device pointers
	KeyType								*d_in_keys;
	SizeT								*d_spine;

	int 								warp_id;
	int 								warp_idx;


	//---------------------------------------------------------------------
	// Helper Structures
	//---------------------------------------------------------------------

	/**
	 * Unrolled tile processing
	 */
	struct UnrollTiles
	{
		// Recurse over counts
		template <int UNROLL_COUNT, int __dummy = 0>
		struct Iterate
		{
			static const int HALF = UNROLL_COUNT / 2;

			template <typename Cta>
			static __device__ __forceinline__ void ProcessTiles(
				Cta *cta, SizeT cta_offset)
			{
				Iterate<HALF>::ProcessTiles(cta, cta_offset);
				Iterate<HALF>::ProcessTiles(cta, cta_offset + (KernelConfig::TILE_ELEMENTS * HALF));
			}
		};

		// Terminate (process one tile)
		template <int __dummy>
		struct Iterate<1, __dummy>
		{
			template <typename Cta>
			static __device__ __forceinline__ void ProcessTiles(
				Cta *cta, SizeT cta_offset)
			{
				cta->ProcessFullTile(cta_offset);
			}
		};
	};


	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------

	/**
	 * Constructor
	 */
	__device__ __forceinline__ Cta(
		SmemStorage 	&smem_storage,
		KeyType 		*d_in_keys,
		SizeT 			*d_spine) :
			smem_storage(smem_storage),
			d_in_keys(d_in_keys),
			d_spine(d_spine),
			warp_id(threadIdx.x >> B40C_LOG_WARP_THREADS(__B40C_CUDA_ARCH__)),
			warp_idx(util::LaneId()) {}


	/**
	 * Decodes a key and places increments the corresponding composite
	 * counter in the corresponding lane
	 */
	__device__ __forceinline__ void Bucket(KeyType key) 
	{
		// Pre-process key with bit-twiddling functor if necessary
		KernelConfig::PreprocessTraits::Preprocess(key, true);

		// Extract lane containing corresponding composite counter
		int lane;
		radix_sort::ExtractKeyBits<
			KeyType,
			KernelConfig::CURRENT_BIT + 2,
			KernelConfig::RADIX_BITS - 2>::Extract(lane, key);

		if (__B40C_CUDA_ARCH__ >= 200) {

			// GF100+ has special bit-extraction instructions (instead of shift+mask)
			int quad_byte;
			if (KernelConfig::RADIX_BITS < 2) {
				radix_sort::ExtractKeyBits<KeyType, KernelConfig::CURRENT_BIT, 1>::Extract(quad_byte, key);
			} else {
				radix_sort::ExtractKeyBits<KeyType, KernelConfig::CURRENT_BIT, 2>::Extract(quad_byte, key);
			}

			// Increment sub-field in composite counter
			smem_storage.composite_counters.counters[lane][threadIdx.x][quad_byte]++;

		} else {

			// GT200 can save an instruction because it can source an operand
			// directly from smem
			const int BYTE_ENCODE_SHIFT 		= 0x3;
			const KeyType QUAD_MASK 			= (KernelConfig::RADIX_BITS < 2) ? 0x1 : 0x3;

			int quad_shift = util::MagnitudeShift<KeyType, BYTE_ENCODE_SHIFT - KernelConfig::CURRENT_BIT>(
				key & (QUAD_MASK << KernelConfig::CURRENT_BIT));

			// Increment sub-field in composite counter
			smem_storage.composite_counters.words[lane][threadIdx.x] += (1 << quad_shift);;
		}
	}


	/**
	 * Processes a single, full tile
	 */
	__device__ __forceinline__ void ProcessFullTile(
		SizeT cta_offset)
	{
		Tile<
			KeyType,
			KernelConfig::LOG_LOADS_PER_TILE,
			KernelConfig::LOG_LOAD_VEC_SIZE> tile;

		util::io::LoadTile<
			KernelConfig::LOG_LOADS_PER_TILE,
			KernelConfig::LOG_LOAD_VEC_SIZE,
			KernelConfig::THREADS,
			KernelConfig::READ_MODIFIER,
			true>::Invoke(tile.keys, d_in_keys + cta_offset);

		if (KernelConfig::LOADS_PER_TILE > 1) __syncthreads();		// Prevents bucketing from being hoisted up into loads

		// Bucket tile of keys
		tile.Bucket((Dispatch*) this);
	}


	/**
	 * Processes a single load (may have some threads masked off)
	 */
	__device__ __forceinline__ void ProcessSingleLoad(
		SizeT cta_offset)
	{
		KeyType key;
		util::io::ModifiedLoad<KernelConfig::READ_MODIFIER>::Ld(
			key, d_in_keys + cta_offset);
		Bucket(key);
	}


	/**
	 * Processes all tiles
	 */
	__device__ __forceinline__ void ProcessTiles(
		SizeT cta_offset,
		SizeT cta_out_of_bounds)
	{
		Dispatch *dispatch = (Dispatch*) this;

		Composites<KernelConfig>::ResetCounters(dispatch);
		Lanes<KernelConfig>::ResetCompositeCounters(dispatch);

		__syncthreads();

		// Unroll batches of full tiles
		const int UNROLLED_ELEMENTS = KernelConfig::UNROLL_COUNT * KernelConfig::TILE_ELEMENTS;
		while (cta_offset < cta_out_of_bounds - UNROLLED_ELEMENTS) {

			UnrollTiles::template Iterate<KernelConfig::UNROLL_COUNT>::ProcessTiles(
				dispatch,
				cta_offset);
			cta_offset += UNROLLED_ELEMENTS;

			__syncthreads();

			// Aggregate back into local_count registers to prevent overflow
			Composites<KernelConfig>::ReduceComposites(dispatch);

			__syncthreads();

			// Reset composite counters in lanes
			Lanes<KernelConfig>::ResetCompositeCounters(dispatch);
		}

		// Unroll single full tiles
		while (cta_offset < cta_out_of_bounds - KernelConfig::TILE_ELEMENTS) {

			UnrollTiles::template Iterate<1>::ProcessTiles(
				dispatch,
				cta_offset);
			cta_offset += KernelConfig::TILE_ELEMENTS;
		}

		// Process partial tile if necessary using single loads
		cta_offset += threadIdx.x;
		while (cta_offset < cta_out_of_bounds) {

			dispatch->ProcessSingleLoad(cta_offset);
			cta_offset += KernelConfig::THREADS;
		}

		__syncthreads();

		// Aggregate back into local_count registers
		Composites<KernelConfig>::ReduceComposites(dispatch);

		__syncthreads();

		//Final raking reduction of counts by digit, output to spine.

		Composites<KernelConfig>::PlacePartials(dispatch);

		__syncthreads();

		// Rake-reduce and write out the digit_count reductions
		if (threadIdx.x < KernelConfig::RADIX_DIGITS) {

			SizeT digit_count = util::reduction::SerialReduce<KernelConfig::AGGREGATED_PARTIALS_PER_ROW>::Invoke(
				smem_storage.aggregate[threadIdx.x]);

			int spine_digit_offset = util::FastMul(gridDim.x, threadIdx.x) + blockIdx.x;

			util::io::ModifiedStore<KernelConfig::WRITE_MODIFIER>::St(
					digit_count, d_spine + spine_digit_offset);
		}
	}
};



} // namespace upsweep
} // namespace distribution
} // namespace radix_sort
} // namespace b40c

