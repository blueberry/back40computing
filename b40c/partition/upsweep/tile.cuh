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
 * Abstract tile-processing functionality for partitioning upsweep reduction
 * kernels
 ******************************************************************************/

#pragma once

namespace b40c {
namespace partition {
namespace upsweep {


/**
 * Tile
 *
 * Abstract class
 */
template <
	int LOG_LOADS_PER_TILE,
	int LOG_LOAD_VEC_SIZE,
	typename KernelPolicy,
	typename DerivedTile>
struct Tile
{
	//---------------------------------------------------------------------
	// Typedefs and Constants
	//---------------------------------------------------------------------

	typedef typename KernelPolicy::KeyType 			KeyType;
	typedef typename KernelPolicy::SizeT			SizeT;
	typedef DerivedTile 							Dispatch;

	enum {
		LOADS_PER_TILE 		= 1 << LOG_LOADS_PER_TILE,
		LOAD_VEC_SIZE 		= 1 << LOG_LOAD_VEC_SIZE
	};

	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Dequeued vertex ids
	KeyType 	keys[LOADS_PER_TILE][LOAD_VEC_SIZE];


	//---------------------------------------------------------------------
	// Abstract Interface
	//---------------------------------------------------------------------

	/**
	 * Returns the bin into which the specified key is to be placed
	 *
	 * To be overloaded.
	 */
	template <typename Cta>
	__device__ __forceinline__ int DecodeBin(KeyType key, Cta *cta);


	/**
	 * Returns whether or not the key is valid.
	 *
	 * To be overloaded.
	 */
	template <int LOAD, int VEC>
	__device__ __forceinline__ bool IsValid();


	/**
	 * Loads keys into the tile
	 *
	 * To be overloaded.
	 */
	template <typename Cta>
	__device__ __forceinline__ void LoadKeys(Cta *cta, SizeT cta_offset);


	/**
	 * Stores keys from the tile (if necessary)
	 *
	 * To be overloaded.
	 */
	template <typename Cta>
	__device__ __forceinline__ void StoreKeys(Cta *cta, SizeT cta_offset);


	//---------------------------------------------------------------------
	// Iteration Structures
	//---------------------------------------------------------------------

	/**
	 * Iterate next vector element
	 */
	template <int LOAD, int VEC, int dummy = 0>
	struct Iterate
	{
		// Bucket
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void Bucket(Cta *cta, Tile *tile)
		{
			if (tile->template IsValid<LOAD, VEC>()) {

				const KeyType COUNTER_BYTE_MASK = (KernelPolicy::LOG_BINS < 2) ? 0x1 : 0x3;


				if (__CUB_CUDA_ARCH__ >= 200) {
/*
					int lane = util::BFE(
						tile->keys[LOAD][VEC],
						KernelPolicy::CURRENT_BIT + 2,
						KernelPolicy::LOG_BINS - 2);

					int bucket = util::BFE(
						tile->keys[LOAD][VEC],
						KernelPolicy::CURRENT_BIT,
						KernelPolicy::LOG_BINS);

					int counter = cta->smem_storage.composite_counters.words[lane][threadIdx.x];
					int shift;
					util::BFI(shift, 0, bucket, 3, 2);
					counter = util::SHL_ADD(1, shift, counter);
					cta->smem_storage.composite_counters.words[lane][threadIdx.x] = counter;
*/
/*
					int bucket = util::BFE(
						tile->keys[LOAD][VEC],
						KernelPolicy::CURRENT_BIT,
						KernelPolicy::LOG_BINS);

					int index = (KernelPolicy::THREADS / 4) * (~3 & bucket);
					int counter = *(cta->smem_storage.composite_counters.direct + threadIdx.x + index);
					int shift;
					util::BFI(shift, 0, bucket, 3, 2);
					counter = util::SHL_ADD(1, shift, counter);

					*(cta->smem_storage.composite_counters.direct + threadIdx.x + index) = counter;
*/

					int sub_counter = util::BFE(
						tile->keys[LOAD][VEC],
						KernelPolicy::CURRENT_BIT,
						2);

					int lane = (KernelPolicy::LOG_BINS <= 2) ?
						0 :
						util::BFE(
							tile->keys[LOAD][VEC],
							KernelPolicy::CURRENT_BIT + 2,
							KernelPolicy::LOG_BINS - 2);

					// Increment sub-field in composite counter
					cta->smem_storage.composite_counters.counters[lane][threadIdx.x][sub_counter]++;

				} else {

					// Decode the bin for this key
					int bin = tile->DecodeBin(tile->keys[LOAD][VEC], cta);

					// Decode composite-counter lane and sub-counter from bin
					int lane = bin >> 2;										// extract composite counter lane
					int sub_counter = bin & COUNTER_BYTE_MASK;					// extract 8-bit counter offset

					// Increment sub-field in composite counter
					cta->smem_storage.composite_counters.words[lane][threadIdx.x] += (1 << (sub_counter << 0x3));
				}
			}

			Iterate<LOAD, VEC + 1>::Bucket(cta, tile);
		}
	};


	/**
	 * Iterate next load
	 */
	template <int LOAD, int dummy>
	struct Iterate<LOAD, LOAD_VEC_SIZE, dummy>
	{
		// Bucket
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void Bucket(Cta *cta, Tile *tile)
		{
			Iterate<LOAD + 1, 0>::Bucket(cta, tile);
		}
	};


	/**
	 * Terminate iteration
	 */
	template <int dummy>
	struct Iterate<LOADS_PER_TILE, 0, dummy>
	{
		// Bucket
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void Bucket(Cta *cta, Tile *tile) {}
	};


	//---------------------------------------------------------------------
	// Interface
	//---------------------------------------------------------------------

	/**
	 * Decode keys in this tile and updates the cta's corresponding composite counters
	 */
	template <typename Cta>
	__device__ __forceinline__ void Bucket(Cta *cta)
	{
		Iterate<0, 0>::Bucket(cta, (Dispatch *) this);
	}
};



} // namespace upsweep
} // namespace partition
} // namespace b40c
