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
 * Thanks!
 * 
 ******************************************************************************/


/******************************************************************************
 * Downsweep tile processing abstraction
 ******************************************************************************/

#pragma once

#include <b40c/util/io/load_tile.cuh>
#include <b40c/util/io/scatter_tile.cuh>

#include <b40c/partition/downsweep/tile.cuh>

#include <b40c/radix_sort/sort_utils.cuh>

namespace b40c {
namespace graph {
namespace bfs {
namespace partition_compact {
namespace downsweep {


/**
 * Tile
 *
 * Derives from partition::downsweep::Tile
 */
template <typename KernelPolicy>
struct Tile :
	partition::downsweep::Tile<
		KernelPolicy,
		Tile<KernelPolicy> >						// This class
{
	//---------------------------------------------------------------------
	// Typedefs and Constants
	//---------------------------------------------------------------------

	typedef typename KernelPolicy::KeyType 					KeyType;
	typedef typename KernelPolicy::ValidFlag				ValidFlag;
	typedef typename KernelPolicy::SizeT 					SizeT;


	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	ValidFlag 	flags[Tile::CYCLES_PER_TILE][Tile::LOADS_PER_CYCLE][Tile::LOAD_VEC_SIZE];


	//---------------------------------------------------------------------
	// Derived Interface
	//---------------------------------------------------------------------

	/**
	 * Returns the bin into which the specified key is to be placed.
	 */
	__device__ __forceinline__ int DecodeBin(KeyType key)
	{
		int bin;
		radix_sort::ExtractKeyBits<
			KeyType,
			0,												// Extract from low order bits
			KernelPolicy::LOG_BINS>::Extract(bin, key);

		return bin;
	}


	/**
	 * Returns whether or not the key is valid.
	 */
	template <int CYCLE, int LOAD, int VEC>
	__device__ __forceinline__ bool IsValid()
	{
		return flags[CYCLE][LOAD][VEC];
	}


	/**
	 * Loads keys and flags into the tile
	 */
	template <typename Cta>
	__device__ __forceinline__ void LoadKeys(
		Cta *cta,
		SizeT cta_offset)
	{
		// Read tile of keys
		util::io::LoadTile<
			KernelPolicy::LOG_LOADS_PER_TILE,
			KernelPolicy::LOG_LOAD_VEC_SIZE,
			KernelPolicy::THREADS,
			KernelPolicy::READ_MODIFIER>::LoadValid(
					(KeyType (*)[KernelPolicy::LOAD_VEC_SIZE]) keys,
					cta->d_in_keys + cta_offset);

		// Read tile of flags
		util::io::LoadTile<
			KernelPolicy::LOG_LOADS_PER_TILE,
			KernelPolicy::LOG_LOAD_VEC_SIZE,
			KernelPolicy::THREADS,
			KernelPolicy::READ_MODIFIER>::LoadValid(
				(ValidFlag (*)[KernelPolicy::LOAD_VEC_SIZE]) flags,
				cta->d_flags_in + cta_offset);
	}

	/**
	 * Loads keys and flags into the tile
	 */
	template <typename Cta>
	__device__ __forceinline__ void LoadKeys(
		Cta *cta,
		SizeT cta_offset,
		const SizeT &guarded_elements)
	{
		// Read tile of keys, use -1 if key is out-of-bounds
		util::io::LoadTile<
			KernelPolicy::LOG_LOADS_PER_TILE,
			KernelPolicy::LOG_LOAD_VEC_SIZE,
			KernelPolicy::THREADS,
			KernelPolicy::READ_MODIFIER>::LoadValid(
					(KeyType (*)[KernelPolicy::LOAD_VEC_SIZE]) keys,
					(KeyType) -1,
					cta->d_in_keys + cta_offset,
					guarded_elements);

		// Read tile of flags, use 0 if flag is out-of-bounds
		util::io::LoadTile<
			KernelPolicy::LOG_LOADS_PER_TILE,
			KernelPolicy::LOG_LOAD_VEC_SIZE,
			KernelPolicy::THREADS,
			KernelPolicy::READ_MODIFIER>::LoadValid(
				(ValidFlag (*)[KernelPolicy::LOAD_VEC_SIZE]) flags,
				(ValidFlag) 0,
				cta->d_flags_in + cta_offset,
				guarded_elements);
	}


	/**
	 * Scatter keys from the tile
	 */
	template <typename Cta>
	__device__ __forceinline__ void ScatterKeys(Cta *cta)
	{
		// Scatter only the compacted keys to global bin partitions
		util::io::ScatterTile<
			KernelPolicy::TILE_ELEMENTS_PER_THREAD,
			KernelPolicy::THREADS,
			KernelPolicy::WRITE_MODIFIER>::Scatter(
				cta->d_out_keys,
				(KeyType *) keys,
				scatter_offsets,
				cta->smem_storage.bin_warpscan[1][KernelPolicy::BINS - 1]);		// num compacted
	}


	/**
	 * Scatter keys from the tile
	 */
	template <typename Cta>
	__device__ __forceinline__ void ScatterKeys(
		Cta *cta,
		const SizeT &guarded_elements)
	{
		ScatterKeys(cta);
	}


	/**
	 * Scatter values from the tile
	 *
	 * To be overloaded.
	 */
	template <typename Cta>
	__device__ __forceinline__ void ScatterValues(Cta *cta)
	{
		// Scatter values to global bin partitions
		util::io::ScatterTile<
			KernelPolicy::TILE_ELEMENTS_PER_THREAD,
			KernelPolicy::THREADS,
			KernelPolicy::WRITE_MODIFIER>::Scatter(
				cta->d_out_values,
				values,
				scatter_offsets,
				cta->smem_storage.bin_warpscan[1][KernelPolicy::BINS - 1]);		// num compacted
	}


	/**
	 * Scatter values from the tile
	 *
	 * To be overloaded.
	 */
	template <typename Cta>
	__device__ __forceinline__ void ScatterValues(
		Cta *cta,
		const SizeT &guarded_elements)
	{
		ScatterValues(cta);
	}
};


} // namespace downsweep
} // namespace partition_compact
} // namespace bfs
} // namespace graph
} // namespace b40c

