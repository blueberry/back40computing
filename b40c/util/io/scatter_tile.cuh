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
 * Kernel utilities for scattering data
 ******************************************************************************/

#pragma once

#include <b40c/util/io/modified_store.cuh>
#include <b40c/util/operators.cuh>

namespace b40c {
namespace util {
namespace io {




/**
 * Scatter a tile of data items using the corresponding tile of scatter_offsets
 *
 * Uses vec-1 stores.
 */
template <
	int ELEMENTS_PER_THREAD,						// Number of tile elements per thread
	int ACTIVE_THREADS,								// Active threads that will be loading
	st::CacheModifier CACHE_MODIFIER>				// Cache modifier (e.g., CA/CG/CS/NONE/etc.)
struct ScatterTile
{
	//---------------------------------------------------------------------
	// Helper Structures
	//---------------------------------------------------------------------

	// Iterate next load
	template <int LOAD, int TOTAL_LOADS>
	struct Iterate
	{
		// predicated on index
		template <bool UNGUARDED_IO, typename T, void Transform(T&), typename SizeT>
		static __device__ __forceinline__ void Invoke(
			T *dest,
			T src[ELEMENTS_PER_THREAD],
			SizeT scatter_offsets[ELEMENTS_PER_THREAD],
			const SizeT &tile_size)
		{
			if (UNGUARDED_IO || ((ACTIVE_THREADS * LOAD) + threadIdx.x < tile_size)) {

				Transform(src[LOAD]);
				ModifiedStore<CACHE_MODIFIER>::St(src[LOAD], dest + scatter_offsets[LOAD]);
			}

			Iterate<LOAD + 1, TOTAL_LOADS>::template Invoke<UNGUARDED_IO, T, Transform, SizeT>(
				dest, src, scatter_offsets, tile_size);
		}

		// predicated on valid
		template <typename T, void Transform(T&), typename Flag, typename SizeT>
		static __device__ __forceinline__ void Invoke(
			T *dest,
			T src[ELEMENTS_PER_THREAD],
			Flag valid_flags[ELEMENTS_PER_THREAD],
			SizeT scatter_offsets[ELEMENTS_PER_THREAD])
		{
			if (valid_flags[LOAD]) {

				Transform(src[LOAD]);
				ModifiedStore<CACHE_MODIFIER>::St(src[LOAD], dest + scatter_offsets[LOAD]);
			}

			Iterate<LOAD + 1, TOTAL_LOADS>::template Invoke<T, Transform, Flag, SizeT>(
				dest, src, valid_flags, scatter_offsets);
		}
	};


	// Terminate
	template <int TOTAL_LOADS>
	struct Iterate<TOTAL_LOADS, TOTAL_LOADS>
	{
		// predicated on index
		template <bool UNGUARDED_IO, typename T, void Transform(T&), typename SizeT>
		static __device__ __forceinline__ void Invoke(
			T *dest,
			T src[ELEMENTS_PER_THREAD],
			SizeT scatter_offsets[ELEMENTS_PER_THREAD],
			const SizeT &tile_size) {}

		// predicated on valid
		template <typename T, void Transform(T&), typename Flag, typename SizeT>
		static __device__ __forceinline__ void Invoke(
			T *dest,
			T src[ELEMENTS_PER_THREAD],
			Flag valid_flags[ELEMENTS_PER_THREAD],
			SizeT scatter_offsets[ELEMENTS_PER_THREAD]) {}
	};

	//---------------------------------------------------------------------
	// Interface
	//---------------------------------------------------------------------

	/**
	 * Scatter to destination with transform.  The write is
	 * predicated on the tile element's index in the CTA's tile not
	 * exceeding tile_size
	 */
	template <
		typename T,
		void Transform(T&), 							// Assignment function to transform the stored value
		typename SizeT>
	static __device__ __forceinline__ void Scatter(
		T *dest,
		T src[ELEMENTS_PER_THREAD],
		SizeT scatter_offsets[ELEMENTS_PER_THREAD],
		const SizeT &tile_size = ELEMENTS_PER_THREAD * ACTIVE_THREADS)
	{
		if (tile_size < ELEMENTS_PER_THREAD * ACTIVE_THREADS) {
			// guarded IO
			Iterate<0, ELEMENTS_PER_THREAD>::template Invoke<false, T, Transform, SizeT>(
				dest, src, scatter_offsets, tile_size);
		} else {
			// unguarded IO
			Iterate<0, ELEMENTS_PER_THREAD>::template Invoke<true, T, Transform, SizeT>(
				dest, src, scatter_offsets, tile_size);
		}
	}

	/**
	 * Scatter to destination.  The write is predicated on the tile element's
	 * index in the CTA's tile not exceeding tile_size
	 */
	template <typename T, typename SizeT>
	static __device__ __forceinline__ void Scatter(
		T *dest,
		T src[ELEMENTS_PER_THREAD],
		SizeT scatter_offsets[ELEMENTS_PER_THREAD],
		const SizeT &tile_size = ELEMENTS_PER_THREAD * ACTIVE_THREADS)
	{
		Scatter<T, NopTransform<T> >(
			dest, src, scatter_offsets, tile_size);
	}

	/**
	 * Scatter to destination with transform, predicated on the valid flag
	 */
	template <
		typename T,
		void Transform(T&), 							// Assignment function to transform the stored value
		typename Flag,
		typename SizeT>
	static __device__ __forceinline__ void Scatter(
		T *dest,
		T src[ELEMENTS_PER_THREAD],
		Flag valid_flags[ELEMENTS_PER_THREAD],
		SizeT scatter_offsets[ELEMENTS_PER_THREAD])
	{
		Iterate<0, ELEMENTS_PER_THREAD>::template Invoke<T, Transform>(
			dest, src, valid_flags, scatter_offsets);
	}

	/**
	 * Scatter to destination predicated on the valid flag
	 */
	template <typename T, typename Flag, typename SizeT>
	static __device__ __forceinline__ void Scatter(
		T *dest,
		T src[ELEMENTS_PER_THREAD],
		Flag valid_flags[ELEMENTS_PER_THREAD],
		SizeT scatter_offsets[ELEMENTS_PER_THREAD])
	{
		Scatter<T, NopTransform<T>, Flag, SizeT>(
			dest, src, valid_flags, scatter_offsets);
	}

};



} // namespace io
} // namespace util
} // namespace b40c

