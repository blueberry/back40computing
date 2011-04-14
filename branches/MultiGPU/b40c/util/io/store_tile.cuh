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
 * Kernel utilities for storing tiles of data through global memory
 * with cache modifiers
 ******************************************************************************/

#pragma once

#include <b40c/util/cuda_properties.cuh>
#include <b40c/util/vector_types.cuh>
#include <b40c/util/io/modified_store.cuh>

namespace b40c {
namespace util {
namespace io {


/**
 * Store a tile of items
 */
template <
	int LOG_STORES_PER_TILE, 
	int LOG_STORE_VEC_SIZE,
	int ACTIVE_THREADS,
	st::CacheModifier CACHE_MODIFIER,
	bool UNGUARDED_IO> 
		struct StoreTile;

/**
 * Store of a tile of items using unguarded stores 
 */
template <
	int LOG_STORES_PER_TILE, 
	int LOG_STORE_VEC_SIZE,
	int ACTIVE_THREADS,
	st::CacheModifier CACHE_MODIFIER>
struct StoreTile <
	LOG_STORES_PER_TILE,
	LOG_STORE_VEC_SIZE,
	ACTIVE_THREADS,
	CACHE_MODIFIER,
	true>
{
	static const int STORES_PER_TILE = 1 << LOG_STORES_PER_TILE;
	static const int STORE_VEC_SIZE = 1 << LOG_STORE_VEC_SIZE;
	
	// Iterate over stores
	template <int STORE, int __dummy = 0>
	struct Iterate 
	{
		template <typename VectorType>
		static __device__ __forceinline__ void Invoke(
			VectorType vectors[], 
			VectorType *d_in_vectors) 
		{
			ModifiedStore<CACHE_MODIFIER>::St(
				vectors[STORE], d_in_vectors + threadIdx.x);
			
			Iterate<STORE + 1>::Invoke(vectors, d_in_vectors + ACTIVE_THREADS);
		}
	};

	// Terminate
	template <int __dummy>
	struct Iterate<STORES_PER_TILE, __dummy> 
	{
		template <typename VectorType>
		static __device__ __forceinline__ void Invoke(
			VectorType vectors[], VectorType *d_in_vectors) {} 
	};
	
	// Interface
	template <typename T, typename SizeT>
	static __device__ __forceinline__ void Invoke(
		T data[][STORE_VEC_SIZE],
		T *d_out,
		SizeT cta_offset,
		const SizeT &out_of_bounds = 0)
	{
		// Aliased vector type
		typedef typename VecType<T, STORE_VEC_SIZE>::Type VectorType;

		// Use an aliased pointer to keys array to perform built-in vector stores
		VectorType *vectors = (VectorType *) data;
		VectorType *d_in_vectors = (VectorType *) (d_out + cta_offset);
		
		Iterate<0>::Invoke(vectors, d_in_vectors);
	}
};
	

/**
 * Store of a tile of items using guarded stores 
 */
template <
	int LOG_STORES_PER_TILE, 
	int LOG_STORE_VEC_SIZE,
	int ACTIVE_THREADS,
	st::CacheModifier CACHE_MODIFIER>
struct StoreTile <
	LOG_STORES_PER_TILE,
	LOG_STORE_VEC_SIZE,
	ACTIVE_THREADS,
	CACHE_MODIFIER,
	false>
{
	static const int STORES_PER_TILE = 1 << LOG_STORES_PER_TILE;
	static const int STORE_VEC_SIZE = 1 << LOG_STORE_VEC_SIZE;

	// Iterate over vec-elements
	template <int STORE, int VEC>
	struct Iterate
	{
		template <typename T, typename SizeT>
		static __device__ __forceinline__ void Invoke(
			T data[][STORE_VEC_SIZE],
			T *d_out,
			SizeT cta_offset,
			SizeT out_of_bounds)
		{
			SizeT thread_offset = cta_offset + VEC;

			if (thread_offset < out_of_bounds) {
				ModifiedStore<CACHE_MODIFIER>::St(data[STORE][VEC], d_out + thread_offset);
			}
			Iterate<STORE, VEC + 1>::Invoke(data, d_out, cta_offset, out_of_bounds);
		}
	};

	// Iterate over stores
	template <int STORE>
	struct Iterate<STORE, STORE_VEC_SIZE>
	{
		template <typename T, typename SizeT>
		static __device__ __forceinline__ void Invoke(
			T data[][STORE_VEC_SIZE],
			T *d_out,
			SizeT cta_offset,
			SizeT out_of_bounds)
		{
			Iterate<STORE + 1, 0>::Invoke(
				data, d_out, cta_offset + (ACTIVE_THREADS << LOG_STORE_VEC_SIZE), out_of_bounds);
		}
	};
	
	// Terminate
	template <int VEC>
	struct Iterate<STORES_PER_TILE, VEC>
	{
		template <typename T, typename SizeT>
		static __device__ __forceinline__ void Invoke(
			T data[][STORE_VEC_SIZE],
			T *d_out,
			SizeT cta_offset,
			SizeT out_of_bounds) {}
	};

	// Interface
	template <typename T, typename SizeT>
	static __device__ __forceinline__ void Invoke(
		T data[][STORE_VEC_SIZE],
		T *d_out,
		SizeT cta_offset,
		SizeT out_of_bounds)
	{
		Iterate<0, 0>::template Invoke<T, SizeT>(
			data,
			d_out,
			cta_offset + (threadIdx.x << LOG_STORE_VEC_SIZE),
			out_of_bounds);
	} 
};



} // namespace io
} // namespace util
} // namespace b40c

