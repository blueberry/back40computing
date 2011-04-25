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
 * Kernel utilities for loading tiles of data through global memory
 * with cache modifiers
 ******************************************************************************/

#pragma once

#include <b40c/util/operators.cuh>
#include <b40c/util/vector_types.cuh>
#include <b40c/util/io/modified_load.cuh>

namespace b40c {
namespace util {
namespace io {


/**
 * Load a tile of items and initialize discontinuity flags
 */
template <
	int LOG_LOADS_PER_TILE, 									// Number of vector loads (log)
	int LOG_LOAD_VEC_SIZE,										// Number of items per vector load (log)
	int ACTIVE_THREADS,											// Active threads that will be loading
	ld::CacheModifier CACHE_MODIFIER,							// Cache modifier (e.g., CA/CG/CS/NONE/etc.)
	bool UNGUARDED_IO>											// Whether or not bounds-checking is to be done
struct LoadTile;

/**
 * Load of a tile of items using unguarded loads and initialize discontinuity flags
 */
template <
	int LOG_LOADS_PER_TILE,
	int LOG_LOAD_VEC_SIZE,
	int ACTIVE_THREADS,
	ld::CacheModifier CACHE_MODIFIER>
struct LoadTile <
	LOG_LOADS_PER_TILE,
	LOG_LOAD_VEC_SIZE,
	ACTIVE_THREADS,
	CACHE_MODIFIER,
	true>
{
	static const int LOADS_PER_TILE = 1 << LOG_LOADS_PER_TILE;
	static const int LOAD_VEC_SIZE = 1 << LOG_LOAD_VEC_SIZE;
	
	//---------------------------------------------------------------------
	// Helper Structures
	//---------------------------------------------------------------------

	template <bool FIRST_TILE, int LOAD, int VEC, int dummy = 0> struct Iterate;

	/**
	 * First vec element of a vector-load
	 */
	template <bool FIRST_TILE, int LOAD, int dummy>
	struct Iterate<FIRST_TILE, LOAD, 0, dummy>
	{
		// Regular
		template <typename T, void Transform(T&, bool), typename VectorType>
		static __device__ __forceinline__ void Invoke(
			T data[][LOAD_VEC_SIZE],
			VectorType vectors[],
			VectorType *d_in_vectors)
		{
			ModifiedLoad<CACHE_MODIFIER>::Ld(vectors[LOAD], d_in_vectors);
			Transform(data[LOAD][0], true);		// Apply transform function with in_bounds = true

			Iterate<FIRST_TILE, LOAD, 1>::template Invoke<T, Transform, VectorType>(
				data, vectors, d_in_vectors);
		}

		// With discontinuity flags
		template <typename T, typename Flag, typename VectorType>
		static __device__ __forceinline__ void Invoke(
			T data[][LOAD_VEC_SIZE],
			Flag flags[][LOAD_VEC_SIZE],
			VectorType vectors[], 
			VectorType *d_in_vectors) 
		{
			// Load the vector
			ModifiedLoad<CACHE_MODIFIER>::Ld(vectors[LOAD], d_in_vectors);

			// Process first vec element
			T current = data[LOAD][0];

			// Get the previous vector element
			T *d_ptr = (T*) d_in_vectors;
			T previous;
			ModifiedLoad<CACHE_MODIFIER>::Ld(previous, d_ptr - 1);

			// Initialize discontinuity flag
			flags[LOAD][0] = (previous != current);

			Iterate<FIRST_TILE, LOAD, 1>::Invoke(data, flags, vectors, d_in_vectors);
		}
	};

	/**
	 * First vec element of first load of first tile
	 */
	template <int dummy>
	struct Iterate<true, 0, 0, dummy>
	{
		// With discontinuity flags
		template <typename T, typename Flag, typename VectorType>
		static __device__ __forceinline__ void Invoke(
			T data[][LOAD_VEC_SIZE],
			Flag flags[][LOAD_VEC_SIZE],
			VectorType vectors[],
			VectorType *d_in_vectors)
		{
			// Load the vector
			ModifiedLoad<CACHE_MODIFIER>::Ld(vectors[0], d_in_vectors);

			// Process first vec element
			T current = data[0][0];

			if ((blockIdx.x == 0) && (threadIdx.x == 0)) {

				// First load of first tile of first CTA: discontinuity
				flags[0][0] = 1;

			} else {

				// Get the previous vector element
				T *d_ptr = (T*) d_in_vectors;
				T previous;
				ModifiedLoad<CACHE_MODIFIER>::Ld(previous, d_ptr - 1);

				// Initialize discontinuity flag
				flags[0][0] = (previous != current);
			}

			Iterate<true, 0, 1>::Invoke(data, flags, vectors, d_in_vectors);
		}
	};

	/**
	 * Next vec element of a vector-load
	 */
	template <bool FIRST_TILE, int LOAD, int VEC, int dummy>
	struct Iterate
	{
		// Regular
		template <typename T, void Transform(T&, bool), typename VectorType>
		static __device__ __forceinline__ void Invoke(
			T data[][LOAD_VEC_SIZE],
			VectorType vectors[],
			VectorType *d_in_vectors)
		{
			Transform(data[LOAD][VEC], true);	// Apply transform function with in_bounds = true
			Iterate<FIRST_TILE, LOAD, VEC + 1>::template Invoke<T, Transform, VectorType>(
				data, vectors, d_in_vectors);
		}

		// With discontinuity flags
		template <typename T, typename Flag, typename VectorType>
		static __device__ __forceinline__ void Invoke(
			T data[][LOAD_VEC_SIZE],
			Flag flags[][LOAD_VEC_SIZE],
			VectorType vectors[], 
			VectorType *d_in_vectors)
		{
			T current = data[LOAD][VEC];
			T previous = data[LOAD][VEC - 1];
			flags[LOAD][VEC] = (previous != current);

			Iterate<FIRST_TILE, LOAD, VEC + 1>::Invoke(data, flags, vectors, d_in_vectors);
		}
	};

	/**
	 * Next load
	 */
	template <bool FIRST_TILE, int LOAD, int dummy>
	struct Iterate<FIRST_TILE, LOAD, LOAD_VEC_SIZE, dummy>
	{
		// Regular
		template <typename T, void Transform(T&, bool), typename VectorType>
		static __device__ __forceinline__ void Invoke(
			T data[][LOAD_VEC_SIZE],
			VectorType vectors[],
			VectorType *d_in_vectors)
		{
			Iterate<FIRST_TILE, LOAD + 1, 0>::template Invoke<T, Transform, VectorType>(
				data, vectors, d_in_vectors + ACTIVE_THREADS);
		}

		// With discontinuity flags
		template <typename T, typename Flag, typename VectorType>
		static __device__ __forceinline__ void Invoke(
			T data[][LOAD_VEC_SIZE],
			Flag flags[][LOAD_VEC_SIZE],
			VectorType vectors[], 
			VectorType *d_in_vectors)
		{
			Iterate<FIRST_TILE, LOAD + 1, 0>::Invoke(
				data, flags, vectors, d_in_vectors + ACTIVE_THREADS);
		}
	};
	
	/**
	 * Terminate
	 */
	template <bool FIRST_TILE, int dummy>
	struct Iterate<FIRST_TILE, LOADS_PER_TILE, 0, dummy>
	{
		// Regular
		template <typename T, void Transform(T&, bool), typename VectorType>
		static __device__ __forceinline__ void Invoke(
			T data[][LOAD_VEC_SIZE],
			VectorType vectors[],
			VectorType *d_in_vectors) {}

		// With discontinuity flags
		template <typename T, typename Flag, typename VectorType>
		static __device__ __forceinline__ void Invoke(
			T data[][LOAD_VEC_SIZE],
			Flag flags[][LOAD_VEC_SIZE],
			VectorType vectors[], 
			VectorType *d_in_vectors) {} 
	};


	//---------------------------------------------------------------------
	// Interface
	//---------------------------------------------------------------------

	/**
	 * Load a tile with transform (unguarded)
	 */
	template <
		typename T,
		void Transform(T&, bool), 						// Assignment function to transform the loaded value (can be used assign default values for items deemed not in bounds)
		typename SizeT>
	static __device__ __forceinline__ void Invoke(
		T data[][LOAD_VEC_SIZE],
		T *d_in,
		const SizeT &guarded_elements)
	{
		// Use an aliased pointer to keys array to perform built-in vector loads
		typedef typename VecType<T, LOAD_VEC_SIZE>::Type VectorType;

		VectorType *vectors = (VectorType *) data;
		VectorType *d_in_vectors = (VectorType *) (d_in + (threadIdx.x << LOG_LOAD_VEC_SIZE));

		Iterate<false, 0,0>::template Invoke<T, Transform, VectorType>(
			data, vectors, d_in_vectors);
	}


	/**
	 * Load a tile with transform (unguarded)
	 */
	template <
		typename T,
		void Transform(T&, bool)> 						// Assignment function to transform the loaded value (can be used assign default values for items deemed not in bounds)
	static __device__ __forceinline__ void Invoke(
		T data[][LOAD_VEC_SIZE],
		T *d_in)
	{
		Invoke<T, Transform>(data, d_in, 0);
	}


	/**
	 * Load a tile (unguarded)
	 */
	template <typename T, typename SizeT>
	static __device__ __forceinline__ void Invoke(
		T data[][LOAD_VEC_SIZE],
		T *d_in,
		const SizeT &guarded_elements)
	{
		Invoke<T, NopLdTransform<T>, SizeT>(data, d_in, guarded_elements);
	}


	/**
	 * Load a tile (unguarded)
	 */
	template <typename T>
	static __device__ __forceinline__ void Invoke(
		T data[][LOAD_VEC_SIZE],
		T *d_in)
	{
		Invoke<T, NopLdTransform<T> >(data, d_in, 0);
	}


	/**
	 * Load a tile and initialize discontinuity flags when values change
	 * between consecutive elements (unguarded)
	 */
	template <
		bool FIRST_TILE,											// Whether or not this is the first tile loaded by the CTA
		typename T,
		typename Flag,												// Discontinuity flag type
		typename SizeT>												// Integer type for indexing into global arrays
	static __device__ __forceinline__ void Invoke(
		T data[][LOAD_VEC_SIZE],
		Flag flags[][LOAD_VEC_SIZE],
		T *d_in,
		const SizeT &guarded_elements)
	{
		// Use an aliased pointer to keys array to perform built-in vector loads
		typedef typename VecType<T, LOAD_VEC_SIZE>::Type VectorType;

		VectorType *vectors = (VectorType *) data;
		VectorType *d_in_vectors = (VectorType *) (d_in + (threadIdx.x << LOG_LOAD_VEC_SIZE));
		Iterate<FIRST_TILE, 0,0>::Invoke(data, flags, vectors, d_in_vectors);
	} 


	/**
	 * Load a tile and initialize discontinuity flags when values change
	 * between consecutive elements (unguarded)
	 */
	template <
		bool FIRST_TILE,											// Whether or not this is the first tile loaded by the CTA
		typename T,
		typename Flag>												// Integer type for indexing into global arrays
	static __device__ __forceinline__ void Invoke(
		T data[][LOAD_VEC_SIZE],
		Flag flags[][LOAD_VEC_SIZE],
		T *d_in)
	{
		Invoke<FIRST_TILE, T, Flag>(data, flags, d_in, 0);
	}
};





/**
 * Load of a tile of items using guarded loads and initialize discontinuity flags
 */
template <
	int LOG_LOADS_PER_TILE, 
	int LOG_LOAD_VEC_SIZE,
	int ACTIVE_THREADS,
	ld::CacheModifier CACHE_MODIFIER>
struct LoadTile <
	LOG_LOADS_PER_TILE,
	LOG_LOAD_VEC_SIZE,
	ACTIVE_THREADS,
	CACHE_MODIFIER,
	false>
{
	static const int LOADS_PER_TILE = 1 << LOG_LOADS_PER_TILE;
	static const int LOAD_VEC_SIZE = 1 << LOG_LOAD_VEC_SIZE;

	//---------------------------------------------------------------------
	// Helper Structures
	//---------------------------------------------------------------------

	template <bool FIRST_TILE, int LOAD, int VEC, int __dummy = 0> struct Iterate;

	/**
	 * First vec element of a vector-load
	 */
	template <bool FIRST_TILE, int LOAD, int dummy>
	struct Iterate<FIRST_TILE, LOAD, 0, dummy>
	{
		// Regular
		template <typename T, void Transform(T&, bool), typename SizeT>
		static __device__ __forceinline__ void Invoke(
			T data[][LOAD_VEC_SIZE],
			T *d_in,
			const SizeT &guarded_elements)
		{
			SizeT thread_offset = (threadIdx.x << LOG_LOAD_VEC_SIZE) + (LOAD * ACTIVE_THREADS * LOAD_VEC_SIZE) + 0;

			if (thread_offset < guarded_elements) {
				ModifiedLoad<CACHE_MODIFIER>::Ld(data[LOAD][0], d_in + thread_offset);
				Transform(data[LOAD][0], true);
			} else {
				Transform(data[LOAD][0], false);	// !in_bounds
			}

			Iterate<FIRST_TILE, LOAD, 1>::template Invoke<T, Transform, SizeT>(
				data, d_in, guarded_elements);
		}

		// With discontinuity flags
		template <typename T, typename Flag, typename SizeT>
		static __device__ __forceinline__ void Invoke(
			T data[][LOAD_VEC_SIZE],
			Flag flags[][LOAD_VEC_SIZE],
			T *d_in,
			const SizeT &guarded_elements)
		{
			SizeT thread_offset = (threadIdx.x << LOG_LOAD_VEC_SIZE) + (LOAD * ACTIVE_THREADS * LOAD_VEC_SIZE) + 0;

			if (thread_offset < guarded_elements) {
				ModifiedLoad<CACHE_MODIFIER>::Ld(data[LOAD][0], d_in + thread_offset);

				// Get the previous vector element (which is in range b/c this one is in range)
				T previous;
				ModifiedLoad<CACHE_MODIFIER>::Ld(previous, d_in + thread_offset - 1);
				flags[LOAD][0] = (previous != data[LOAD][0]);

			} else {
				flags[LOAD][0] = 0;
			}

			Iterate<FIRST_TILE, LOAD, 1>::Invoke(
				data, flags, d_in, guarded_elements);
		}
	};

	/**
	 * First vec element of first load of first tile
	 */
	template <int dummy>
	struct Iterate<true, 0, 0, dummy>
	{
		// With discontinuity flags
		template <typename T, typename Flag, typename SizeT>
		static __device__ __forceinline__ void Invoke(
			T data[][LOAD_VEC_SIZE],
			Flag flags[][LOAD_VEC_SIZE],
			T *d_in,
			const SizeT &guarded_elements)
		{
			SizeT thread_offset = (threadIdx.x << LOG_LOAD_VEC_SIZE) + (0 * ACTIVE_THREADS * LOAD_VEC_SIZE) + 0;

			if (thread_offset < guarded_elements) {
				ModifiedLoad<CACHE_MODIFIER>::Ld(data[0][0], d_in + thread_offset);

				if ((blockIdx.x == 0) && (threadIdx.x == 0)) {

					// First load of first tile of first CTA: discontinuity
					flags[0][0] = 1;

				} else {

					// Get the previous vector element (which is in range b/c this one is in range)
					T previous;
					ModifiedLoad<CACHE_MODIFIER>::Ld(previous, d_in + thread_offset - 1);
					flags[0][0] = (previous != data[0][0]);
				}

			} else {
				flags[0][0] = 0;
			}

			Iterate<true, 0, 1>::Invoke(
				data, flags, d_in, guarded_elements);
		}
	};

	/**
	 * Next vec element of a vector-load
	 */
	template <bool FIRST_TILE, int LOAD, int VEC, int __dummy>
	struct Iterate 
	{
		// Regular
		template <typename T, void Transform(T&, bool), typename SizeT>
		static __device__ __forceinline__ void Invoke(
			T data[][LOAD_VEC_SIZE],
			T *d_in,
			const SizeT &guarded_elements)
		{
			SizeT thread_offset = (threadIdx.x << LOG_LOAD_VEC_SIZE) + (LOAD * ACTIVE_THREADS * LOAD_VEC_SIZE) + VEC;

			if (thread_offset < guarded_elements) {
				ModifiedLoad<CACHE_MODIFIER>::Ld(data[LOAD][VEC], d_in + thread_offset);
				Transform(data[LOAD][VEC], true);
			} else {
				Transform(data[LOAD][VEC], false);	// !in_bounds
			}

			Iterate<FIRST_TILE, LOAD, VEC + 1>::template Invoke<T, Transform, SizeT>(
				data, d_in, guarded_elements);
		}

		// With discontinuity flags
		template <typename T, typename Flag, typename SizeT>
		static __device__ __forceinline__ void Invoke(
			T data[][LOAD_VEC_SIZE],
			Flag flags[][LOAD_VEC_SIZE],
			T *d_in,
			const SizeT &guarded_elements)
		{
			SizeT thread_offset = (threadIdx.x << LOG_LOAD_VEC_SIZE) + (LOAD * ACTIVE_THREADS * LOAD_VEC_SIZE) + VEC;

			if (thread_offset < guarded_elements) {
				ModifiedLoad<CACHE_MODIFIER>::Ld(data[LOAD][VEC], d_in + thread_offset);

				T previous = data[LOAD][VEC - 1];
				T current = data[LOAD][VEC];
				flags[LOAD][VEC] = (previous != current);

			} else {

				flags[LOAD][VEC] = 0;
			}
			
			Iterate<FIRST_TILE, LOAD, VEC + 1>::Invoke(
				data, flags, d_in, guarded_elements);
		}
	};

	/**
	 * Next load
	 */
	template <bool FIRST_TILE, int LOAD, int __dummy>
	struct Iterate<FIRST_TILE, LOAD, LOAD_VEC_SIZE, __dummy>
	{
		// Regular
		template <typename T, void Transform(T&, bool), typename SizeT>
		static __device__ __forceinline__ void Invoke(
			T data[][LOAD_VEC_SIZE],
			T *d_in,
			const SizeT &guarded_elements)
		{
			Iterate<FIRST_TILE, LOAD + 1, 0>::template Invoke<T, Transform, SizeT>(
				data,
				d_in,
				guarded_elements);
		}

		// With discontinuity flags
		template <typename T, typename Flag, typename SizeT>
		static __device__ __forceinline__ void Invoke(
			T data[][LOAD_VEC_SIZE],
			Flag flags[][LOAD_VEC_SIZE],
			T *d_in,
			const SizeT &guarded_elements)
		{
			Iterate<FIRST_TILE, LOAD + 1, 0>::Invoke(
				data,
				flags,
				d_in,
				guarded_elements);
		}
	};
	
	/**
	 * Terminate
	 */
	template <bool FIRST_TILE, int __dummy>
	struct Iterate<FIRST_TILE, LOADS_PER_TILE, 0, __dummy>
	{
		// Regular
		template <typename T, void Transform(T&, bool), typename SizeT>
		static __device__ __forceinline__ void Invoke(
			T data[][LOAD_VEC_SIZE],
			T *d_in,
			const SizeT &guarded_elements) {}

		// With discontinuity flags
		template <typename T, typename Flag, typename SizeT>
		static __device__ __forceinline__ void Invoke(
			T data[][LOAD_VEC_SIZE],
			Flag flags[][LOAD_VEC_SIZE],
			T *d_in,
			const SizeT &guarded_elements) {}
	};

	//---------------------------------------------------------------------
	// Interface
	//---------------------------------------------------------------------

	/**
	 * Load a tile with transform (guarded)
	 */
	template <
		typename T,
		void Transform(T&, bool), 						// Assignment function to transform the loaded value (can be used assign default values for items deemed not in bounds)
		typename SizeT>
	static __device__ __forceinline__ void Invoke(
		T data[][LOAD_VEC_SIZE],
		T *d_in,
		const SizeT &guarded_elements)
	{
		Iterate<false, 0, 0>::template Invoke<T, Transform, SizeT>(
			data, d_in, guarded_elements);
	}

	/**
	 * Load a tile (guarded)
	 */
	template <typename T, typename SizeT>
	static __device__ __forceinline__ void Invoke(
		T data[][LOAD_VEC_SIZE],
		T *d_in,
		const SizeT &guarded_elements)
	{
		Invoke<T, NopLdTransform<T>, SizeT>(data, d_in, guarded_elements);
	}

	/**
	 * Load a tile and initialize discontinuity flags when values change
	 * between consecutive elements (guarded)
	 */
	template <
		bool FIRST_TILE,								// Whether or not this is the first tile loaded by the CTA
		typename T,										// Tile type
		typename Flag,									// Discontinuity flag type
		typename SizeT>									// Integer type for indexing into global arrays
	static __device__ __forceinline__ void Invoke(
		T data[][LOAD_VEC_SIZE],
		Flag flags[][LOAD_VEC_SIZE],
		T *d_in,
		const SizeT &guarded_elements)
	{
		Iterate<FIRST_TILE, 0, 0>::Invoke(data, flags, d_in, guarded_elements);
	} 
};


} // namespace io
} // namespace util
} // namespace b40c

