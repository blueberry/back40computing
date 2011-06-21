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
	bool CHECK_ALIGNMENT>										// Whether or not to check alignment to see if vector loads can be used
struct LoadTile
{
	enum {
		LOADS_PER_TILE 				= 1 << LOG_LOADS_PER_TILE,
		LOAD_VEC_SIZE 				= 1 << LOG_LOAD_VEC_SIZE,
		LOG_ELEMENTS_PER_THREAD		= LOG_LOADS_PER_TILE + LOG_LOAD_VEC_SIZE,
		ELEMENTS_PER_THREAD			= 1 << LOG_ELEMENTS_PER_THREAD,
		TILE_SIZE 					= ACTIVE_THREADS * ELEMENTS_PER_THREAD,
	};
	
	//---------------------------------------------------------------------
	// Iteration Structures
	//---------------------------------------------------------------------

	template <int LOAD, int VEC, int dummy = 0> struct Iterate;

	/**
	 * First vec element of a vector-load
	 */
	template <int LOAD, int dummy>
	struct Iterate<LOAD, 0, dummy>
	{
		// Vector (unguarded)
		template <typename T, void Transform(T&), typename VectorType>
		static __device__ __forceinline__ void LoadVector(
			T data[][LOAD_VEC_SIZE],
			VectorType vectors[],
			VectorType *d_in_vectors)
		{
			ModifiedLoad<CACHE_MODIFIER>::Ld(vectors[LOAD], d_in_vectors);
			Transform(data[LOAD][0]);		// Apply transform function with in_bounds = true

			Iterate<LOAD, 1>::template LoadVector<T, Transform>(
				data, vectors, d_in_vectors);
		}

		// Regular (unguarded)
		template <typename T, void Transform(T&)>
		static __device__ __forceinline__ void LoadValid(
			T data[][LOAD_VEC_SIZE],
			T *d_in)
		{
			int thread_offset = (threadIdx.x << LOG_LOAD_VEC_SIZE) + (LOAD * ACTIVE_THREADS * LOAD_VEC_SIZE) + 0;

			ModifiedLoad<CACHE_MODIFIER>::Ld(data[LOAD][0], d_in + thread_offset);
			Transform(data[LOAD][0]);

			Iterate<LOAD, 1>::template LoadValid<T, Transform>(
				data, d_in);
		}

		// Regular (guarded)
		template <typename T, void Transform(T&), typename SizeT>
		static __device__ __forceinline__ void LoadValid(
			T data[][LOAD_VEC_SIZE],
			T *d_in,
			const SizeT &guarded_elements)
		{
			SizeT thread_offset = (threadIdx.x << LOG_LOAD_VEC_SIZE) + (LOAD * ACTIVE_THREADS * LOAD_VEC_SIZE) + 0;

			if (thread_offset < guarded_elements) {
				ModifiedLoad<CACHE_MODIFIER>::Ld(data[LOAD][0], d_in + thread_offset);
				Transform(data[LOAD][0]);
			}

			Iterate<LOAD, 1>::template LoadValid<T, Transform>(
				data, d_in, guarded_elements);
		}

		// Regular (guarded with out-of-bounds default)
		template <typename T, void Transform(T&), typename SizeT>
		static __device__ __forceinline__ void LoadValid(
			T data[][LOAD_VEC_SIZE],
			const T &oob_default,
			T *d_in,
			const SizeT &guarded_elements)
		{
			SizeT thread_offset = (threadIdx.x << LOG_LOAD_VEC_SIZE) + (LOAD * ACTIVE_THREADS * LOAD_VEC_SIZE) + 0;

			if (thread_offset < guarded_elements) {
				ModifiedLoad<CACHE_MODIFIER>::Ld(data[LOAD][0], d_in + thread_offset);
				Transform(data[LOAD][0]);
			} else {
				data[LOAD][0] = oob_default;
			}

			Iterate<LOAD, 1>::template LoadValid<T, Transform>(
				data, oob_default, d_in, guarded_elements);
		}

		// Vector with discontinuity flags (unguarded)
		template <
			bool FIRST_TILE,
			typename T,
			typename Flag,
			typename VectorType,
			typename EqualityOp>
		static __device__ __forceinline__ void LoadDiscontinuity(
			T data[][LOAD_VEC_SIZE],
			Flag flags[][LOAD_VEC_SIZE],
			VectorType vectors[], 
			VectorType *d_in_vectors,
			EqualityOp equality_op)
		{
			// Load the vector
			ModifiedLoad<CACHE_MODIFIER>::Ld(vectors[LOAD], d_in_vectors);

			// Process first vec element
			T current = data[LOAD][0];

			if (FIRST_TILE && (LOAD == 0) && (blockIdx.x == 0) && (threadIdx.x == 0)) {

				// First load of first tile of first CTA: discontinuity
				flags[LOAD][0] = 1;

			} else {

				// Get the previous vector element
				T *d_ptr = (T*) d_in_vectors;
				T previous;
				ModifiedLoad<CACHE_MODIFIER>::Ld(previous, d_ptr - 1);

				// Initialize discontinuity flag
				flags[LOAD][0] = !equality_op(previous, current);
			}

			Iterate<LOAD, 1>::template LoadDiscontinuity<FIRST_TILE>(
				data, flags, vectors, d_in_vectors, equality_op);
		}

		// With discontinuity flags (unguarded)
		template <
			bool FIRST_TILE,
			typename T,
			typename Flag,
			typename EqualityOp>
		static __device__ __forceinline__ void LoadDiscontinuity(
			T data[][LOAD_VEC_SIZE],
			Flag flags[][LOAD_VEC_SIZE],
			T *d_in,
			EqualityOp equality_op)
		{
			int thread_offset = (threadIdx.x << LOG_LOAD_VEC_SIZE) + (LOAD * ACTIVE_THREADS * LOAD_VEC_SIZE) + 0;

			ModifiedLoad<CACHE_MODIFIER>::Ld(data[LOAD][0], d_in + thread_offset);

			if (FIRST_TILE && (LOAD == 0) && (blockIdx.x == 0) && (threadIdx.x == 0)) {

				// First load of first tile of first CTA: discontinuity
				flags[LOAD][0] = 1;

			} else {

				// Get the previous vector element (which is in range b/c this one is in range)
				T previous;
				ModifiedLoad<CACHE_MODIFIER>::Ld(previous, d_in + thread_offset - 1);
				flags[LOAD][0] = !equality_op(previous, data[LOAD][0]);
			}

			Iterate<LOAD, 1>::template LoadDiscontinuity<FIRST_TILE>(
				data, flags, d_in, equality_op);
		}

		// With discontinuity flags (guarded)
		template <
			bool FIRST_TILE,
			typename T,
			typename Flag,
			typename SizeT,
			typename EqualityOp>
		static __device__ __forceinline__ void LoadDiscontinuity(
			T data[][LOAD_VEC_SIZE],
			Flag flags[][LOAD_VEC_SIZE],
			T *d_in,
			const SizeT &guarded_elements,
			const Flag &oob_default_flag,
			EqualityOp equality_op)
		{
			SizeT thread_offset = (threadIdx.x << LOG_LOAD_VEC_SIZE) + (LOAD * ACTIVE_THREADS * LOAD_VEC_SIZE) + 0;

			if (thread_offset < guarded_elements) {
				ModifiedLoad<CACHE_MODIFIER>::Ld(data[LOAD][0], d_in + thread_offset);

				if (FIRST_TILE && (LOAD == 0) && (blockIdx.x == 0) && (threadIdx.x == 0)) {

					// First load of first tile of first CTA: discontinuity
					flags[LOAD][0] = 1;

				} else {

					// Get the previous vector element (which is in range b/c this one is in range)
					T previous;
					ModifiedLoad<CACHE_MODIFIER>::Ld(previous, d_in + thread_offset - 1);
					flags[LOAD][0] = !equality_op(previous, data[LOAD][0]);
				}

			} else {
				flags[LOAD][0] = oob_default_flag;
			}

			Iterate<LOAD, 1>::template LoadDiscontinuity<FIRST_TILE>(
				data, flags, d_in, guarded_elements, oob_default_flag, equality_op);
		}
	};


	/**
	 * Next vec element of a vector-load
	 */
	template <int LOAD, int VEC, int dummy>
	struct Iterate
	{
		// Vector (unguarded)
		template <typename T, void Transform(T&), typename VectorType>
		static __device__ __forceinline__ void LoadVector(
			T data[][LOAD_VEC_SIZE],
			VectorType vectors[],
			VectorType *d_in_vectors)
		{
			Transform(data[LOAD][VEC]);

			Iterate<LOAD, VEC + 1>::template LoadVector<T, Transform>(
				data, vectors, d_in_vectors);
		}

		// Regular (unguarded)
		template <typename T, void Transform(T&)>
		static __device__ __forceinline__ void LoadValid(
			T data[][LOAD_VEC_SIZE],
			T *d_in)
		{
			int thread_offset = (threadIdx.x << LOG_LOAD_VEC_SIZE) + (LOAD * ACTIVE_THREADS * LOAD_VEC_SIZE) + VEC;

			ModifiedLoad<CACHE_MODIFIER>::Ld(data[LOAD][VEC], d_in + thread_offset);
			Transform(data[LOAD][VEC]);

			Iterate<LOAD, VEC + 1>::template LoadValid<T, Transform>(
				data, d_in);
		}

		// Regular (guarded)
		template <typename T, void Transform(T&), typename SizeT>
		static __device__ __forceinline__ void LoadValid(
			T data[][LOAD_VEC_SIZE],
			T *d_in,
			const SizeT &guarded_elements)
		{
			SizeT thread_offset = (threadIdx.x << LOG_LOAD_VEC_SIZE) + (LOAD * ACTIVE_THREADS * LOAD_VEC_SIZE) + VEC;

			if (thread_offset < guarded_elements) {
				ModifiedLoad<CACHE_MODIFIER>::Ld(data[LOAD][VEC], d_in + thread_offset);
				Transform(data[LOAD][VEC]);
			}

			Iterate<LOAD, VEC + 1>::template LoadValid<T, Transform>(
				data, d_in, guarded_elements);
		}

		// Regular (guarded with out-of-bounds default)
		template <typename T, void Transform(T&), typename SizeT>
		static __device__ __forceinline__ void LoadValid(
			T data[][LOAD_VEC_SIZE],
			const T &oob_default,
			T *d_in,
			const SizeT &guarded_elements)
		{
			SizeT thread_offset = (threadIdx.x << LOG_LOAD_VEC_SIZE) + (LOAD * ACTIVE_THREADS * LOAD_VEC_SIZE) + VEC;

			if (thread_offset < guarded_elements) {
				ModifiedLoad<CACHE_MODIFIER>::Ld(data[LOAD][VEC], d_in + thread_offset);
				Transform(data[LOAD][VEC]);
			} else {
				data[LOAD][VEC] = oob_default;
			}

			Iterate<LOAD, VEC + 1>::template LoadValid<T, Transform>(
				data, oob_default, d_in, guarded_elements);
		}

		// Vector with discontinuity flags
		template <
			bool FIRST_TILE,
			typename T,
			typename Flag,
			typename VectorType,
			typename EqualityOp>
		static __device__ __forceinline__ void LoadDiscontinuity(
			T data[][LOAD_VEC_SIZE],
			Flag flags[][LOAD_VEC_SIZE],
			VectorType vectors[], 
			VectorType *d_in_vectors,
			EqualityOp equality_op)
		{
			T current = data[LOAD][VEC];
			T previous = data[LOAD][VEC - 1];
			flags[LOAD][VEC] = !equality_op(previous, current);

			Iterate<LOAD, VEC + 1>::template LoadDiscontinuity<FIRST_TILE>(
				data, flags, vectors, d_in_vectors, equality_op);
		}

		// With discontinuity flags (unguarded)
		template <
			bool FIRST_TILE,
			typename T,
			typename Flag,
			typename EqualityOp>
		static __device__ __forceinline__ void LoadDiscontinuity(
			T data[][LOAD_VEC_SIZE],
			Flag flags[][LOAD_VEC_SIZE],
			T *d_in,
			EqualityOp equality_op)
		{
			int thread_offset = (threadIdx.x << LOG_LOAD_VEC_SIZE) + (LOAD * ACTIVE_THREADS * LOAD_VEC_SIZE) + VEC;

			ModifiedLoad<CACHE_MODIFIER>::Ld(data[LOAD][VEC], d_in + thread_offset);

			T previous = data[LOAD][VEC - 1];
			T current = data[LOAD][VEC];
			flags[LOAD][VEC] = !equality_op(previous, current);

			Iterate<LOAD, VEC + 1>::template LoadDiscontinuity<FIRST_TILE>(
				data, flags, d_in, equality_op);
		}

		// With discontinuity flags (guarded)
		template <
			bool FIRST_TILE,
			typename T,
			typename Flag,
			typename SizeT,
			typename EqualityOp>
		static __device__ __forceinline__ void LoadDiscontinuity(
			T data[][LOAD_VEC_SIZE],
			Flag flags[][LOAD_VEC_SIZE],
			T *d_in,
			const SizeT &guarded_elements,
			const Flag &oob_default_flag,
			EqualityOp equality_op)
		{
			SizeT thread_offset = (threadIdx.x << LOG_LOAD_VEC_SIZE) + (LOAD * ACTIVE_THREADS * LOAD_VEC_SIZE) + VEC;

			if (thread_offset < guarded_elements) {
				ModifiedLoad<CACHE_MODIFIER>::Ld(data[LOAD][VEC], d_in + thread_offset);

				T previous = data[LOAD][VEC - 1];
				T current = data[LOAD][VEC];
				flags[LOAD][VEC] = !equality_op(previous, current);

			} else {

				flags[LOAD][VEC] = oob_default_flag;
			}

			Iterate<LOAD, VEC + 1>::template LoadDiscontinuity<FIRST_TILE>(
				data, flags, d_in, guarded_elements, oob_default_flag, equality_op);
		}
	};


	/**
	 * Next load
	 */
	template <int LOAD, int dummy>
	struct Iterate<LOAD, LOAD_VEC_SIZE, dummy>
	{
		// Vector (unguarded)
		template <typename T, void Transform(T&), typename VectorType>
		static __device__ __forceinline__ void LoadVector(
			T data[][LOAD_VEC_SIZE],
			VectorType vectors[],
			VectorType *d_in_vectors)
		{
			Iterate<LOAD + 1, 0>::template LoadVector<T, Transform>(
				data, vectors, d_in_vectors + ACTIVE_THREADS);
		}

		// Regular (unguarded)
		template <typename T, void Transform(T&)>
		static __device__ __forceinline__ void LoadValid(
			T data[][LOAD_VEC_SIZE],
			T *d_in)
		{
			Iterate<LOAD + 1, 0>::template LoadValid<T, Transform>(
				data, d_in);
		}

		// Regular (guarded)
		template <typename T, void Transform(T&), typename SizeT>
		static __device__ __forceinline__ void LoadValid(
			T data[][LOAD_VEC_SIZE],
			T *d_in,
			const SizeT &guarded_elements)
		{
			Iterate<LOAD + 1, 0>::template LoadValid<T, Transform>(
				data, d_in, guarded_elements);
		}

		// Regular (guarded with out-of-bounds default)
		template <typename T, void Transform(T&), typename SizeT>
		static __device__ __forceinline__ void LoadValid(
			T data[][LOAD_VEC_SIZE],
			const T &oob_default,
			T *d_in,
			const SizeT &guarded_elements)
		{
			Iterate<LOAD + 1, 0>::template LoadValid<T, Transform>(
				data, oob_default, d_in, guarded_elements);
		}

		// Vector with discontinuity flags (unguarded)
		template <
			bool FIRST_TILE,
			typename T,
			typename Flag,
			typename VectorType,
			typename EqualityOp>
		static __device__ __forceinline__ void LoadDiscontinuity(
			T data[][LOAD_VEC_SIZE],
			Flag flags[][LOAD_VEC_SIZE],
			VectorType vectors[], 
			VectorType *d_in_vectors,
			EqualityOp equality_op)
		{
			Iterate<LOAD + 1, 0>::template LoadDiscontinuity<FIRST_TILE>(
				data, flags, vectors, d_in_vectors + ACTIVE_THREADS, equality_op);
		}

		// With discontinuity flags (unguarded)
		template <
			bool FIRST_TILE,
			typename T,
			typename Flag,
			typename EqualityOp>
		static __device__ __forceinline__ void LoadDiscontinuity(
			T data[][LOAD_VEC_SIZE],
			Flag flags[][LOAD_VEC_SIZE],
			T *d_in,
			EqualityOp equality_op)
		{
			Iterate<LOAD + 1, 0>::template LoadDiscontinuity<FIRST_TILE>(
				data, flags, d_in, equality_op);
		}

		// With discontinuity flags (guarded)
		template <
			bool FIRST_TILE,
			typename T,
			typename Flag,
			typename SizeT,
			typename EqualityOp>
		static __device__ __forceinline__ void LoadDiscontinuity(
			T data[][LOAD_VEC_SIZE],
			Flag flags[][LOAD_VEC_SIZE],
			T *d_in,
			const SizeT &guarded_elements,
			const Flag &oob_default_flag,
			EqualityOp equality_op)
		{
			Iterate<LOAD + 1, 0>::template LoadDiscontinuity<FIRST_TILE>(
				data, flags, d_in, guarded_elements, oob_default_flag, equality_op);
		}
	};
	
	/**
	 * Terminate
	 */
	template <int dummy>
	struct Iterate<LOADS_PER_TILE, 0, dummy>
	{
		// Vector (unguarded)
		template <typename T, void Transform(T&), typename VectorType>
		static __device__ __forceinline__ void LoadVector(
			T data[][LOAD_VEC_SIZE],
			VectorType vectors[],
			VectorType *d_in_vectors) {}

		// Regular (unguarded)
		template <typename T, void Transform(T&)>
		static __device__ __forceinline__ void LoadValid(
			T data[][LOAD_VEC_SIZE],
			T *d_in) {}

		// Regular (guarded)
		template <typename T, void Transform(T&), typename SizeT>
		static __device__ __forceinline__ void LoadValid(
			T data[][LOAD_VEC_SIZE],
			T *d_in,
			const SizeT &guarded_elements) {}

		// Regular (guarded with out-of-bounds default)
		template <typename T, void Transform(T&), typename SizeT>
		static __device__ __forceinline__ void LoadValid(
			T data[][LOAD_VEC_SIZE],
			const T &oob_default,
			T *d_in,
			const SizeT &guarded_elements) {}

		// Vector with discontinuity flags (unguarded)
		template <
			bool FIRST_TILE,
			typename T,
			typename Flag,
			typename VectorType,
			typename EqualityOp>
		static __device__ __forceinline__ void LoadDiscontinuity(
			T data[][LOAD_VEC_SIZE],
			Flag flags[][LOAD_VEC_SIZE],
			VectorType vectors[], 
			VectorType *d_in_vectors,
			EqualityOp equality_op) {}

		// With discontinuity flags (unguarded)
		template <
			bool FIRST_TILE,
			typename T,
			typename Flag,
			typename EqualityOp>
		static __device__ __forceinline__ void LoadDiscontinuity(
			T data[][LOAD_VEC_SIZE],
			Flag flags[][LOAD_VEC_SIZE],
			T *d_in,
			EqualityOp equality_op) {}

		// With discontinuity flags (guarded)
		template <
			bool FIRST_TILE,
			typename T,
			typename Flag,
			typename SizeT,
			typename EqualityOp>
		static __device__ __forceinline__ void LoadDiscontinuity(
			T data[][LOAD_VEC_SIZE],
			Flag flags[][LOAD_VEC_SIZE],
			T *d_in,
			const SizeT &guarded_elements,
			const Flag &oob_default_flag,
			EqualityOp equality_op) {}
	};


	//---------------------------------------------------------------------
	// Interface
	//---------------------------------------------------------------------

	/**
	 * Load a full tile with transform
	 */
	template <
		typename T,
		void Transform(T&),
		typename SizeT>
	static __device__ __forceinline__ void LoadValid(
		T data[][LOAD_VEC_SIZE],
		T *d_in,
		SizeT cta_offset)
	{
		const size_t MASK = ((sizeof(T) * 8 * LOAD_VEC_SIZE) - 1);

		if ((CHECK_ALIGNMENT) && (LOAD_VEC_SIZE > 1) && (((size_t) d_in) & MASK)) {

			Iterate<0, 0>::template LoadValid<T, Transform>(
				data, d_in + cta_offset);

		} else {

			// Use an aliased pointer to keys array to perform built-in vector loads
			typedef typename VecType<T, LOAD_VEC_SIZE>::Type VectorType;

			VectorType *vectors = (VectorType *) data;
			VectorType *d_in_vectors = (VectorType *) (d_in + cta_offset + (threadIdx.x << LOG_LOAD_VEC_SIZE));

			Iterate<0, 0>::template LoadVector<T, Transform>(
				data, vectors, d_in_vectors);
		}
	}


	/**
	 * Load a full tile
	 */
	template <
		typename T,
		typename SizeT>
	static __device__ __forceinline__ void LoadValid(
		T data[][LOAD_VEC_SIZE],
		T *d_in,
		SizeT cta_offset)
	{
		LoadValid<T, Operators<T>::NopTransform>(data, d_in, cta_offset);
	}


	/**
	 * Load guarded_elements of a tile with transform and out-of-bounds default
	 */
	template <
		typename T,
		void Transform(T&),
		typename SizeT>
	static __device__ __forceinline__ void LoadValid(
		T data[][LOAD_VEC_SIZE],
		T *d_in,
		SizeT cta_offset,
		const SizeT &guarded_elements,
		const T &oob_default)
	{
		if (guarded_elements >= TILE_SIZE) {
			LoadValid<T, Transform>(data, d_in, cta_offset);
		} else {
			Iterate<0, 0>::template LoadValid<T, Transform>(
				data, oob_default, d_in + cta_offset, guarded_elements);
		}
	}


	/**
	 * Load guarded_elements of a tile with transform
	 */
	template <
		typename T,
		void Transform(T&),
		typename SizeT>
	static __device__ __forceinline__ void LoadValid(
		T data[][LOAD_VEC_SIZE],
		T *d_in,
		SizeT cta_offset,
		const SizeT &guarded_elements)
	{
		if (guarded_elements >= TILE_SIZE) {
			LoadValid<T, Transform>(data, d_in, cta_offset);
		} else {
			Iterate<0, 0>::template LoadValid<T, Transform>(
				data, d_in + cta_offset, guarded_elements);
		}
	}


	/**
	 * Load guarded_elements of a tile and out_of_bounds default
	 */
	template <
		typename T,
		typename SizeT>
	static __device__ __forceinline__ void LoadValid(
		T data[][LOAD_VEC_SIZE],
		T *d_in,
		SizeT cta_offset,
		const SizeT &guarded_elements,
		const T &oob_default)
	{
		LoadValid<T, Operators<T>::NopTransform>(
			data, d_in, cta_offset, guarded_elements, oob_default);
	}


	/**
	 * Load guarded_elements of a tile
	 */
	template <
		typename T,
		typename SizeT>
	static __device__ __forceinline__ void LoadValid(
		T data[][LOAD_VEC_SIZE],
		T *d_in,
		SizeT cta_offset,
		const SizeT &guarded_elements)
	{
		LoadValid<T, Operators<T>::NopTransform, int>(
			data, d_in, cta_offset, guarded_elements);
	}


	/**
	 * Load a full tile and initialize discontinuity flags when values change
	 * between consecutive elements
	 */
	template <
		bool FIRST_TILE,								// Whether or not this is the first tile loaded by the CTA
		typename T,										// Tile type
		typename Flag,									// Discontinuity flag type
		typename SizeT,
		typename EqualityOp>
	static __device__ __forceinline__ void LoadDiscontinuity(
		T data[][LOAD_VEC_SIZE],
		Flag flags[][LOAD_VEC_SIZE],
		T *d_in,
		SizeT cta_offset,
		EqualityOp equality_op)
	{
		const size_t MASK = ((sizeof(T) * 8 * LOAD_VEC_SIZE) - 1);

		if ((CHECK_ALIGNMENT) && (LOAD_VEC_SIZE > 1) && (((size_t) d_in) & MASK)) {

			Iterate<0, 0>::template LoadDiscontinuity<FIRST_TILE>(
				data, flags, d_in + cta_offset, equality_op);

		} else {

			// Use an aliased pointer to keys array to perform built-in vector loads
			typedef typename VecType<T, LOAD_VEC_SIZE>::Type VectorType;

			VectorType *vectors = (VectorType *) data;
			VectorType *d_in_vectors = (VectorType *) (d_in + cta_offset + (threadIdx.x << LOG_LOAD_VEC_SIZE));

			Iterate<0, 0>::template LoadDiscontinuity<FIRST_TILE>(
				data, flags, vectors, d_in_vectors, equality_op);
		}
	}

	/**
	 * Load guarded_elements of a tile and initialize discontinuity flags when
	 * values change between consecutive elements
	 */
	template <
		bool FIRST_TILE,								// Whether or not this is the first tile loaded by the CTA
		typename T,										// Tile type
		typename Flag,									// Discontinuity flag type
		typename SizeT,									// Integer type for indexing into global arrays
		typename EqualityOp>
	static __device__ __forceinline__ void LoadDiscontinuity(
		T data[][LOAD_VEC_SIZE],
		Flag flags[][LOAD_VEC_SIZE],
		T *d_in,
		SizeT cta_offset,
		const SizeT &guarded_elements,
		const Flag &oob_default_flag,
		EqualityOp equality_op)
	{
		if (guarded_elements >= TILE_SIZE) {

			LoadDiscontinuity<FIRST_TILE>(data, flags, d_in, cta_offset, equality_op);

		} else {

			Iterate<0, 0>::template LoadDiscontinuity<FIRST_TILE>(
				data, flags, d_in + cta_offset, guarded_elements, oob_default_flag, equality_op);
		}
	} 
};


} // namespace io
} // namespace util
} // namespace b40c

