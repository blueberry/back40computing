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
 * Thanks!
 * 
 ******************************************************************************/

/******************************************************************************
 * Kernel utilities for loading tiles of loaded through global memory
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
 * Cast transform
 */
template <typename T, typename S>
struct CastTransformOp
{
	__device__ __forceinline__ T operator ()(S item)
	{
		return (T) item;
	}
};



/**
 * Load a tile of items
 */
template <
	int ACTIVE_THREADS,								// Active threads that will be loading
	ld::CacheModifier CACHE_MODIFIER = ld::NONE>	// Cache modifier (e.g., CA/CG/CS/NONE/etc.)
struct LoadTile
{

	//---------------------------------------------------------------------
	// Iteration Structures
	//---------------------------------------------------------------------

	// Iteration
	template <int CURRENT, int TOTAL>
	struct Iterate
	{

		//---------------------------------------------------------------------
		// Load elements within a raking segment
		//---------------------------------------------------------------------

		// Unguarded vector
		template <typename VectorType>
		static __device__ __forceinline__ void LoadVector(
			const int SEGMENT,
			VectorType data_vectors[],
			VectorType *d_in_vectors)
		{
			const int OFFSET = (SEGMENT * ACTIVE_THREADS * TOTAL) + CURRENT;

			ModifiedLoad<CACHE_MODIFIER>::Ld(
				data_vectors[CURRENT],
				d_in_vectors + OFFSET);

			// Next vector in segment
			Iterate<CURRENT + 1, TOTAL>::LoadVector(SEGMENT, data_vectors, d_in_vectors);
		}

		// Unguarded tex vector
		template <typename VectorType>
		static __device__ __forceinline__ void LoadTexVector(
			const int SEGMENT,
			VectorType data_vectors[],
			const texture<VectorType, cudaTextureType1D, cudaReadModeElementType> &ref,
			unsigned int base_thread_offset)
		{
			const int OFFSET = (SEGMENT * ACTIVE_THREADS * TOTAL) + CURRENT;

			data_vectors[CURRENT] = tex1Dfetch(
				ref,
				base_thread_offset + OFFSET);

			// Next vector in segment
			Iterate<CURRENT + 1, TOTAL>::LoadTexVector(SEGMENT, data_vectors, ref, base_thread_offset);
		}


		// Guarded singleton
		template <
			typename Flag,
			typename T,
			typename S,
			typename SizeT,
			typename TransformOp,
			int ELEMENTS>
		static __device__ __forceinline__ void LoadGuarded(
			const int SEGMENT,
			Flag (&valid)[ELEMENTS],
			T (&transformed)[ELEMENTS],
			S (&loaded)[ELEMENTS],
			S *d_in,
			const SizeT &guarded_elements,
			TransformOp transform_op)
		{
			const int OFFSET = (SEGMENT * ACTIVE_THREADS * TOTAL) + CURRENT;

			if ((threadIdx.x * ELEMENTS) + OFFSET < guarded_elements) {

				ModifiedLoad<CACHE_MODIFIER>::Ld(loaded[CURRENT], d_in + OFFSET);
				transformed[CURRENT] = transform_op(loaded[CURRENT]);
				valid[CURRENT] = 1;

			} else {

				valid[CURRENT] = 0;
			}

			// Next load in segment
			Iterate<CURRENT + 1, TOTAL>::LoadGuarded(
				SEGMENT,
				valid,
				transformed,
				loaded,
				d_in,
				guarded_elements,
				transform_op);
		}

		// Initialize valid flag within an unguarded segment
		template <
			typename T,
			typename S,
			typename Flag,
			typename TransformOp>
		static __device__ __forceinline__ void InitUnguarded(
			Flag valid[],
			T transformed[],
			S loaded[],
			TransformOp transform_op)
		{

			valid[CURRENT] = 1;
			transformed[CURRENT] = transform_op(loaded[CURRENT]);

			// Next load in segment
			Iterate<CURRENT + 1, TOTAL>::InitUnguarded(
				valid,
				transformed,
				loaded,
				transform_op);
		}


		//---------------------------------------------------------------------
		// Load strided segments
		//---------------------------------------------------------------------

		// Segment of unguarded vectors
		template <
			typename Flag,
			typename T,
			typename S,
			typename VectorType,
			typename TransformOp,
			int ELEMENTS,
			int VECTORS>
		static __device__ __forceinline__ void LoadVectorSegment(
			Flag valid[][ELEMENTS],
			T transformed[][ELEMENTS],
			S loaded[][ELEMENTS],
			VectorType data_vectors[][VECTORS],
			VectorType *d_in_vectors,
			TransformOp transform_op)
		{
			// Perform raking vector loads for this segment
			Iterate<0, VECTORS>::LoadVector(
				CURRENT,
				data_vectors[CURRENT],
				d_in_vectors);

			// Next segment
			Iterate<CURRENT + 1, TOTAL>::LoadVectorSegment(
				valid,
				transformed,
				loaded,
				data_vectors,
				d_in_vectors,
				transform_op);
		}

		// Segment of unguarded tex vectors
		template <
			typename Flag,
			typename T,
			typename S,
			typename VectorType,
			typename SizeT,
			typename TransformOp,
			int ELEMENTS,
			int VECTORS>
		static __device__ __forceinline__ void LoadTexVectorSegment(
			Flag valid[][ELEMENTS],
			T transformed[][ELEMENTS],
			S loaded[][ELEMENTS],
			VectorType data_vectors[][VECTORS],
			const texture<VectorType, cudaTextureType1D, cudaReadModeElementType> &ref,
			SizeT base_thread_offset,
			TransformOp transform_op)
		{
			// Perform raking tex vector loads for this segment
			Iterate<0, VECTORS>::LoadTexVector(
				CURRENT,
				data_vectors[CURRENT],
				ref,
				base_thread_offset);

			// Next segment
			Iterate<CURRENT + 1, TOTAL>::LoadTexVectorSegment(
				valid,
				transformed,
				loaded,
				data_vectors,
				ref,
				base_thread_offset,
				transform_op);
		}

		// Segment of guarded singletons
		template <
			typename Flag,
			typename T,
			typename S,
			typename SizeT,
			typename TransformOp,
			int ELEMENTS>
		static __device__ __forceinline__ void LoadSegmentGuarded(
			Flag valid[][ELEMENTS],
			T transformed[][ELEMENTS],
			S loaded[][ELEMENTS],
			S *d_in,
			const SizeT &guarded_elements,
			TransformOp transform_op)
		{
			// Perform guarded, transforming raking vector loads for this segment
			Iterate<0, ELEMENTS>::LoadGuarded(
				CURRENT,
				valid[CURRENT],
				transformed[CURRENT],
				loaded[CURRENT],
				d_in,
				guarded_elements,
				transform_op);

			// Next segment
			Iterate<CURRENT + 1, TOTAL>::LoadSegmentGuarded(
				valid,
				transformed,
				loaded,
				d_in,
				guarded_elements,
				transform_op);
		}
	};


	// Termination
	template <int TOTAL>
	struct Iterate<TOTAL, TOTAL>
	{
		// Unguarded vector
		template <typename VectorType>
		static __device__ __forceinline__ void LoadVector(
			const int SEGMENT,
			VectorType data_vectors[],
			VectorType *d_in_vectors) {}

		// Unguarded tex vector
		template <typename VectorType>
		static __device__ __forceinline__ void LoadTexVector(
			const int SEGMENT,
			VectorType data_vectors[],
			const texture<VectorType, cudaTextureType1D, cudaReadModeElementType> &ref,
			unsigned int base_thread_offset) {}

		// Guarded singleton
		template <typename Flag, typename T, typename S, typename SizeT, typename TransformOp, int ELEMENTS>
		static __device__ __forceinline__ void LoadGuarded(
			const int SEGMENT,
			Flag (&valid)[ELEMENTS],
			T (&transformed)[ELEMENTS],
			S (&loaded)[ELEMENTS],
			S *d_in,
			const SizeT &guarded_elements,
			TransformOp transform_op) {}

		// Initialize valid flag within an unguarded segment
		template <typename T, typename S, typename Flag, typename TransformOp>
		static __device__ __forceinline__ void InitUnguarded(
			Flag valid[],
			T transformed[],
			S loaded[],
			TransformOp transform_op) {}

		// Segment of unguarded vectors
		template <typename Flag, typename T, typename S, typename VectorType, typename TransformOp, int ELEMENTS, int VECTORS>
		static __device__ __forceinline__ void LoadVectorSegment(
			Flag valid[][ELEMENTS],
			T transformed[][ELEMENTS],
			S loaded[][ELEMENTS],
			VectorType data_vectors[][VECTORS],
			VectorType *d_in_vectors,
			TransformOp transform_op) {}

		// Segment of unguarded tex vectors
		template <typename Flag, typename T, typename S, typename VectorType, typename SizeT, typename TransformOp, int ELEMENTS, int VECTORS>
		static __device__ __forceinline__ void LoadTexVectorSegment(
			Flag valid[][ELEMENTS],
			T transformed[][ELEMENTS],
			S loaded[][ELEMENTS],
			VectorType data_vectors[][VECTORS],
			const texture<VectorType, cudaTextureType1D, cudaReadModeElementType> &ref,
			SizeT base_thread_offset,
			TransformOp transform_op) {}

		// Segment of guarded singletons
		template <typename Flag, typename T, typename S, typename SizeT, typename TransformOp, int ELEMENTS>
		static __device__ __forceinline__ void LoadSegmentGuarded(
			Flag valid[][ELEMENTS],
			T transformed[][ELEMENTS],
			S loaded[][ELEMENTS],
			S *d_in,
			const SizeT &guarded_elements,
			TransformOp transform_op) {}
	};



	//---------------------------------------------------------------------
	// Interface
	//---------------------------------------------------------------------

	/**
	 * Load a full tile
	 */
	template <
		typename Flag,
		typename T,
		typename S,
		typename SizeT,
		typename TransformOp,
		int SEGMENTS,
		int ELEMENTS>
	static __device__ __forceinline__ void LoadTileUnguarded(
		Flag (&valid)[SEGMENTS][ELEMENTS],
		T (&data)[SEGMENTS][ELEMENTS],
		S *d_in,
		SizeT cta_offset,
		TransformOp transform_op)
	{
		const int VEC_ELEMENTS 		= B40C_MIN(MAX_VEC_ELEMENTS, ELEMENTS);
		const int VECTORS 			= ELEMENTS / VEC_ELEMENTS;

		typedef typename VecType<S, VEC_ELEMENTS>::Type VectorType;

		// Data to load
		S loaded[SEGMENTS][ELEMENTS];

		// Use an aliased pointer to loaded array
		VectorType (*data_vectors)[VECTORS] = (VectorType (*)[VECTORS]) loaded;
		VectorType *d_in_vectors = (VectorType *) (d_in + (threadIdx.x * ELEMENTS) + cta_offset);

		Iterate<0, SEGMENTS>::LoadVectorSegment(
			valid,
			data,
			loaded,
			data_vectors,
			d_in_vectors,
			transform_op);

		// Initialize valid and transform loaded
		Iterate<0, SEGMENTS * ELEMENTS>::InitUnguarded(
			(Flag*) valid,
			(T*) data,
			(S*) loaded,
			transform_op);
	}


	/**
	 * Load a full tile
	 */
	template <
		typename T,
		typename S,
		typename SizeT,
		int SEGMENTS,
		int ELEMENTS>
	static __device__ __forceinline__ void LoadTileUnguarded(
		T (&data)[SEGMENTS][ELEMENTS],
		S *d_in,
		SizeT cta_offset)
	{
		int valid[SEGMENTS][ELEMENTS];
		CastTransformOp<T, S> transform_op;

		LoadTileUnguarded(
			valid,
			data,
			d_in,
			cta_offset,
			transform_op);
	}


	/**
	 * Load a full tile (optionally using tex if READ_MODIFIER == ld::tex)
	 */
	template <
		typename Flag,
		typename T,
		typename VectorType,
		typename S,
		typename SizeT,
		typename TransformOp,
		int SEGMENTS,
		int ELEMENTS>
	static __device__ __forceinline__ void LoadTileUnguarded(
		Flag (&valid)[SEGMENTS][ELEMENTS],
		T (&data)[SEGMENTS][ELEMENTS],
		const texture<VectorType, cudaTextureType1D, cudaReadModeElementType> &ref,
		S *d_in,
		SizeT cta_offset,
		TransformOp transform_op)
	{
		if (CACHE_MODIFIER == ld::tex) {

			// Use tex
			const int VEC_ELEMENTS 		= sizeof(VectorType) / sizeof(S);
			const int VECTORS 			= ELEMENTS / VEC_ELEMENTS;

			// Data to load
			S loaded[SEGMENTS][ELEMENTS];

			// Use an aliased pointer to loaded array
			VectorType (*data_vectors)[VECTORS] = (VectorType (*)[VECTORS]) loaded;

			SizeT base_thread_offset = cta_offset / VEC_ELEMENTS;

			Iterate<0, SEGMENTS>::LoadTexVectorSegment(
				valid,
				data,
				loaded,
				data_vectors,
				ref,
				base_thread_offset + (threadIdx.x * VECTORS),
				transform_op);

			// Initialize valid and transform loaded
			Iterate<0, SEGMENTS * ELEMENTS>::InitUnguarded(
				(Flag*) valid,
				(T*) data,
				(S*) loaded,
				transform_op);

		} else {

			// Use normal loads
			LoadTileUnguarded(
				valid,
				data,
				d_in,
				cta_offset,
				transform_op);
		}
	}


	/**
	 * Load a full tile (optionally using tex if READ_MODIFIER == ld::tex)
	 */
	template <
		typename T,
		typename VectorType,
		typename S,
		typename SizeT,
		int SEGMENTS,
		int ELEMENTS>
	static __device__ __forceinline__ void LoadTileUnguarded(
		T (&data)[SEGMENTS][ELEMENTS],
		const texture<VectorType, cudaTextureType1D, cudaReadModeElementType> &ref,
		S *d_in,
		SizeT cta_offset)
	{
		int valid[SEGMENTS][ELEMENTS];
		CastTransformOp<T, S> transform_op;

		LoadTileUnguarded(
			valid,
			data,
			ref,
			d_in,
			cta_offset,
			transform_op);
	}


	/**
	 * Load a guarded tile
	 */
	template <
		typename Flag,
		typename T,
		typename S,
		typename SizeT,
		typename TransformOp,
		int SEGMENTS,
		int ELEMENTS>
	static __device__ __forceinline__ void LoadTileGuarded(
		Flag (&valid)[SEGMENTS][ELEMENTS],
		T (&data)[SEGMENTS][ELEMENTS],
		S *d_in,
		SizeT cta_offset,
		const SizeT &guarded_elements,
		TransformOp transform_op)
	{
		// Data to load
		S loaded[SEGMENTS][ELEMENTS];

		Iterate<0, SEGMENTS>::LoadSegmentGuarded(
			valid,
			data,
			loaded,
			d_in + (threadIdx.x * ELEMENTS) + cta_offset,
			guarded_elements,
			transform_op);
	}


	/**
	 * Load a guarded tile
	 */
	template <
		typename T,
		typename S,
		typename SizeT,
		int SEGMENTS,
		int ELEMENTS>
	static __device__ __forceinline__ void LoadTileGuarded(
		T (&data)[SEGMENTS][ELEMENTS],
		S *d_in,
		SizeT cta_offset,
		const SizeT &guarded_elements)
	{
		int valid[SEGMENTS][ELEMENTS];
		CastTransformOp<T, S> transform_op;

		LoadTileGuarded(
			valid,
			data,
			d_in,
			cta_offset,
			guarded_elements,
			transform_op);
	}

};



} // namespace io
} // namespace util
} // namespace b40c

