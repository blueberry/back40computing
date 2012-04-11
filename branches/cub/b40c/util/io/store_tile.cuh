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
 * Kernel utilities loading for CTA-sized tiles of data from memory
 ******************************************************************************/

#pragma once

#include <b40c/util/operators.cuh>
#include <b40c/util/vector_types.cuh>
#include <b40c/util/io/modified_store.cuh>

namespace b40c {
namespace util {
namespace io {

/**
 * Store a tile of items
 */
template <
	int ACTIVE_THREADS,											// Active threads that will be storing
	st::CacheModifier CACHE_MODIFIER = st::NONE>							// Cache modifier (e.g., WB/CG/CS/NONE/etc.)
class TileWriter
{
private:

	//---------------------------------------------------------------------
	// Iteration Structures
	//---------------------------------------------------------------------

	// Iteration
	template <int CURRENT, int TOTAL>
	struct Iterate
	{

		//---------------------------------------------------------------------
		// Store elements within a raking segment
		//---------------------------------------------------------------------

		// Unguarded vector
		template <typename VectorType>
		static __device__ __forceinline__ void StoreVector(
			const int SEGMENT,
			VectorType data_vectors[],
			VectorType *d_out_vectors)
		{
			const int OFFSET = (SEGMENT * ACTIVE_THREADS * TOTAL) + CURRENT;

			ModifiedStore<CACHE_MODIFIER>::St(
				data_vectors[CURRENT],
				d_out_vectors + OFFSET);

			// Next vector in segment
			Iterate<CURRENT + 1, TOTAL>::StoreVector(
				SEGMENT,
				data_vectors,
				d_out_vectors);
		}


		// Guarded singleton
		template <
			typename T,
			typename S,
			typename SizeT,
			typename TransformOp,
			int ELEMENTS>
		static __device__ __forceinline__ void StoreGuarded(
			const int SEGMENT,
			T (&data)[ELEMENTS],
			S *d_out,
			const SizeT &guarded_elements,
			TransformOp transform_op)
		{
			const int OFFSET = (SEGMENT * ACTIVE_THREADS * TOTAL) + CURRENT;

			if ((threadIdx.x * ELEMENTS) + OFFSET < guarded_elements) {
				// Transform and store
				S raw = transform_op(data[CURRENT]);
				ModifiedStore<CACHE_MODIFIER>::St(raw, d_out + OFFSET);
			}

			// Next store in segment
			Iterate<CURRENT + 1, TOTAL>::StoreGuarded(
				SEGMENT,
				data,
				d_out,
				guarded_elements,
				transform_op);
		}

		// Guarded singleton by flag
		template <
			typename Flag,
			int ELEMENTS,
			typename T,
			typename S,
			typename TransformOp>
		static __device__ __forceinline__ void StoreGuardedByFlag(
			const int SEGMENT,
			Flag (&valid)[ELEMENTS],
			T (&data)[ELEMENTS],
			S *d_out,
			TransformOp transform_op)
		{
			const int OFFSET = (SEGMENT * ACTIVE_THREADS * TOTAL) + CURRENT;

			if (valid[CURRENT]) {
				// Transform and store
				S raw = transform_op(data[CURRENT]);
				ModifiedStore<CACHE_MODIFIER>::St(raw[CURRENT], d_out + OFFSET);
			}

			// Next store in segment
			Iterate<CURRENT + 1, TOTAL>::StoreGuardedByFlag(
				SEGMENT,
				data,
				d_out,
				guarded_elements,
				transform_op);
		}

		// Transform data within an unguarded segment
		template <
			typename T,
			typename S,
			typename TransformOp>
		static __device__ __forceinline__ void TransformRaw(
			T data[],
			S raw[],
			TransformOp transform_op)
		{
			raw[CURRENT] = transform_op(data[CURRENT]);

			// Next store in segment
			Iterate<CURRENT + 1, TOTAL>::TransformRaw(
				data,
				raw,
				transform_op);
		}


		//---------------------------------------------------------------------
		// Store strided segments
		//---------------------------------------------------------------------

		// Segment of unguarded vectors
		template <
			typename VectorType,
			int VECTORS>
		static __device__ __forceinline__ void StoreVectorSegment(
			VectorType data_vectors[][VECTORS],
			VectorType *d_out_vectors)
		{
			// Perform raking vector stores for this segment
			Iterate<0, VECTORS>::StoreVector(
				CURRENT,
				data_vectors[CURRENT],
				d_out_vectors);

			// Next segment
			Iterate<CURRENT + 1, TOTAL>::StoreVectorSegment(
				data_vectors,
				d_out_vectors);
		}

		// Segment of guarded singletons
		template <
			typename T,
			int ELEMENTS,
			typename S,
			typename SizeT,
			typename TransformOp>
		static __device__ __forceinline__ void StoreSegmentGuarded(
			T data[][ELEMENTS],
			S *d_out,
			const SizeT &guarded_elements,
			TransformOp transform_op)
		{
			// Perform guarded, transforming raking vector stores for this segment
			Iterate<0, ELEMENTS>::StoreGuarded(
				CURRENT,
				data[CURRENT],
				d_out,
				guarded_elements,
				transform_op);

			// Next segment
			Iterate<CURRENT + 1, TOTAL>::StoreSegmentGuarded(
				data,
				d_out,
				guarded_elements,
				transform_op);
		}

		// Segment of guarded singletons by flag
		template <
			typename Flag,
			int ELEMENTS,
			typename T,
			typename S,
			typename TransformOp>
		static __device__ __forceinline__ void StoreSegmentGuardedByFlag(
			Flag valid[][ELEMENTS],
			T data[][ELEMENTS],
			S *d_out,
			TransformOp transform_op)
		{
			// Perform guarded, transforming raking vector stores for this segment
			Iterate<0, ELEMENTS>::StoreGuardedByFlag(
				CURRENT,
				valid[CURRENT],
				data[CURRENT],
				d_out,
				transform_op);

			// Next segment
			Iterate<CURRENT + 1, TOTAL>::StoreSegmentGuardedByFlag(
				valid,
				data,
				d_out,
				transform_op);
		}
	};


	// Termination
	template <int TOTAL>
	struct Iterate<TOTAL, TOTAL>
	{
		// Unguarded vector
		template <typename VectorType>
		static __device__ __forceinline__ void StoreVector(
			const int SEGMENT,
			VectorType data_vectors[],
			VectorType *d_out_vectors) {}

		// Guarded singleton
		template <typename T, int ELEMENTS, typename S, typename SizeT, typename TransformOp>
		static __device__ __forceinline__ void StoreGuarded(
			const int SEGMENT,
			T (&data)[ELEMENTS],
			S *d_out,
			const SizeT &guarded_elements,
			TransformOp transform_op) {}

		// Guarded singleton by flag
		template <typename Flag, int ELEMENTS, typename T, typename S, typename TransformOp>
		static __device__ __forceinline__ void StoreGuardedByFlag(
			const int SEGMENT,
			Flag (&valid)[ELEMENTS],
			T (&data)[ELEMENTS],
			S *d_out,
			TransformOp transform_op) {}

		// Transform data within an unguarded segment
		template <typename T, typename S, typename TransformOp>
		static __device__ __forceinline__ void TransformRaw(
			T data[],
			S raw[],
			TransformOp transform_op) {}

		// Segment of unguarded vectors
		template <typename VectorType, int VECTORS>
		static __device__ __forceinline__ void StoreVectorSegment(
			VectorType data_vectors[][VECTORS],
			VectorType *d_out_vectors) {}

		// Segment of guarded singletons
		template <typename T, int ELEMENTS, typename S, typename SizeT, typename TransformOp>
		static __device__ __forceinline__ void StoreSegmentGuarded(
			T data[][ELEMENTS],
			S *d_out,
			const SizeT &guarded_elements,
			TransformOp transform_op) {}

		// Segment of guarded singletons by flag
		template <typename Flag, int ELEMENTS, typename T, typename S, typename TransformOp>
		static __device__ __forceinline__ void StoreSegmentGuardedByFlag(
			Flag valid[][ELEMENTS],
			T data[][ELEMENTS],
			S *d_out,
			TransformOp transform_op) {}

	};

public:

	//---------------------------------------------------------------------
	// Unguarded tile interface
	//---------------------------------------------------------------------

	/**
	 * Store a unguarded tile
	 */
	template <
		typename T,
		typename S,
		typename SizeT,
		typename TransformOp,
		int SEGMENTS,
		int ELEMENTS>
	static __device__ __forceinline__ void StoreUnguarded(
		T (&data)[SEGMENTS][ELEMENTS],
		S *d_out,
		SizeT cta_offset,
		TransformOp transform_op)
	{
		const int VEC_ELEMENTS 		= B40C_MIN(MAX_VEC_ELEMENTS, ELEMENTS);
		const int VECTORS 			= ELEMENTS / VEC_ELEMENTS;

		typedef typename VecType<S, VEC_ELEMENTS>::Type VectorType;

		// Raw data to store
		S raw[SEGMENTS][ELEMENTS];

		// Transform into raw
		Iterate<0, SEGMENTS * ELEMENTS>::TransformRaw(
			(T*) data,
			(S*) raw,
			transform_op);

		// Alias pointers
		VectorType (*data_vectors)[VECTORS] =
			reinterpret_cast<VectorType (*)[VECTORS]>(raw);

		VectorType *d_out_vectors =
			reinterpret_cast<VectorType *>(d_out + (threadIdx.x * ELEMENTS) + cta_offset);

		Iterate<0, SEGMENTS>::StoreVectorSegment(
			data_vectors,
			d_out_vectors);
	}


	/**
	 * Store a unguarded tile
	 */
	template <
		typename T,
		typename S,
		typename SizeT,
		int SEGMENTS,
		int ELEMENTS>
	static __device__ __forceinline__ void StoreUnguarded(
		T (&data)[SEGMENTS][ELEMENTS],
		S *d_out,
		SizeT cta_offset)
	{
		CastTransformOp<T, S> transform_op;
		StoreUnguarded(data, d_out, cta_offset, transform_op);
	}


	//---------------------------------------------------------------------
	// Guarded tile interface
	//---------------------------------------------------------------------

	/**
	 * Store a tile guarded by range
	 */
	template <
		typename T,
		int SEGMENTS,
		int ELEMENTS,
		typename S,
		typename SizeT,
		typename TransformOp>
	static __device__ __forceinline__ void StoreGuarded(
		T (&data)[SEGMENTS][ELEMENTS],
		S *d_out,
		SizeT cta_offset,
		const SizeT &guarded_elements,
		TransformOp transform_op)
	{
		Iterate<0, SEGMENTS>::StoreSegmentGuarded(
			data,
			d_out + (threadIdx.x * ELEMENTS) + cta_offset,
			guarded_elements,
			transform_op);
	}


	/**
	 * Store a tile guarded by range
	 */
	template <
		typename T,
		int SEGMENTS,
		int ELEMENTS,
		typename S,
		typename SizeT>
	static __device__ __forceinline__ void StoreGuarded(
		T (&data)[SEGMENTS][ELEMENTS],
		S *d_out,
		SizeT cta_offset,
		const SizeT &guarded_elements)
	{
		CastTransformOp<T, S> transform_op;
		StoreGuarded(data, d_out, cta_offset, guarded_elements, transform_op);
	}


	/**
	 * Store a tile guarded by flag
	 */
	template <
		typename Flag,
		int SEGMENTS,
		int ELEMENTS,
		typename T,
		typename S,
		typename SizeT,
		typename TransformOp>
	static __device__ __forceinline__ void StoreGuarded(
		Flag (&valid)[SEGMENTS][ELEMENTS],
		T (&data)[SEGMENTS][ELEMENTS],
		S *d_out,
		SizeT cta_offset,
		TransformOp transform_op = CastTransformOp<T, S>())
	{
		// Data to store
		S raw[SEGMENTS][ELEMENTS];

		Iterate<0, SEGMENTS>::StoreSegmentGuardedByFlag(
			valid,
			data,
			raw,
			d_out + (threadIdx.x * ELEMENTS) + cta_offset,
			transform_op);
	}


	/**
	 * Store a tile guarded by flag
	 */
	template <
		typename Flag,
		int SEGMENTS,
		int ELEMENTS,
		typename T,
		typename S,
		typename SizeT>
	static __device__ __forceinline__ void StoreGuarded(
		Flag (&valid)[SEGMENTS][ELEMENTS],
		T (&data)[SEGMENTS][ELEMENTS],
		S *d_out,
		SizeT cta_offset)
	{
		CastTransformOp<T, S> transform_op;
		StoreGuarded(valid, data, d_out, cta_offset, transform_op);
	}
};



} // namespace io
} // namespace util
} // namespace b40c

