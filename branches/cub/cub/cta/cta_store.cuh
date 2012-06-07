/******************************************************************************
 * 
 * Copyright (c) 2010-2012, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2012, NVIDIA CORPORATION.  All rights reserved.
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
 ******************************************************************************/

/******************************************************************************
 * Cooperative store abstraction for CTAs.
 ******************************************************************************/

#pragma once

#include <cub/macro_utils.cuh>
#include <cub/operators.cuh>
#include <cub/type_utils.cuh>
#include <cub/ns_umbrella.cuh>
#include <cub/thread/thread_store.cuh>

CUB_NS_PREFIX
namespace cub {

/**
 * Store a tile of items
 */
template <
	int 			ACTIVE_THREADS,						// Active threads that will be storing
	StoreModifier 	MODIFIER = STORE_NONE>				// Cache modifier (e.g., WB/CG/CS/NONE/etc.)
class CtaStore
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
		template <typename Vector>
		static __device__ __forceinline__ void StoreVector(
			const int STRIP,
			Vector data_vectors[],
			Vector *d_out_vectors)
		{
			const int OFFSET = (STRIP * ACTIVE_THREADS * TOTAL) + CURRENT;

			ThreadStore<MODIFIER>(d_out_vectors + OFFSET, data_vectors[CURRENT]);

			// Next vector in segment
			Iterate<CURRENT + 1, TOTAL>::StoreVector(
				STRIP,
				data_vectors,
				d_out_vectors);
		}


		// Guarded singleton
		template <
			typename T,
			typename S,
			typename SizeT,
			typename TransformOp,
			int THREAD_STRIP_ELEMENTS>
		static __device__ __forceinline__ void StoreGuarded(
			const int STRIP,
			T (&data)[THREAD_STRIP_ELEMENTS],
			S *d_out,
			const SizeT &guarded_elements,
			TransformOp transform_op)
		{
			const int OFFSET = (STRIP * ACTIVE_THREADS * TOTAL) + CURRENT;

			if ((threadIdx.x * THREAD_STRIP_ELEMENTS) + OFFSET < guarded_elements)
			{
				// Transform and store
				S raw = transform_op(data[CURRENT]);
				ThreadStore<MODIFIER>(d_out + OFFSET, raw);
			}

			// Next store in segment
			Iterate<CURRENT + 1, TOTAL>::StoreGuarded(
				STRIP,
				data,
				d_out,
				guarded_elements,
				transform_op);
		}

		// Guarded singleton by flag
		template <
			typename Flag,
			int THREAD_STRIP_ELEMENTS,
			typename T,
			typename S,
			typename TransformOp>
		static __device__ __forceinline__ void StoreGuardedByFlag(
			const int STRIP,
			Flag (&valid)[THREAD_STRIP_ELEMENTS],
			T (&data)[THREAD_STRIP_ELEMENTS],
			S *d_out,
			TransformOp transform_op)
		{
			const int OFFSET = (STRIP * ACTIVE_THREADS * TOTAL) + CURRENT;

			if (valid[CURRENT])
			{
				// Transform and store
				S raw = transform_op(data[CURRENT]);
				ThreadStore<MODIFIER>(d_out + OFFSET, raw[CURRENT]);
			}

			// Next store in segment
			Iterate<CURRENT + 1, TOTAL>::StoreGuardedByFlag(
				STRIP,
				valid,
				data,
				d_out,
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
			typename Vector,
			int VECTORS>
		static __device__ __forceinline__ void StoreVectorSegment(
			Vector data_vectors[][VECTORS],
			Vector *d_out_vectors)
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
			int THREAD_STRIP_ELEMENTS,
			typename S,
			typename SizeT,
			typename TransformOp>
		static __device__ __forceinline__ void StoreSegmentGuarded(
			T data[][THREAD_STRIP_ELEMENTS],
			S *d_out,
			const SizeT &guarded_elements,
			TransformOp transform_op)
		{
			// Perform guarded, transforming raking vector stores for this segment
			Iterate<0, THREAD_STRIP_ELEMENTS>::StoreGuarded(
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
			int THREAD_STRIP_ELEMENTS,
			typename T,
			typename S,
			typename TransformOp>
		static __device__ __forceinline__ void StoreSegmentGuardedByFlag(
			Flag valid[][THREAD_STRIP_ELEMENTS],
			T data[][THREAD_STRIP_ELEMENTS],
			S *d_out,
			TransformOp transform_op)
		{
			// Perform guarded, transforming raking vector stores for this segment
			Iterate<0, THREAD_STRIP_ELEMENTS>::StoreGuardedByFlag(
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
		template <typename Vector>
		static __device__ __forceinline__ void StoreVector(
			const int STRIP,
			Vector data_vectors[],
			Vector *d_out_vectors) {}

		// Guarded singleton
		template <typename T, int THREAD_STRIP_ELEMENTS, typename S, typename SizeT, typename TransformOp>
		static __device__ __forceinline__ void StoreGuarded(
			const int STRIP,
			T (&data)[THREAD_STRIP_ELEMENTS],
			S *d_out,
			const SizeT &guarded_elements,
			TransformOp transform_op) {}

		// Guarded singleton by flag
		template <typename Flag, int THREAD_STRIP_ELEMENTS, typename T, typename S, typename TransformOp>
		static __device__ __forceinline__ void StoreGuardedByFlag(
			const int STRIP,
			Flag (&valid)[THREAD_STRIP_ELEMENTS],
			T (&data)[THREAD_STRIP_ELEMENTS],
			S *d_out,
			TransformOp transform_op) {}

		// Transform data within an unguarded segment
		template <typename T, typename S, typename TransformOp>
		static __device__ __forceinline__ void TransformRaw(
			T data[],
			S raw[],
			TransformOp transform_op) {}

		// Segment of unguarded vectors
		template <typename Vector, int VECTORS>
		static __device__ __forceinline__ void StoreVectorSegment(
			Vector data_vectors[][VECTORS],
			Vector *d_out_vectors) {}

		// Segment of guarded singletons
		template <typename T, int THREAD_STRIP_ELEMENTS, typename S, typename SizeT, typename TransformOp>
		static __device__ __forceinline__ void StoreSegmentGuarded(
			T data[][THREAD_STRIP_ELEMENTS],
			S *d_out,
			const SizeT &guarded_elements,
			TransformOp transform_op) {}

		// Segment of guarded singletons by flag
		template <typename Flag, int THREAD_STRIP_ELEMENTS, typename T, typename S, typename TransformOp>
		static __device__ __forceinline__ void StoreSegmentGuardedByFlag(
			Flag valid[][THREAD_STRIP_ELEMENTS],
			T data[][THREAD_STRIP_ELEMENTS],
			S *d_out,
			TransformOp transform_op) {}

	};

public:

	//---------------------------------------------------------------------
	// Unguarded tile interface
	//---------------------------------------------------------------------

	/**
	 * Store a full, strip-mined tile
	 */
	template <
		typename T,
		int CTA_STRIPS,
		int THREAD_STRIP_ELEMENTS,
		typename S,
		typename SizeT,
		typename TransformOp>
	static __device__ __forceinline__ void StoreUnguarded(
		T (&data)[CTA_STRIPS][THREAD_STRIP_ELEMENTS],
		S *d_out,
		SizeT cta_offset,
		TransformOp transform_op)
	{
		enum
		{
			MAX_VEC_ELEMENTS 	= CUB_MIN(MAX_VEC_ELEMENTS, THREAD_STRIP_ELEMENTS),
			VEC_ELEMENTS		= (THREAD_STRIP_ELEMENTS % MAX_VEC_ELEMENTS == 0) ? MAX_VEC_ELEMENTS : 1,	// Elements must be an even multiple of vector size
			VECTORS 			= THREAD_STRIP_ELEMENTS / VEC_ELEMENTS,
		};

		typedef typename VectorType<S, VEC_ELEMENTS>::Type Vector;

		// Raw data to store
		S raw[CTA_STRIPS][THREAD_STRIP_ELEMENTS];

		// Transform into raw
		Iterate<0, CTA_STRIPS * THREAD_STRIP_ELEMENTS>::TransformRaw(
			(T*) data,
			(S*) raw,
			transform_op);

		// Alias pointers
		Vector (*data_vectors)[VECTORS] =
			reinterpret_cast<Vector (*)[VECTORS]>(raw);

		Vector *d_out_vectors =
			reinterpret_cast<Vector *>(d_out + (threadIdx.x * THREAD_STRIP_ELEMENTS) + cta_offset);

		Iterate<0, CTA_STRIPS>::StoreVectorSegment(
			data_vectors,
			d_out_vectors);
	}


	/**
	 * Store a full tile
	 */
	template <
		typename T,
		int THREAD_ELEMENTS,
		typename S,
		typename SizeT,
		typename TransformOp>
	static __device__ __forceinline__ void StoreUnguarded(
		T (&data)[THREAD_ELEMENTS],
		S *d_out,
		SizeT cta_offset,
		TransformOp transform_op)
	{
		T (&data_2d)[1][THREAD_ELEMENTS] = reinterpret_cast<T (&)[1][THREAD_ELEMENTS]>(data);
		StoreUnguarded(data_2d, d_out, cta_offset, transform_op);
	}


	/**
	 * Store a full, strip-mined tile
	 */
	template <
		typename T,
		int CTA_STRIPS,
		int THREAD_STRIP_ELEMENTS,
		typename S,
		typename SizeT>
	static __device__ __forceinline__ void StoreUnguarded(
		T (&data)[CTA_STRIPS][THREAD_STRIP_ELEMENTS],
		S *d_out,
		SizeT cta_offset)
	{
		CastTransformOp<T, S> transform_op;
		StoreUnguarded(data, d_out, cta_offset, transform_op);
	}


	/**
	 * Store a full tile
	 */
	template <
		typename T,
		int THREAD_ELEMENTS,
		typename S,
		typename SizeT>
	static __device__ __forceinline__ void StoreUnguarded(
		T (&data)[THREAD_ELEMENTS],
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
	 * Store a strip-mined tile, guarded by range.
	 */
	template <
		typename T,
		int CTA_STRIPS,
		int THREAD_STRIP_ELEMENTS,
		typename S,
		typename SizeT,
		typename TransformOp>
	static __device__ __forceinline__ void StoreGuarded(
		T (&data)[CTA_STRIPS][THREAD_STRIP_ELEMENTS],
		S *d_out,
		SizeT cta_offset,
		const SizeT &guarded_elements,
		TransformOp transform_op)
	{
		Iterate<0, CTA_STRIPS>::StoreSegmentGuarded(
			data,
			d_out + (threadIdx.x * THREAD_STRIP_ELEMENTS) + cta_offset,
			guarded_elements,
			transform_op);
	}


	/**
	 * Store a tile, guarded by range
	 */
	template <
		typename T,
		int THREAD_ELEMENTS,
		typename S,
		typename SizeT,
		typename TransformOp>
	static __device__ __forceinline__ void StoreGuarded(
		T (&data)[THREAD_ELEMENTS],
		S *d_out,
		SizeT cta_offset,
		const SizeT &guarded_elements,
		TransformOp transform_op)
	{
		T (&data_2d)[1][THREAD_ELEMENTS] = reinterpret_cast<T (&)[1][THREAD_ELEMENTS]>(data);
		StoreGuarded(data_2d, d_out, cta_offset, guarded_elements, transform_op);
	}


	/**
	 * Store a strip-mined tile, guarded by range
	 */
	template <
		typename T,
		int CTA_STRIPS,
		int THREAD_STRIP_ELEMENTS,
		typename S,
		typename SizeT>
	static __device__ __forceinline__ void StoreGuarded(
		T (&data)[CTA_STRIPS][THREAD_STRIP_ELEMENTS],
		S *d_out,
		SizeT cta_offset,
		const SizeT &guarded_elements)
	{
		CastTransformOp<T, S> transform_op;
		StoreGuarded(data, d_out, cta_offset, guarded_elements, transform_op);
	}


	/**
	 * Store a tile, guarded by range
	 */
	template <
		typename T,
		int THREAD_ELEMENTS,
		typename S,
		typename SizeT>
	static __device__ __forceinline__ void StoreGuarded(
		T (&data)[THREAD_ELEMENTS],
		S *d_out,
		SizeT cta_offset,
		const SizeT &guarded_elements)
	{
		CastTransformOp<T, S> transform_op;
		StoreGuarded(data, d_out, cta_offset, guarded_elements, transform_op);
	}


	/**
	 * Store a strip-mined tile, guarded by flag
	 */
	template <
		typename Flag,
		int CTA_STRIPS,
		int THREAD_STRIP_ELEMENTS,
		typename T,
		typename S,
		typename SizeT,
		typename TransformOp>
	static __device__ __forceinline__ void StoreGuarded(
		Flag (&valid)[CTA_STRIPS][THREAD_STRIP_ELEMENTS],
		T (&data)[CTA_STRIPS][THREAD_STRIP_ELEMENTS],
		S *d_out,
		SizeT cta_offset,
		TransformOp transform_op)
	{
		// Data to store
		S raw[CTA_STRIPS][THREAD_STRIP_ELEMENTS];

		Iterate<0, CTA_STRIPS>::StoreSegmentGuardedByFlag(
			valid,
			data,
			raw,
			d_out + (threadIdx.x * THREAD_STRIP_ELEMENTS) + cta_offset,
			transform_op);
	}


	/**
	 * Store a tile guarded by flag
	 */
	template <
		typename Flag,
		int THREAD_ELEMENTS,
		typename T,
		typename S,
		typename SizeT,
		typename TransformOp>
	static __device__ __forceinline__ void StoreGuarded(
		Flag (&valid)[THREAD_ELEMENTS],
		T (&data)[THREAD_ELEMENTS],
		S *d_out,
		SizeT cta_offset,
		TransformOp transform_op)
	{
		Flag (&valid_2d)[1][THREAD_ELEMENTS] = reinterpret_cast<T (&)[1][THREAD_ELEMENTS]>(valid);
		T (&data_2d)[1][THREAD_ELEMENTS] = reinterpret_cast<T (&)[1][THREAD_ELEMENTS]>(data);
		StoreGuarded(valid_2d, data_2d, d_out, cta_offset, transform_op);
	}


	/**
	 * Store a strip-mined tile, guarded by flag
	 */
	template <
		typename Flag,
		int CTA_STRIPS,
		int THREAD_STRIP_ELEMENTS,
		typename T,
		typename S,
		typename SizeT>
	static __device__ __forceinline__ void StoreGuarded(
		Flag (&valid)[CTA_STRIPS][THREAD_STRIP_ELEMENTS],
		T (&data)[CTA_STRIPS][THREAD_STRIP_ELEMENTS],
		S *d_out,
		SizeT cta_offset)
	{
		CastTransformOp<T, S> transform_op;
		StoreGuarded(valid, data, d_out, cta_offset, transform_op);
	}


	/**
	 * Store a tile guarded by flag
	 */
	template <
		typename Flag,
		int THREAD_ELEMENTS,
		typename T,
		typename S,
		typename SizeT>
	static __device__ __forceinline__ void StoreGuarded(
		Flag (&valid)[THREAD_ELEMENTS],
		T (&data)[THREAD_ELEMENTS],
		S *d_out,
		SizeT cta_offset)
	{
		CastTransformOp<T, S> transform_op;
		StoreGuarded(valid, data, d_out, cta_offset, transform_op);
	}
};




} // namespace cub
CUB_NS_POSTFIX
