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
 * Cooperative load abstraction for CTAs.
 ******************************************************************************/

#pragma once

#include <cub/macro_utils.cuh>
#include <cub/operators.cuh>
#include <cub/type_utils.cuh>
#include <cub/ns_umbrella.cuh>
#include <cub/thread/thread_load.cuh>

CUB_NS_PREFIX
namespace cub {

/**
 * Load a tile of items
 */
template <
	int 			CTA_THREADS,				// Active threads that will be loading
	LoadModifier 	MODIFIER = LOAD_NONE>		// Cache modifier (e.g., TEX/CA/CG/CS/NONE/etc.)
class CtaLoad
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
		// Load elements within a raking segment
		//---------------------------------------------------------------------

		// Unguarded vector
		template <typename Vector>
		static __device__ __forceinline__ void LoadVector(
			const int STRIP,
			Vector data_vectors[],
			Vector *d_in_vectors)
		{
			const int OFFSET = (STRIP * CTA_THREADS * TOTAL) + CURRENT;

			data_vectors[CURRENT] = ThreadLoad<MODIFIER>(d_in_vectors + OFFSET);

			// Next vector in segment
			Iterate<CURRENT + 1, TOTAL>::LoadVector(
				STRIP,
				data_vectors,
				d_in_vectors);
		}

		// Unguarded tex vector
		template <typename Vector>
		static __device__ __forceinline__ void LoadTexVector(
			const int STRIP,
			Vector data_vectors[],
			texture<Vector, cudaTextureType1D, cudaReadModeElementType> ref,
			unsigned int base_thread_offset)
		{
			const int OFFSET = (STRIP * CTA_THREADS * TOTAL) + CURRENT;

			data_vectors[CURRENT] = tex1Dfetch(
				ref,
				base_thread_offset + OFFSET);

			// Next vector in segment
			Iterate<CURRENT + 1, TOTAL>::LoadTexVector(
				STRIP,
				data_vectors,
				ref,
				base_thread_offset);
		}


		// Guarded singleton
		template <
			typename Flag,
			typename T,
			typename S,
			int THREAD_STRIP_ELEMENTS,
			typename SizeT,
			typename TransformOp>
		static __device__ __forceinline__ void LoadGuarded(
			const int STRIP,
			Flag (&valid)[THREAD_STRIP_ELEMENTS],
			T (&data)[THREAD_STRIP_ELEMENTS],
			S *d_in,
			const SizeT &guarded_elements,
			TransformOp transform_op)
		{
			const int OFFSET = (STRIP * CTA_THREADS * TOTAL) + CURRENT;

			valid[CURRENT] = ((threadIdx.x * THREAD_STRIP_ELEMENTS) + OFFSET) < guarded_elements;

			if (valid[CURRENT])
			{
				// Load and transform
				S raw_data = ThreadLoad<MODIFIER>(d_in + OFFSET);
				data[CURRENT] = transform_op(raw_data);
			}

			// Next load in segment
			Iterate<CURRENT + 1, TOTAL>::LoadGuarded(
				STRIP,
				valid,
				data,
				d_in,
				guarded_elements,
				transform_op);
		}

		// Transform data and initialize valid flag within an unguarded segment
		template <
			typename T,
			typename S,
			typename TransformOp>
		static __device__ __forceinline__ void TransformRaw(
			T data[],
			S raw[],
			TransformOp transform_op)
		{
			data[CURRENT] = transform_op(raw[CURRENT]);

			// Next load in segment
			Iterate<CURRENT + 1, TOTAL>::TransformRaw(
				data,
				raw,
				transform_op);
		}


		//---------------------------------------------------------------------
		// Load strided segments
		//---------------------------------------------------------------------

		// Segment of unguarded vectors
		template <
			typename Vector,
			int VECTORS>
		static __device__ __forceinline__ void LoadVectorSegment(
			Vector data_vectors[][VECTORS],
			Vector *d_in_vectors)
		{
			// Perform raking vector loads for this segment
			Iterate<0, VECTORS>::LoadVector(
				CURRENT,
				data_vectors[CURRENT],
				d_in_vectors);

			// Next segment
			Iterate<CURRENT + 1, TOTAL>::LoadVectorSegment(
				data_vectors,
				d_in_vectors);
		}

		// Segment of unguarded tex vectors
		template <
			typename T,
			typename S,
			int THREAD_STRIP_ELEMENTS,
			typename Vector,
			typename SizeT,
			int VECTORS>
		static __device__ __forceinline__ void LoadTexVectorSegment(
			T data[][THREAD_STRIP_ELEMENTS],
			S raw[][THREAD_STRIP_ELEMENTS],
			Vector data_vectors[][VECTORS],
			texture<Vector, cudaTextureType1D, cudaReadModeElementType> ref,
			SizeT base_thread_offset)
		{
			// Perform raking tex vector loads for this segment
			Iterate<0, VECTORS>::LoadTexVector(
				CURRENT,
				data_vectors[CURRENT],
				ref,
				base_thread_offset);

			// Next segment
			Iterate<CURRENT + 1, TOTAL>::LoadTexVectorSegment(
				data,
				raw,
				data_vectors,
				ref,
				base_thread_offset);
		}

		// Segment of guarded singletons
		template <
			typename Flag,
			typename T,
			typename S,
			int THREAD_STRIP_ELEMENTS,
			typename SizeT,
			typename TransformOp>
		static __device__ __forceinline__ void LoadSegmentGuarded(
			Flag valid[][THREAD_STRIP_ELEMENTS],
			T data[][THREAD_STRIP_ELEMENTS],
			S *d_in,
			const SizeT &guarded_elements,
			TransformOp transform_op)
		{
			// Perform guarded, transforming raking vector loads for this segment
			Iterate<0, THREAD_STRIP_ELEMENTS>::LoadGuarded(
				CURRENT,
				valid[CURRENT],
				data[CURRENT],
				d_in,
				guarded_elements,
				transform_op);

			// Next segment
			Iterate<CURRENT + 1, TOTAL>::LoadSegmentGuarded(
				valid,
				data,
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
		template <typename Vector>
		static __device__ __forceinline__ void LoadVector(
			const int STRIP,
			Vector data_vectors[],
			Vector *d_in_vectors) {}

		// Unguarded tex vector
		template <typename Vector>
		static __device__ __forceinline__ void LoadTexVector(
			const int STRIP,
			Vector data_vectors[],
			texture<Vector, cudaTextureType1D, cudaReadModeElementType> ref,
			unsigned int base_thread_offset) {}

		// Guarded singleton
		template <typename Flag, typename T, typename S, int THREAD_STRIP_ELEMENTS, typename SizeT, typename TransformOp>
		static __device__ __forceinline__ void LoadGuarded(
			const int STRIP,
			Flag (&valid)[THREAD_STRIP_ELEMENTS],
			T (&data)[THREAD_STRIP_ELEMENTS],
			S *d_in,
			const SizeT &guarded_elements,
			TransformOp transform_op) {}

		// Initialize valid flag within an unguarded segment
		template <typename T, typename S, typename TransformOp>
		static __device__ __forceinline__ void TransformRaw(
			T data[],
			S raw[],
			TransformOp transform_op) {}

		// Segment of unguarded vectors
		template <typename Vector, int VECTORS>
		static __device__ __forceinline__ void LoadVectorSegment(
			Vector data_vectors[][VECTORS],
			Vector *d_in_vectors) {}

		// Segment of unguarded tex vectors
		template <typename T, typename S, int THREAD_STRIP_ELEMENTS, typename Vector, typename SizeT, int VECTORS>
		static __device__ __forceinline__ void LoadTexVectorSegment(
			T data[][THREAD_STRIP_ELEMENTS],
			S raw[][THREAD_STRIP_ELEMENTS],
			Vector data_vectors[][VECTORS],
			texture<Vector, cudaTextureType1D, cudaReadModeElementType> ref,
			SizeT base_thread_offset) {}

		// Segment of guarded singletons
		template <typename Flag, typename T, typename S, int THREAD_STRIP_ELEMENTS, typename SizeT, typename TransformOp>
		static __device__ __forceinline__ void LoadSegmentGuarded(
			Flag valid[][THREAD_STRIP_ELEMENTS],
			T data[][THREAD_STRIP_ELEMENTS],
			S *d_in,
			const SizeT &guarded_elements,
			TransformOp transform_op) {}
	};

public:

	//---------------------------------------------------------------------
	// Unguarded tile interface
	//---------------------------------------------------------------------

	/**
	 * Load a full, strip-mined tile
	 */
	template <
		int CTA_STRIPS,
		int THREAD_STRIP_ELEMENTS,
		typename T,
		typename S,
		typename SizeT,
		typename TransformOp>
	static __device__ __forceinline__ void LoadUnguarded(
		T (&data)[CTA_STRIPS][THREAD_STRIP_ELEMENTS],
		S *d_in,
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

		// Raw data to load
		S raw[CTA_STRIPS][THREAD_STRIP_ELEMENTS];

		// Alias pointers
		Vector (*data_vectors)[VECTORS] =
			reinterpret_cast<Vector (*)[VECTORS]>(raw);

		Vector *d_in_vectors =
			reinterpret_cast<Vector *>(d_in + (threadIdx.x * THREAD_STRIP_ELEMENTS) + cta_offset);

		// Load raw data
		Iterate<0, CTA_STRIPS>::LoadVectorSegment(
			data_vectors,
			d_in_vectors);

		// Transform from raw and initialize valid
		Iterate<0, CTA_STRIPS * THREAD_STRIP_ELEMENTS>::TransformRaw(
			(T*) data,
			(S*) raw,
			transform_op);
	}


	/**
	 * Load a full tile
	 */
	template <
		typename T,
		int THREAD_ELEMENTS,
		typename S,
		typename SizeT,
		typename TransformOp>
	static __device__ __forceinline__ void LoadUnguarded(
		T (&data)[THREAD_ELEMENTS],
		S *d_in,
		SizeT cta_offset,
		TransformOp transform_op)
	{
		T (&data_2d)[1][THREAD_ELEMENTS] = reinterpret_cast<T (&)[1][THREAD_ELEMENTS]>(data);

		return LoadUnguarded(data_2d, d_in, cta_offset, transform_op);
	}


	/**
	 * Load a full, strip-mined tile
	 */
	template <
		typename T,
		int CTA_STRIPS,
		int THREAD_STRIP_ELEMENTS,
		typename S,
		typename SizeT>
	static __device__ __forceinline__ void LoadUnguarded(
		T (&data)[CTA_STRIPS][THREAD_STRIP_ELEMENTS],
		S *d_in,
		SizeT cta_offset)
	{
		CastTransformOp<T, S> transform_op;
		LoadUnguarded(data, d_in, cta_offset, transform_op);
	}


	/**
	 * Load a full tile
	 */
	template <
		typename T,
		int THREAD_ELEMENTS,
		typename S,
		typename SizeT>
	static __device__ __forceinline__ void LoadUnguarded(
		T (&data)[THREAD_ELEMENTS],
		S *d_in,
		SizeT cta_offset)
	{
		CastTransformOp<T, S> transform_op;
		LoadUnguarded(data, d_in, cta_offset, transform_op);
	}


	/**
	 * Load a full, strip-mined tile (optionally using tex if READ_MODIFIER == ld::tex)
	 */
	template <
		typename T,
		int CTA_STRIPS,
		int THREAD_STRIP_ELEMENTS,
		typename Vector,
		typename S,
		typename SizeT,
		typename TransformOp>
	static __device__ __forceinline__ void LoadUnguarded(
		T (&data)[CTA_STRIPS][THREAD_STRIP_ELEMENTS],
		texture<Vector, cudaTextureType1D, cudaReadModeElementType> ref,
		S *d_in,
		SizeT cta_offset,
		TransformOp transform_op)
	{
		if (MODIFIER == LOAD_TEX)
		{
			// Use tex
			const int VEC_ELEMENTS 		= sizeof(Vector) / sizeof(S);
			const int VECTORS 			= THREAD_STRIP_ELEMENTS / (sizeof(Vector) / sizeof(S));

			// Data to load
			S raw[CTA_STRIPS][THREAD_STRIP_ELEMENTS];

			// Use an aliased pointer to raw array
			Vector (*data_vectors)[VECTORS] = (Vector (*)[VECTORS]) raw;

			SizeT base_thread_offset = cta_offset / VEC_ELEMENTS;

			Iterate<0, CTA_STRIPS>::LoadTexVectorSegment(
				data,
				raw,
				data_vectors,
				ref,
				base_thread_offset + (threadIdx.x * VECTORS));

			// Transform raw and initialize
			Iterate<0, CTA_STRIPS * THREAD_STRIP_ELEMENTS>::TransformRaw(
				(T*) data,
				(S*) raw,
				transform_op);
		}
		else
		{
			// Use normal loads
			LoadUnguarded(data, d_in, cta_offset, transform_op);
		}
	}


	/**
	 * Load a full tile (optionally using tex if READ_MODIFIER == ld::tex)
	 */
	template <
		typename T,
		int THREAD_ELEMENTS,
		typename Vector,
		typename S,
		typename SizeT,
		typename TransformOp>
	static __device__ __forceinline__ void LoadUnguarded(
		T (&data)[THREAD_ELEMENTS],
		texture<Vector, cudaTextureType1D, cudaReadModeElementType> ref,
		S *d_in,
		SizeT cta_offset,
		TransformOp transform_op)
	{
		T (&data_2d)[1][THREAD_ELEMENTS] = reinterpret_cast<T (&)[1][THREAD_ELEMENTS]>(data);

		return LoadUnguarded(data_2d, ref, d_in, cta_offset, transform_op);
	}


	/**
	 * Load a full, strip-mined tile (optionally using tex if READ_MODIFIER == ld::tex)
	 */
	template <
		typename T,
		int CTA_STRIPS,
		int THREAD_STRIP_ELEMENTS,
		typename Vector,
		typename S,
		typename SizeT>
	static __device__ __forceinline__ void LoadUnguarded(
		T (&data)[CTA_STRIPS][THREAD_STRIP_ELEMENTS],
		texture<Vector, cudaTextureType1D, cudaReadModeElementType> ref,
		S *d_in,
		SizeT cta_offset)
	{
		CastTransformOp<T, S> transform_op;
		LoadUnguarded(data, ref, d_in, cta_offset, transform_op);
	}


	/**
	 * Load a full tile (optionally using tex if READ_MODIFIER == ld::tex)
	 */
	template <
		typename T,
		int THREAD_ELEMENTS,
		typename Vector,
		typename S,
		typename SizeT>
	static __device__ __forceinline__ void LoadUnguarded(
		T (&data)[THREAD_ELEMENTS],
		texture<Vector, cudaTextureType1D, cudaReadModeElementType> ref,
		S *d_in,
		SizeT cta_offset)
	{
		CastTransformOp<T, S> transform_op;
		LoadUnguarded(data, ref, d_in, cta_offset, transform_op);
	}


	//---------------------------------------------------------------------
	// Guarded tile interface
	//---------------------------------------------------------------------

	/**
	 * Load a strip-mined tile, guarded by range
	 */
	template <
		typename Flag,
		int CTA_STRIPS,
		int THREAD_STRIP_ELEMENTS,
		typename T,
		typename S,
		typename SizeT,
		typename TransformOp>
	static __device__ __forceinline__ void LoadGuarded(
		Flag (&valid)[CTA_STRIPS][THREAD_STRIP_ELEMENTS],
		T (&data)[CTA_STRIPS][THREAD_STRIP_ELEMENTS],
		S *d_in,
		SizeT cta_offset,
		const SizeT &guarded_elements,
		TransformOp transform_op)
	{
		Iterate<0, CTA_STRIPS>::LoadSegmentGuarded(
			valid,
			data,
			d_in + (threadIdx.x * THREAD_STRIP_ELEMENTS) + cta_offset,
			guarded_elements,
			transform_op);
	}


	/**
	 * Load a tile, guarded by range
	 */
	template <
		typename Flag,
		int THREAD_ELEMENTS,
		typename T,
		typename S,
		typename SizeT,
		typename TransformOp>
	static __device__ __forceinline__ void LoadGuarded(
		Flag (&valid)[THREAD_ELEMENTS],
		T (&data)[THREAD_ELEMENTS],
		S *d_in,
		SizeT cta_offset,
		const SizeT &guarded_elements,
		TransformOp transform_op)
	{
		Flag (&valid_2d)[1][THREAD_ELEMENTS] = reinterpret_cast<Flag (&)[1][THREAD_ELEMENTS]>(valid);
		T (&data_2d)[1][THREAD_ELEMENTS] = reinterpret_cast<T (&)[1][THREAD_ELEMENTS]>(data);

		LoadGuarded(valid_2d, data_2d, d_in, cta_offset, guarded_elements, transform_op);
	}


	/**
	 * Load a strip-mined tile, guarded by range
	 */
	template <
		typename Flag,
		int CTA_STRIPS,
		int THREAD_STRIP_ELEMENTS,
		typename T,
		typename S,
		typename SizeT>
	static __device__ __forceinline__ void LoadGuarded(
		Flag (&valid)[CTA_STRIPS][THREAD_STRIP_ELEMENTS],
		T (&data)[CTA_STRIPS][THREAD_STRIP_ELEMENTS],
		S *d_in,
		SizeT cta_offset,
		const SizeT &guarded_elements)
	{
		CastTransformOp<T, S> transform_op;
		LoadGuarded(valid, data, d_in, cta_offset, guarded_elements, transform_op);
	}


	/**
	 * Load a tile, guarded by range
	 */
	template <
		typename Flag,
		int THREAD_ELEMENTS,
		typename T,
		typename S,
		typename SizeT>
	static __device__ __forceinline__ void LoadGuarded(
		Flag (&valid)[THREAD_ELEMENTS],
		T (&data)[THREAD_ELEMENTS],
		S *d_in,
		SizeT cta_offset,
		const SizeT &guarded_elements)
	{
		CastTransformOp<T, S> transform_op;
		LoadGuarded(valid, data, d_in, cta_offset, guarded_elements);
	}


	/**
	 * Load a strip-mined tile, guarded by range
	 */
	template <
		typename T,
		int CTA_STRIPS,
		int THREAD_STRIP_ELEMENTS,
		typename S,
		typename SizeT>
	static __device__ __forceinline__ void LoadGuarded(
		T (&data)[CTA_STRIPS][THREAD_STRIP_ELEMENTS],
		S *d_in,
		SizeT cta_offset,
		const SizeT &guarded_elements)
	{
		int valid[CTA_STRIPS][THREAD_STRIP_ELEMENTS];
		CastTransformOp<T, S> transform_op;
		LoadGuarded(valid, data, d_in, cta_offset, guarded_elements, transform_op);
	}


	/**
	 * Load a tile, guarded by range
	 */
	template <
		typename T,
		int THREAD_ELEMENTS,
		typename S,
		typename SizeT>
	static __device__ __forceinline__ void LoadGuarded(
		T (&data)[THREAD_ELEMENTS],
		S *d_in,
		SizeT cta_offset,
		const SizeT &guarded_elements)
	{
		int valid[THREAD_ELEMENTS];
		CastTransformOp<T, S> transform_op;
		LoadGuarded(valid, data, d_in, cta_offset, guarded_elements, transform_op);
	}
};




} // namespace cub
CUB_NS_POSTFIX
