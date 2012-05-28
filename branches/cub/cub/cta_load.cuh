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

#include <cub/basic_utils.cuh>
#include <cub/operators.cuh>
#include <cub/type_utils.cuh>
#include <cub/thread_load.cuh>
#include <cub/ns_umbrella.cuh>

CUB_NS_PREFIX
namespace cub {

/**
 * Load a tile of items
 */
template <
	int CTA_THREADS,						// Active threads that will be loading
	LoadModifier MODIFIER = LOAD_NONE>		// Cache modifier (e.g., TEX/CA/CG/CS/NONE/etc.)
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
			const int SEGMENT,
			Vector data_vectors[],
			Vector *d_in_vectors)
		{
			const int OFFSET = (SEGMENT * CTA_THREADS * TOTAL) + CURRENT;

			data_vectors[CURRENT] = ThreadLoad<MODIFIER>(d_in_vectors + OFFSET);

			// Next vector in segment
			Iterate<CURRENT + 1, TOTAL>::LoadVector(
				SEGMENT,
				data_vectors,
				d_in_vectors);
		}

		// Unguarded tex vector
		template <typename Vector>
		static __device__ __forceinline__ void LoadTexVector(
			const int SEGMENT,
			Vector data_vectors[],
			texture<Vector, cudaTextureType1D, cudaReadModeElementType> ref,
			unsigned int base_thread_offset)
		{
			const int OFFSET = (SEGMENT * CTA_THREADS * TOTAL) + CURRENT;

			data_vectors[CURRENT] = tex1Dfetch(
				ref,
				base_thread_offset + OFFSET);

			// Next vector in segment
			Iterate<CURRENT + 1, TOTAL>::LoadTexVector(
				SEGMENT,
				data_vectors,
				ref,
				base_thread_offset);
		}


		// Guarded singleton
		template <
			typename Flag,
			typename T,
			typename S,
			int ELEMENTS,
			typename SizeT,
			typename TransformOp>
		static __device__ __forceinline__ void LoadGuarded(
			const int SEGMENT,
			Flag (&valid)[ELEMENTS],
			T (&data)[ELEMENTS],
			S *d_in,
			const SizeT &guarded_elements,
			TransformOp transform_op)
		{
			const int OFFSET = (SEGMENT * CTA_THREADS * TOTAL) + CURRENT;

			valid[CURRENT] = ((threadIdx.x * ELEMENTS) + OFFSET) < guarded_elements;

			if (valid[CURRENT])
			{
				// Load and transform
				S raw_data = ThreadLoad<MODIFIER>(d_in + OFFSET);
				data[CURRENT] = transform_op(raw_data);
			}

			// Next load in segment
			Iterate<CURRENT + 1, TOTAL>::LoadGuarded(
				SEGMENT,
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
			int ELEMENTS,
			typename Vector,
			typename SizeT,
			int VECTORS>
		static __device__ __forceinline__ void LoadTexVectorSegment(
			T data[][ELEMENTS],
			S raw[][ELEMENTS],
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
			int ELEMENTS,
			typename SizeT,
			typename TransformOp>
		static __device__ __forceinline__ void LoadSegmentGuarded(
			Flag valid[][ELEMENTS],
			T data[][ELEMENTS],
			S *d_in,
			const SizeT &guarded_elements,
			TransformOp transform_op)
		{
			// Perform guarded, transforming raking vector loads for this segment
			Iterate<0, ELEMENTS>::LoadGuarded(
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
			const int SEGMENT,
			Vector data_vectors[],
			Vector *d_in_vectors) {}

		// Unguarded tex vector
		template <typename Vector>
		static __device__ __forceinline__ void LoadTexVector(
			const int SEGMENT,
			Vector data_vectors[],
			texture<Vector, cudaTextureType1D, cudaReadModeElementType> ref,
			unsigned int base_thread_offset) {}

		// Guarded singleton
		template <typename Flag, typename T, typename S, int ELEMENTS, typename SizeT, typename TransformOp>
		static __device__ __forceinline__ void LoadGuarded(
			const int SEGMENT,
			Flag (&valid)[ELEMENTS],
			T (&data)[ELEMENTS],
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
		template <typename T, typename S, int ELEMENTS, typename Vector, typename SizeT, int VECTORS>
		static __device__ __forceinline__ void LoadTexVectorSegment(
			T data[][ELEMENTS],
			S raw[][ELEMENTS],
			Vector data_vectors[][VECTORS],
			texture<Vector, cudaTextureType1D, cudaReadModeElementType> ref,
			SizeT base_thread_offset) {}

		// Segment of guarded singletons
		template <typename Flag, typename T, typename S, int ELEMENTS, typename SizeT, typename TransformOp>
		static __device__ __forceinline__ void LoadSegmentGuarded(
			Flag valid[][ELEMENTS],
			T data[][ELEMENTS],
			S *d_in,
			const SizeT &guarded_elements,
			TransformOp transform_op) {}
	};

public:

	//---------------------------------------------------------------------
	// Unguarded tile interface
	//---------------------------------------------------------------------

	/**
	 * Load a tile unguarded
	 */
	template <
		int SEGMENTS,
		int ELEMENTS,
		typename T,
		typename S,
		typename SizeT,
		typename TransformOp>
	static __device__ __forceinline__ void LoadUnguarded(
		T (&data)[SEGMENTS][ELEMENTS],
		S *d_in,
		SizeT cta_offset,
		TransformOp transform_op)
	{
		enum
		{
			MAX_VEC_ELEMENTS 	= CUB_MIN(MAX_VEC_ELEMENTS, ELEMENTS),
			VEC_ELEMENTS		= (ELEMENTS % MAX_VEC_ELEMENTS == 0) ? MAX_VEC_ELEMENTS : 1,	// Elements must be an even multiple of vector size
			VECTORS 			= ELEMENTS / VEC_ELEMENTS,
		};

		typedef typename VectorType<S, VEC_ELEMENTS>::Type Vector;

		// Raw data to load
		S raw[SEGMENTS][ELEMENTS];

		// Alias pointers
		Vector (*data_vectors)[VECTORS] =
			reinterpret_cast<Vector (*)[VECTORS]>(raw);

		Vector *d_in_vectors =
			reinterpret_cast<Vector *>(d_in + (threadIdx.x * ELEMENTS) + cta_offset);

		// Load raw data
		Iterate<0, SEGMENTS>::LoadVectorSegment(
			data_vectors,
			d_in_vectors);

		// Transform from raw and initialize valid
		Iterate<0, SEGMENTS * ELEMENTS>::TransformRaw(
			(T*) data,
			(S*) raw,
			transform_op);
	}


	/**
	 * Load a tile unguarded
	 */
	template <
		int ELEMENTS,
		typename T,
		typename S,
		typename SizeT,
		typename TransformOp>
	static __device__ __forceinline__ void LoadUnguarded(
		T (&data)[ELEMENTS],
		S *d_in,
		SizeT cta_offset,
		TransformOp transform_op)
	{
		T (&data_2d)[1][ELEMENTS] = reinterpret_cast<T (&)[1][ELEMENTS]>(data);

		return LoadUnguarded(data_2d, d_in, cta_offset, transform_op);
	}


	/**
	 * Load a unguarded tile
	 */
	template <
		typename T,
		int SEGMENTS,
		int ELEMENTS,
		typename S,
		typename SizeT>
	static __device__ __forceinline__ void LoadUnguarded(
		T (&data)[SEGMENTS][ELEMENTS],
		S *d_in,
		SizeT cta_offset)
	{
		CastTransformOp<T, S> transform_op;
		LoadUnguarded(data, d_in, cta_offset, transform_op);
	}


	/**
	 * Load a unguarded tile
	 */
	template <
		typename T,
		int ELEMENTS,
		typename S,
		typename SizeT>
	static __device__ __forceinline__ void LoadUnguarded(
		T (&data)[ELEMENTS],
		S *d_in,
		SizeT cta_offset)
	{
		CastTransformOp<T, S> transform_op;
		LoadUnguarded(data, d_in, cta_offset, transform_op);
	}


	/**
	 * Load a unguarded tile (optionally using tex if READ_MODIFIER == ld::tex)
	 */
	template <
		int SEGMENTS,
		int ELEMENTS,
		typename T,
		typename Vector,
		typename S,
		typename SizeT,
		typename TransformOp>
	static __device__ __forceinline__ void LoadUnguarded(
		T (&data)[SEGMENTS][ELEMENTS],
		texture<Vector, cudaTextureType1D, cudaReadModeElementType> ref,
		S *d_in,
		SizeT cta_offset,
		TransformOp transform_op)
	{
		if (MODIFIER == LOAD_TEX)
		{
			// Use tex
			const int VEC_ELEMENTS 		= sizeof(Vector) / sizeof(S);
			const int VECTORS 			= ELEMENTS / (sizeof(Vector) / sizeof(S));

			// Data to load
			S raw[SEGMENTS][ELEMENTS];

			// Use an aliased pointer to raw array
			Vector (*data_vectors)[VECTORS] = (Vector (*)[VECTORS]) raw;

			SizeT base_thread_offset = cta_offset / VEC_ELEMENTS;

			Iterate<0, SEGMENTS>::LoadTexVectorSegment(
				data,
				raw,
				data_vectors,
				ref,
				base_thread_offset + (threadIdx.x * VECTORS));

			// Transform raw and initialize
			Iterate<0, SEGMENTS * ELEMENTS>::TransformRaw(
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
	 * Load a unguarded tile (optionally using tex if READ_MODIFIER == ld::tex)
	 */
	template <
		int ELEMENTS,
		typename T,
		typename Vector,
		typename S,
		typename SizeT,
		typename TransformOp>
	static __device__ __forceinline__ void LoadUnguarded(
		T (&data)[ELEMENTS],
		texture<Vector, cudaTextureType1D, cudaReadModeElementType> ref,
		S *d_in,
		SizeT cta_offset,
		TransformOp transform_op)
	{
		T (&data_2d)[1][ELEMENTS] = reinterpret_cast<T (&)[1][ELEMENTS]>(data);

		return LoadUnguarded(data_2d, ref, d_in, cta_offset, transform_op);
	}


	/**
	 * Load a unguarded tile (optionally using tex if READ_MODIFIER == ld::tex)
	 */
	template <
		typename T,
		int SEGMENTS,
		int ELEMENTS,
		typename Vector,
		typename S,
		typename SizeT>
	static __device__ __forceinline__ void LoadUnguarded(
		T (&data)[SEGMENTS][ELEMENTS],
		texture<Vector, cudaTextureType1D, cudaReadModeElementType> ref,
		S *d_in,
		SizeT cta_offset)
	{
		CastTransformOp<T, S> transform_op;
		LoadUnguarded(data, ref, d_in, cta_offset, transform_op);
	}


	/**
	 * Load a unguarded tile (optionally using tex if READ_MODIFIER == ld::tex)
	 */
	template <
		typename T,
		int ELEMENTS,
		typename Vector,
		typename S,
		typename SizeT>
	static __device__ __forceinline__ void LoadUnguarded(
		T (&data)[ELEMENTS],
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
	 * Load a guarded tile
	 */
	template <
		typename Flag,
		int SEGMENTS,
		int ELEMENTS,
		typename T,
		typename S,
		typename SizeT,
		typename TransformOp>
	static __device__ __forceinline__ void LoadGuarded(
		Flag (&valid)[SEGMENTS][ELEMENTS],
		T (&data)[SEGMENTS][ELEMENTS],
		S *d_in,
		SizeT cta_offset,
		const SizeT &guarded_elements,
		TransformOp transform_op)
	{
		Iterate<0, SEGMENTS>::LoadSegmentGuarded(
			valid,
			data,
			d_in + (threadIdx.x * ELEMENTS) + cta_offset,
			guarded_elements,
			transform_op);
	}


	/**
	 * Load a guarded tile
	 */
	template <
		typename Flag,
		int ELEMENTS,
		typename T,
		typename S,
		typename SizeT,
		typename TransformOp>
	static __device__ __forceinline__ void LoadGuarded(
		Flag (&valid)[ELEMENTS],
		T (&data)[ELEMENTS],
		S *d_in,
		SizeT cta_offset,
		const SizeT &guarded_elements,
		TransformOp transform_op)
	{
		Flag (&valid_2d)[1][ELEMENTS] = reinterpret_cast<Flag (&)[1][ELEMENTS]>(valid);
		T (&data_2d)[1][ELEMENTS] = reinterpret_cast<T (&)[1][ELEMENTS]>(data);

		LoadGuarded(valid_2d, data_2d, d_in, cta_offset, guarded_elements, transform_op);
	}


	/**
	 * Load a guarded tile
	 */
	template <
		typename Flag,
		int SEGMENTS,
		int ELEMENTS,
		typename T,
		typename S,
		typename SizeT>
	static __device__ __forceinline__ void LoadGuarded(
		Flag (&valid)[SEGMENTS][ELEMENTS],
		T (&data)[SEGMENTS][ELEMENTS],
		S *d_in,
		SizeT cta_offset,
		const SizeT &guarded_elements)
	{
		CastTransformOp<T, S> transform_op;
		LoadGuarded(valid, data, d_in, cta_offset, guarded_elements, transform_op);
	}


	/**
	 * Load a guarded tile
	 */
	template <
		typename Flag,
		int ELEMENTS,
		typename T,
		typename S,
		typename SizeT>
	static __device__ __forceinline__ void LoadGuarded(
		Flag (&valid)[ELEMENTS],
		T (&data)[ELEMENTS],
		S *d_in,
		SizeT cta_offset,
		const SizeT &guarded_elements)
	{
		CastTransformOp<T, S> transform_op;
		LoadGuarded(valid, data, d_in, cta_offset, guarded_elements);
	}


	/**
	 * Load a guarded tile
	 */
	template <
		typename T,
		int SEGMENTS,
		int ELEMENTS,
		typename S,
		typename SizeT>
	static __device__ __forceinline__ void LoadGuarded(
		T (&data)[SEGMENTS][ELEMENTS],
		S *d_in,
		SizeT cta_offset,
		const SizeT &guarded_elements)
	{
		int valid[SEGMENTS][ELEMENTS];
		CastTransformOp<T, S> transform_op;
		LoadGuarded(valid, data, d_in, cta_offset, guarded_elements, transform_op);
	}


	/**
	 * Load a guarded tile
	 */
	template <
		typename T,
		int ELEMENTS,
		typename S,
		typename SizeT>
	static __device__ __forceinline__ void LoadGuarded(
		T (&data)[ELEMENTS],
		S *d_in,
		SizeT cta_offset,
		const SizeT &guarded_elements)
	{
		int valid[ELEMENTS];
		CastTransformOp<T, S> transform_op;
		LoadGuarded(valid, data, d_in, cta_offset, guarded_elements, transform_op);
	}
};




} // namespace cub
CUB_NS_POSTFIX
