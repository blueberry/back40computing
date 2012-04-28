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
 * Cooperative load abstraction for CTAs.
 ******************************************************************************/

#pragma once

#include <cub/operators.cuh>
#include <cub/vector_types.cuh>
#include <cub/thread/load.cuh>

namespace cub {


/**
 * Texture vector types for reading ELEMENTS consecutive elements of T per thread
 */
template <typename T, int ELEMENTS>
struct TexVectorT
{
	enum {
		TEX_VEC_SIZE = (NumericTraits<T>::BUILT_IN) ?
			1 : 								// vec-1 for non-built-ins (don't actually use!)
			(sizeof(T) > 4) ?
				(ELEMENTS % 2 == 1) ?			// 64-bit built-in types
					2 : 								// cast as vec-2 ints (odd)
					4 :									// cast as vec-4 ints (multiple of two)
				(ELEMENTS % 2 == 1) ?			// 32-bit built-in types
					1 : 								// vec-1 (odd)
					(ELEMENTS % 4 == 0) ?
						4 :								// vec-4 (multiple of 4)
						2,								// vec-2 (multiple of 2)
	};

	// Texture base type
	typedef typename If<(NumericTraits<T>::BUILT_IN),
		char,										// use char for non-built-ins (don't actually use!)
		typename If<(sizeof(T) > 4),
			int,									// use int for 64-bit built-in types
			T>::Type>::Type TexBase; 				// use T for other built-in types

	// Texture vector type
	typedef typename util::VectorT<TexBase, TEX_VEC_SIZE>::Type TexVec;

	// Texture reference type
	typedef texture<TexVec, cudaTextureType1D, cudaReadModeElementType> TexRef;
};



/**
 * Load a tile of items
 */
template <
	int ACTIVE_THREADS,								// Active threads that will be loading
	ld::CacheModifier CACHE_MODIFIER = ld::NONE>	// Cache modifier (e.g., TEX/CA/CG/CS/NONE/etc.)
class TileReader
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
		template <typename VectorTType>
		static __device__ __forceinline__ void LoadVectorT(
			const int SEGMENT,
			VectorTType data_vectors[],
			VectorTType *d_in_vectors)
		{
			const int OFFSET = (SEGMENT * ACTIVE_THREADS * TOTAL) + CURRENT;

			ModifiedLoad<CACHE_MODIFIER>::Ld(
				data_vectors[CURRENT],
				d_in_vectors + OFFSET);

			// Next vector in segment
			Iterate<CURRENT + 1, TOTAL>::LoadVectorT(
				SEGMENT,
				data_vectors,
				d_in_vectors);
		}

		// Unguarded tex vector
		template <typename VectorTType>
		static __device__ __forceinline__ void LoadTexVectorT(
			const int SEGMENT,
			VectorTType data_vectors[],
			texture<VectorTType, cudaTextureType1D, cudaReadModeElementType> ref,
			unsigned int base_thread_offset)
		{
			const int OFFSET = (SEGMENT * ACTIVE_THREADS * TOTAL) + CURRENT;

			data_vectors[CURRENT] = tex1Dfetch(
				ref,
				base_thread_offset + OFFSET);

			// Next vector in segment
			Iterate<CURRENT + 1, TOTAL>::LoadTexVectorT(
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
			const int OFFSET = (SEGMENT * ACTIVE_THREADS * TOTAL) + CURRENT;

			valid[CURRENT] = ((threadIdx.x * ELEMENTS) + OFFSET < guarded_elements);

			if (valid[CURRENT]) {
				// Load and transform
				S raw_data[1];
				ModifiedLoad<CACHE_MODIFIER>::Ld(raw_data[0], d_in + OFFSET);
				data[CURRENT] = transform_op(raw_data[0]);
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
			typename Flag,
			typename TransformOp>
		static __device__ __forceinline__ void TransformRaw(
			Flag valid[],
			T data[],
			S raw[],
			TransformOp transform_op)
		{
			valid[CURRENT] = 1;
			data[CURRENT] = transform_op(raw[CURRENT]);

			// Next load in segment
			Iterate<CURRENT + 1, TOTAL>::TransformRaw(
				valid,
				data,
				raw,
				transform_op);
		}


		//---------------------------------------------------------------------
		// Load strided segments
		//---------------------------------------------------------------------

		// Segment of unguarded vectors
		template <
			typename VectorTType,
			int VECTORS>
		static __device__ __forceinline__ void LoadVectorTSegment(
			VectorTType data_vectors[][VECTORS],
			VectorTType *d_in_vectors)
		{
			// Perform raking vector loads for this segment
			Iterate<0, VECTORS>::LoadVectorT(
				CURRENT,
				data_vectors[CURRENT],
				d_in_vectors);

			// Next segment
			Iterate<CURRENT + 1, TOTAL>::LoadVectorTSegment(
				data_vectors,
				d_in_vectors);
		}

		// Segment of unguarded tex vectors
		template <
			typename T,
			typename S,
			int ELEMENTS,
			typename VectorTType,
			typename SizeT,
			int VECTORS>
		static __device__ __forceinline__ void LoadTexVectorTSegment(
			T data[][ELEMENTS],
			S raw[][ELEMENTS],
			VectorTType data_vectors[][VECTORS],
			texture<VectorTType, cudaTextureType1D, cudaReadModeElementType> ref,
			SizeT base_thread_offset)
		{
			// Perform raking tex vector loads for this segment
			Iterate<0, VECTORS>::LoadTexVectorT(
				CURRENT,
				data_vectors[CURRENT],
				ref,
				base_thread_offset);

			// Next segment
			Iterate<CURRENT + 1, TOTAL>::LoadTexVectorTSegment(
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
		template <typename VectorTType>
		static __device__ __forceinline__ void LoadVectorT(
			const int SEGMENT,
			VectorTType data_vectors[],
			VectorTType *d_in_vectors) {}

		// Unguarded tex vector
		template <typename VectorTType>
		static __device__ __forceinline__ void LoadTexVectorT(
			const int SEGMENT,
			VectorTType data_vectors[],
			texture<VectorTType, cudaTextureType1D, cudaReadModeElementType> ref,
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
		template <typename T, typename S, typename Flag, typename TransformOp>
		static __device__ __forceinline__ void TransformRaw(
			Flag valid[],
			T data[],
			S raw[],
			TransformOp transform_op) {}

		// Segment of unguarded vectors
		template <typename VectorTType, int VECTORS>
		static __device__ __forceinline__ void LoadVectorTSegment(
			VectorTType data_vectors[][VECTORS],
			VectorTType *d_in_vectors) {}

		// Segment of unguarded tex vectors
		template <typename T, typename S, int ELEMENTS, typename VectorTType, typename SizeT, int VECTORS>
		static __device__ __forceinline__ void LoadTexVectorTSegment(
			T data[][ELEMENTS],
			S raw[][ELEMENTS],
			VectorTType data_vectors[][VECTORS],
			texture<VectorTType, cudaTextureType1D, cudaReadModeElementType> ref,
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
		typename Flag,
		int SEGMENTS,
		int ELEMENTS,
		typename T,
		typename S,
		typename SizeT,
		typename TransformOp>
	static __device__ __forceinline__ void LoadUnguarded(
		Flag (&valid)[SEGMENTS][ELEMENTS],
		T (&data)[SEGMENTS][ELEMENTS],
		S *d_in,
		SizeT cta_offset,
		TransformOp transform_op)
	{
		const int VEC_ELEMENTS 		= B40C_MIN(MAX_VEC_ELEMENTS, ELEMENTS);
		const int VECTORS 			= ELEMENTS / (B40C_MIN(MAX_VEC_ELEMENTS, ELEMENTS));

		typedef typename VectorT<S, VEC_ELEMENTS>::Type VectorTType;

		// Raw data to load
		S raw[SEGMENTS][ELEMENTS];

		// Alias pointers
		VectorTType (*data_vectors)[VECTORS] =
			reinterpret_cast<VectorTType (*)[VECTORS]>(raw);

		VectorTType *d_in_vectors =
			reinterpret_cast<VectorTType *>(d_in + (threadIdx.x * ELEMENTS) + cta_offset);

		// Load raw data
		Iterate<0, SEGMENTS>::LoadVectorTSegment(
			data_vectors,
			d_in_vectors);

		// Transform from raw and initialize valid
		Iterate<0, SEGMENTS * ELEMENTS>::TransformRaw(
			(Flag*) valid,
			(T*) data,
			(S*) raw,
			transform_op);
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
		int valid[SEGMENTS][ELEMENTS];
		CastTransformOp<T, S> transform_op;

		LoadUnguarded(
			valid,
			data,
			d_in,
			cta_offset,
			transform_op);
	}


	/**
	 * Load a unguarded tile (optionally using tex if READ_MODIFIER == ld::tex)
	 */
	template <
		typename Flag,
		int SEGMENTS,
		int ELEMENTS,
		typename T,
		typename VectorTType,
		typename S,
		typename SizeT,
		typename TransformOp>
	static __device__ __forceinline__ void LoadUnguarded(
		Flag (&valid)[SEGMENTS][ELEMENTS],
		T (&data)[SEGMENTS][ELEMENTS],
		texture<VectorTType, cudaTextureType1D, cudaReadModeElementType> ref,
		S *d_in,
		SizeT cta_offset,
		TransformOp transform_op)
	{
		if (CACHE_MODIFIER == ld::tex) {

			// Use tex
			const int VEC_ELEMENTS 		= sizeof(VectorTType) / sizeof(S);
			const int VECTORS 			= ELEMENTS / (sizeof(VectorTType) / sizeof(S));

			// Data to load
			S raw[SEGMENTS][ELEMENTS];

			// Use an aliased pointer to raw array
			VectorTType (*data_vectors)[VECTORS] = (VectorTType (*)[VECTORS]) raw;

			SizeT base_thread_offset = cta_offset / VEC_ELEMENTS;

			Iterate<0, SEGMENTS>::LoadTexVectorTSegment(
				data,
				raw,
				data_vectors,
				ref,
				base_thread_offset + (threadIdx.x * VECTORS));

			// Transform raw and initialize
			Iterate<0, SEGMENTS * ELEMENTS>::TransformRaw(
				(Flag*) valid,
				(T*) data,
				(S*) raw,
				transform_op);

		} else {

			// Use normal loads
			LoadUnguarded(
				valid,
				data,
				d_in,
				cta_offset,
				transform_op);
		}
	}


	/**
	 * Load a unguarded tile (optionally using tex if READ_MODIFIER == ld::tex)
	 */
	template <
		typename T,
		int SEGMENTS,
		int ELEMENTS,
		typename VectorTType,
		typename S,
		typename SizeT>
	static __device__ __forceinline__ void LoadUnguarded(
		T (&data)[SEGMENTS][ELEMENTS],
		texture<VectorTType, cudaTextureType1D, cudaReadModeElementType> ref,
		S *d_in,
		SizeT cta_offset)
	{
		int valid[SEGMENTS][ELEMENTS];
		CastTransformOp<T, S> transform_op;

		LoadUnguarded(
			valid,
			data,
			ref,
			d_in,
			cta_offset,
			transform_op);
	}


	/**
	 * Load a single value
	 */
	template <typename T, typename S, typename SizeT, typename TransformOp>
	static __device__ __forceinline__ void LoadUnguarded(
		T &datum,
		S *d_in,
		SizeT cta_offset,
		TransformOp op)
	{
		S raw;
		ModifiedLoad<CACHE_MODIFIER>::Ld(raw, d_in + threadIdx.x + cta_offset);
		datum = op(raw);
	}


	/**
	 * Load a single value
	 */
	template <typename T, typename S, typename SizeT>
	static __device__ __forceinline__ void LoadUnguarded(
		T &datum,
		S *d_in,
		SizeT cta_offset)
	{
		CastTransformOp<T, S> transform_op;
		LoadUnguarded(datum, d_in, cta_offset, transform_op);
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

		LoadGuarded(
			valid,
			data,
			d_in,
			cta_offset,
			guarded_elements,
			transform_op);
	}

};



} // namespace cub

