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
 * CTA abstraction for loading tiles of items
 ******************************************************************************/

#pragma once

#include <iterator>

#include "../ns_wrapper.cuh"
#include "../macro_utils.cuh"
#include "../thread/thread_load.cuh"
#include "../type_utils.cuh"
#include "../vector_type.cuh"
#include "cta_exchange.cuh"

CUB_NS_PREFIX
namespace cub {


//-----------------------------------------------------------------------------
// Tags and constants
//-----------------------------------------------------------------------------

/**
 * Tuning policy for coalescing CTA tile-loading
 */
enum CtaLoadPolicy
{
	CTA_LOAD_DIRECT,			/// Reads consecutive thread-items directly from the input
	CTA_LOAD_TRANSPOSE,		/// Reads CTA-striped inputs as a coalescing optimization and then transposes them through shared memory into the desired blocks of thread-consecutive items
	CTA_LOAD_VECTORIZE,		/// Attempts to use CUDA's built-in vectorized items as a coalescing optimization
};


//-----------------------------------------------------------------------------
// Generic CtaLoad abstraction
//-----------------------------------------------------------------------------

/**
 * CTA collective abstraction for loading tiles of items
 */
template <
	typename 		InputIterator,						/// Input iterator type
	int 			CTA_THREADS,						/// The CTA size in threads
	int				ITEMS_PER_THREAD,					/// The number of items per thread
	CtaLoadPolicy	POLICY 		= CTA_LOAD_DIRECT,		/// (optional) CTA load tuning policy
	PtxLoadModifier 	MODIFIER 	= PTX_LOAD_NONE>			/// (optional) Cache modifier (e.g., PTX_LOAD_NONE/LOAD_WB/PTX_LOAD_CG/PTX_LOAD_CS/LOAD_WT/etc.)
class CtaLoad;



//-----------------------------------------------------------------------------
// CtaLoad abstraction specialized for CTA_LOAD_DIRECT policy
//-----------------------------------------------------------------------------

/**
 * CTA collective abstraction for loading tiles of items.
 *
 * Specialized to read consecutive thread-items directly from the input
 */
template <
	typename 		InputIterator,						/// Input iterator type
	int 			CTA_THREADS,						/// The CTA size in threads
	int				ITEMS_PER_THREAD,					/// The number of items per thread
	PtxLoadModifier 	MODIFIER>							/// (optional) Cache modifier (e.g., PTX_LOAD_NONE/LOAD_WB/PTX_LOAD_CG/PTX_LOAD_CS/LOAD_WT/etc.)
class CtaLoad<InputIterator, CTA_THREADS, ITEMS_PER_THREAD, CTA_LOAD_DIRECT, MODIFIER>
{
	//---------------------------------------------------------------------
	// Type definitions and constants
	//---------------------------------------------------------------------

private:

	// Data type of input iterator
	typedef typename std::iterator_traits<InputIterator>::value_type T;

public:

	/**
	 * Opaque shared memory storage layout
	 */
	typedef NullType SmemStorage;


	//---------------------------------------------------------------------
	// Interface
	//---------------------------------------------------------------------

public:

	/**
	 * Load tile.
	 *
	 * The aggregate set of items is assumed to be ordered across
	 * threads in "blocked" fashion, i.e., each thread owns an array
	 * of logically-consecutive items (and consecutive thread ranks own
	 * logically-consecutive arrays).
	 */
	template <typename SizeT>							/// (inferred) Integer counting type
	static __device__ __forceinline__ void Load(
		SmemStorage		&smem_storage,					/// (opaque) Shared memory storage
		T 				items[ITEMS_PER_THREAD],		/// (out) Data to load
		InputIterator 	itr,							/// (in) Input iterator for loading from
		const SizeT 	&cta_offset)					/// (in) Offset in itr at which to load the tile
	{
		// Read directly in thread-blocked order
		#pragma unroll
		for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
		{
			int item_offset = (threadIdx.x * ITEMS_PER_THREAD) + ITEM;
			items[ITEM] = ThreadLoad<MODIFIER>(itr + cta_offset + item_offset);
		}
	}


	/**
	 * Load tile, guarded by range
	 *
	 * The aggregate set of items is assumed to be ordered across
	 * threads in "blocked" fashion, i.e., each thread owns an array
	 * of logically-consecutive items (and consecutive thread ranks own
	 * logically-consecutive arrays).
	 */
	template <typename SizeT>							/// (inferred) Integer counting type
	static __device__ __forceinline__ void Load(
		SmemStorage		&smem_storage,					/// (opaque) Shared memory storage
		T 				items[ITEMS_PER_THREAD],		/// (out) Data to load
		InputIterator 	itr,							/// (in) Input iterator for loading from
		const SizeT 	&cta_offset,					/// (in) Offset in itr at which to load the tile
		const SizeT 	&guarded_elements)				/// (in) Number of valid items in the tile
	{
		// Read directly in thread-blocked order
		#pragma unroll
		for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
		{
			int item_offset = (threadIdx.x * ITEMS_PER_THREAD) + ITEM;
			if (item_offset < guarded_elements)
			{
				items[ITEM] = ThreadLoad<MODIFIER>(itr + cta_offset + item_offset);
			}
		}
	}
};


/**
 * CTA load direct interface
 */
template <
	int 			CTA_THREADS,					/// The CTA size in threads
	PtxLoadModifier 	MODIFIER,						/// Cache load modifier Cache modifier (e.g., PTX_LOAD_NONE/LOAD_WB/PTX_LOAD_CG/PTX_LOAD_CS/LOAD_WT/etc.)
	typename 		T,								/// (inferred) Value type
	int				ITEMS_PER_THREAD,				/// (inferred) The number of items per thread
	typename 		InputIterator,					/// (inferred) Input iterator type
	typename 		SizeT>							/// (inferred) Integer counting type
__device__ __forceinline__ void CtaLoadDirect(
	T 				(&items)[ITEMS_PER_THREAD],		/// (out) Data to load
	InputIterator 	itr,							/// (in) Input iterator for loading from
	const SizeT 	&cta_offset)					/// (in) Offset in itr at which to load the tile
{
	typedef CtaLoad<InputIterator, CTA_THREADS, ITEMS_PER_THREAD, CTA_LOAD_DIRECT, MODIFIER> CtaLoad;
	CtaLoad::Load(CtaLoad::SmemStorage(), items, itr, cta_offset);
}


/**
 * CTA load direct interface
 */
template <
	int 			CTA_THREADS,					/// The CTA size in threads
	typename 		T,								/// (inferred) Value type
	int				ITEMS_PER_THREAD,				/// (inferred) The number of items per thread
	typename 		InputIterator,					/// (inferred) Input iterator type
	typename 		SizeT>							/// (inferred) Integer counting type
__device__ __forceinline__ void CtaLoadDirect(
	T 				(&items)[ITEMS_PER_THREAD],		/// (out) Data to load
	InputIterator 	itr,							/// (in) Input iterator for loading from
	const SizeT 	&cta_offset)					/// (in) Offset in itr at which to load the tile
{
	CtaLoadDirect<CTA_THREADS, PTX_LOAD_NONE>(items, itr, cta_offset);
}

/**
 * CTA load direct interface, guarded by range
 */
template <
	int 			CTA_THREADS,					/// The CTA size in threads
	PtxLoadModifier 	MODIFIER,						/// Cache load modifier Cache modifier (e.g., PTX_LOAD_NONE/LOAD_WB/PTX_LOAD_CG/PTX_LOAD_CS/LOAD_WT/etc.)
	typename 		T,								/// (inferred) Value type
	int				ITEMS_PER_THREAD,				/// (inferred) The number of items per thread
	typename 		InputIterator,					/// (inferred) Input iterator type
	typename 		SizeT>							/// (inferred) Integer counting type
__device__ __forceinline__ void CtaLoadDirect(
	T 				(&items)[ITEMS_PER_THREAD],		/// (out) Data to load
	InputIterator 	itr,							/// (in) Input iterator for loading from
	const SizeT 	&cta_offset,					/// (in) Offset in itr at which to load the tile
	const SizeT 	&guarded_elements)				/// (in) Number of valid items in the tile
{
	typedef CtaLoad<InputIterator, CTA_THREADS, ITEMS_PER_THREAD, CTA_LOAD_DIRECT, MODIFIER> CtaLoad;
	CtaLoad::Load(CtaLoad::SmemStorage(), items, itr, cta_offset, guarded_elements);
}


/**
 * CTA load direct interface, guarded by range
 */
template <
	int 			CTA_THREADS,					/// The CTA size in threads
	typename 		T,								/// (inferred) Value type
	int				ITEMS_PER_THREAD,				/// (inferred) The number of items per thread
	typename 		InputIterator,					/// (inferred) Input iterator type
	typename 		SizeT>							/// (inferred) Integer counting type
__device__ __forceinline__ void CtaLoadDirect(
	T 				(&items)[ITEMS_PER_THREAD],		/// (out) Data to load
	InputIterator 	itr,							/// (in) Input iterator for loading from
	const SizeT 	&cta_offset,					/// (in) Offset in itr at which to load the tile
	const SizeT 	&guarded_elements)				/// (in) Number of valid items in the tile
{
	CtaLoadDirect<CTA_THREADS, PTX_LOAD_NONE>(items, itr, cta_offset, guarded_elements);
}



//-----------------------------------------------------------------------------
// CtaLoad abstraction specialized for CTA_LOAD_TRANSPOSE policy
//-----------------------------------------------------------------------------

/**
 * CTA collective abstraction for loading tiles of items.
 *
 * Specialized to transpose consecutive thread-items into CTA-striped thread-items as
 * a read-coalescing optimization
 */
template <
	typename 		InputIterator,						/// Input iterator type
	int 			CTA_THREADS,						/// The CTA size in threads
	int				ITEMS_PER_THREAD,					/// The number of items per thread
	PtxLoadModifier 	MODIFIER>							/// (optional) Cache modifier (e.g., PTX_LOAD_NONE/LOAD_WB/PTX_LOAD_CG/PTX_LOAD_CS/LOAD_WT/etc.)
class CtaLoad<InputIterator, CTA_THREADS, ITEMS_PER_THREAD, CTA_LOAD_TRANSPOSE, MODIFIER>
{
	//---------------------------------------------------------------------
	// Type definitions and constants
	//---------------------------------------------------------------------

private:

	// Data type of input iterator
	typedef typename std::iterator_traits<InputIterator>::value_type T;

	// CtaExchange utility type for keys
	typedef CtaExchange<
		T,
		CTA_THREADS,
		ITEMS_PER_THREAD> CtaExchange;


public:

	/**
	 * Opaque shared memory storage layout
	 */
	typedef typename CtaExchange::SmemStorage SmemStorage;


	//---------------------------------------------------------------------
	// Interface
	//---------------------------------------------------------------------

public:

	/**
	 * Load tile.
	 *
	 * The aggregate set of items is assumed to be ordered across
	 * threads in "blocked" fashion, i.e., each thread owns an array
	 * of logically-consecutive items (and consecutive thread ranks own
	 * logically-consecutive arrays).
	 */
	template <typename SizeT>							/// (inferred) Integer counting type
	static __device__ __forceinline__ void Load(
		SmemStorage		&smem_storage,					/// (opaque) Shared memory storage
		T 				items[ITEMS_PER_THREAD],		/// (out) Data to load
		InputIterator 	itr,							/// (in) Input iterator for loading from
		const SizeT 	&cta_offset)					/// (in) Offset in itr at which to load the tile
	{
		// Read in CTA-striped order
		#pragma unroll
		for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
		{
			int item_offset = (ITEM * CTA_THREADS) + threadIdx.x;
			items[ITEM] = ThreadLoad<MODIFIER>(itr + cta_offset + item_offset);
		}

		// Transpose to CTA-striped order
		CtaExchange::TransposeStripedBlocked(smem_storage, items);
	}

	/**
	 * Load tile, guarded by range
	 *
	 * The aggregate set of items is assumed to be ordered across
	 * threads in "blocked" fashion, i.e., each thread owns an array
	 * of logically-consecutive items (and consecutive thread ranks own
	 * logically-consecutive arrays).
	 */
	template <typename SizeT>							/// (inferred) Integer counting type
	static __device__ __forceinline__ void Load(
		SmemStorage		&smem_storage,					/// (opaque) Shared memory storage
		T 				items[ITEMS_PER_THREAD],		/// (out) Data to load
		InputIterator 	itr,							/// (in) Input iterator for loading from
		const SizeT 	&cta_offset,					/// (in) Offset in itr at which to load the tile
		const SizeT 	&guarded_elements)				/// (in) Number of valid items in the tile
	{
		// Read in CTA-striped order
		#pragma unroll
		for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
		{
			int item_offset = (ITEM * CTA_THREADS) + threadIdx.x;
			if (item_offset < guarded_elements)
			{
				items[ITEM] = ThreadLoad<MODIFIER>(itr + cta_offset + item_offset);
			}
		}

		// Transpose to CTA-striped order
		CtaExchange::TransposeStripedBlocked(smem_storage, items);
	}
};


//-----------------------------------------------------------------------------
// CtaLoad abstraction specialized for CTA_LOAD_VECTORIZE
//-----------------------------------------------------------------------------

/**
 * CTA collective abstraction for loading tiles of items.
 *
 * Specialized to minimize load instruction overhead using CUDA's built-in
 * vectorized items as a read-coalescing optimization.  This implementation
 * resorts to CTA_LOAD_DIRECT behavior if:
 * 		(a) The input iterator type is not a native pointer type
 * 		(b) The input pointer is not not quad-item aligned
 * 		(c) The input is guarded
 */
template <
	typename 		InputIterator,						/// Input iterator type
	int 			CTA_THREADS,						/// The CTA size in threads
	int				ITEMS_PER_THREAD,					/// The number of items per thread
	PtxLoadModifier 	MODIFIER>							/// (optional) Cache modifier (e.g., PTX_LOAD_NONE/LOAD_WB/PTX_LOAD_CG/PTX_LOAD_CS/LOAD_WT/etc.)
class CtaLoad<InputIterator, CTA_THREADS, ITEMS_PER_THREAD, CTA_LOAD_VECTORIZE, MODIFIER>
{
	//---------------------------------------------------------------------
	// Type definitions and constants
	//---------------------------------------------------------------------

private:

	enum
	{
		// Maximum CUDA vector size is 4 elements
		MAX_VEC_SIZE 		= CUB_MIN(4, ITEMS_PER_THREAD),

		// Vector size must be a power of two and an even divisor of the items per thread
		VEC_SIZE			= ((((MAX_VEC_SIZE - 1) & MAX_VEC_SIZE) == 0) && ((ITEMS_PER_THREAD % MAX_VEC_SIZE) == 0)) ?
								MAX_VEC_SIZE :
								1,

		VECTORS_PER_THREAD 	= ITEMS_PER_THREAD / VEC_SIZE,
	};

	// Value type
	typedef typename std::iterator_traits<InputIterator>::value_type T;

	// Vector type
	typedef typename VectorType<T, VEC_SIZE>::Type Vector;

	// CTA_LOAD_DIRECT specialization of CtaLoad for vector type
	typedef CtaLoad<Vector*, CTA_THREADS, VECTORS_PER_THREAD, CTA_LOAD_DIRECT, MODIFIER> CtaLoadVector;

	// CTA_LOAD_DIRECT specialization of CtaLoad for singleton type
	typedef CtaLoad<InputIterator, CTA_THREADS, ITEMS_PER_THREAD, CTA_LOAD_DIRECT, MODIFIER> CtaLoadSingly;

public:

	/**
	 * Opaque shared memory storage layout
	 */
	union SmemStorage
	{
		typename CtaLoadVector::SmemStorage vector_storage;
		typename CtaLoadSingly::SmemStorage singly_storage;
	};

	//---------------------------------------------------------------------
	// Interface
	//---------------------------------------------------------------------

public:

	/**
	 * Load tile.
	 *
	 * Specialized for native pointer types (attempts vectorization)
	 *
	 * The aggregate set of items is assumed to be ordered across
	 * threads in "blocked" fashion, i.e., each thread owns an array
	 * of logically-consecutive items (and consecutive thread ranks own
	 * logically-consecutive arrays).
	 */
	template <
		typename SizeT>									/// (inferred) Integer counting type
	static __device__ __forceinline__ void Load(
		SmemStorage		&smem_storage,					/// (opaque) Shared memory storage
		T 				items[ITEMS_PER_THREAD],		/// (out) Data to load
		T 				*ptr,							/// (in) Input iterator for loading from
		const SizeT 	&cta_offset)					/// (in) Offset in ptr at which to load the tile
	{
		// Vectorize if aligned
		if ((size_t(ptr) & (VEC_SIZE - 1)) == 0)
		{
			// Alias pointers (use "raw" array here which should get optimized away to prevent conservative PTXAS lmem spilling)
			T raw_items[ITEMS_PER_THREAD];
			Vector *item_vectors = reinterpret_cast<Vector *>(raw_items);
			Vector *ptr_vectors = reinterpret_cast<Vector *>(ptr + cta_offset);

			// Direct-load using vector types
			CtaLoadVector::Load(smem_storage.vector_storage, item_vectors, ptr_vectors, 0);

			// Copy
			#pragma unroll
			for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
			{
				items[ITEM] = raw_items[ITEM];
			}
		}
		else
		{
			// Unaligned: direct-load of individual items
			CtaLoadSingly::Load(smem_storage.singly_storage, items, ptr, cta_offset);
		}
	}


	/*
	 * Load tile.
	 *
	 * Specialized for opaque input iterators (skips vectorization)
	 *
	 * The aggregate set of items is assumed to be ordered across
	 * threads in "blocked" fashion, i.e., each thread owns an array
	 * of logically-consecutive items (and consecutive thread ranks own
	 * logically-consecutive arrays).
	 */
	template <
		typename _T,									/// (inferred) Value type
		typename _InputIterator,						/// (inferred) Output iterator type
		typename SizeT>									/// (inferred) Integer counting type
	static __device__ __forceinline__ void Load(
		SmemStorage		&smem_storage,					/// (opaque) Shared memory storage
		_T 				items[ITEMS_PER_THREAD],		/// (out) Data to load
		_InputIterator 	itr,							/// (in) Input iterator for loading from
		const SizeT 	&cta_offset)					/// (in) Offset in itr at which to load the tile
	{
		// Direct-load of individual items
		CtaLoadSingly::Load(smem_storage.singly_storage, items, itr, cta_offset);
	}


	/**
	 * Load tile, guarded by range
	 *
	 * The aggregate set of items is assumed to be ordered across
	 * threads in "blocked" fashion, i.e., each thread owns an array
	 * of logically-consecutive items (and consecutive thread ranks own
	 * logically-consecutive arrays).
	 */
	template <
		typename SizeT>									/// (inferred) Integer counting type
	static __device__ __forceinline__ void Load(
		SmemStorage		&smem_storage,					/// (opaque) Shared memory storage
		T 				items[ITEMS_PER_THREAD],		/// (out) Data to load
		InputIterator 	itr,							/// (in) Input iterator for loading from
		const SizeT 	&cta_offset,					/// (in) Offset in itr at which to load the tile
		const SizeT 	&guarded_elements)				/// (in) Number of valid items in the tile
	{
		// Direct-load of individual items
		CtaLoadSingly::Load(smem_storage.singly_storage, items, itr, cta_offset, guarded_elements);
	}
};







} // namespace cub
CUB_NS_POSTFIX
