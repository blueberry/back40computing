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
 * CTA abstraction for storing tiles of items
 ******************************************************************************/

#pragma once

#include <iterator>

#include "../ns_wrapper.cuh"
#include "../macro_utils.cuh"
#include "../thread/thread_store.cuh"
#include "../type_utils.cuh"
#include "../vector_type.cuh"
#include "cta_exchange.cuh"

CUB_NS_PREFIX
namespace cub {


//-----------------------------------------------------------------------------
// Tags and constants
//-----------------------------------------------------------------------------

/**
 * Tuning policy for coalescing CTA tile-storing
 */
enum CtaStorePolicy
{
	CTA_STORE_DIRECT,			/// Writes consecutive thread-items directly to the output
	CTA_STORE_TRANSPOSE,		/// Transposes consecutive thread-items through shared memory and then writes them out in CTA-striped fashion as a write-coalescing optimization
	CTA_STORE_VECTORIZE,		/// Attempts to use CUDA's built-in vectorized items as a coalescing optimization
};


//-----------------------------------------------------------------------------
// Generic CtaStore abstraction
//-----------------------------------------------------------------------------

/**
 * CTA collective abstraction for storing tiles of items
 */
template <
	typename 		OutputIterator,						/// Output iterator type
	int 			CTA_THREADS,						/// The CTA size in threads
	int				ITEMS_PER_THREAD,					/// The number of items per thread
	CtaStorePolicy	POLICY 		= CTA_STORE_DIRECT,		/// (optional) CTA store tuning policy
	PtxStoreModifier 	MODIFIER 	= PTX_STORE_NONE>			/// (optional) Cache modifier (e.g., PTX_STORE_NONE/PTX_STORE_WB/PTX_STORE_CG/PTX_STORE_CS/PTX_STORE_WT/etc.)
class CtaStore;



//-----------------------------------------------------------------------------
// CtaStore abstraction specialized for CTA_STORE_DIRECT policy
//-----------------------------------------------------------------------------

/**
 * CTA collective abstraction for storing tiles of items.
 *
 * Specialized to write consecutive thread-items directly to the output
 */
template <
	typename 		OutputIterator,						/// Output iterator type
	int 			CTA_THREADS,						/// The CTA size in threads
	int				ITEMS_PER_THREAD,					/// The number of items per thread
	PtxStoreModifier 	MODIFIER>							/// (optional) Cache modifier (e.g., PTX_STORE_NONE/PTX_STORE_WB/PTX_STORE_CG/PTX_STORE_CS/PTX_STORE_WT/etc.)
class CtaStore<OutputIterator, CTA_THREADS, ITEMS_PER_THREAD, CTA_STORE_DIRECT, MODIFIER>
{
	//---------------------------------------------------------------------
	// Type definitions and constants
	//---------------------------------------------------------------------

private:

	// Data type of output iterator
	typedef typename std::iterator_traits<OutputIterator>::value_type T;

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
	 * Store tile.
	 *
	 * The aggregate set of items is assumed to be ordered across
	 * threads in "blocked" fashion, i.e., each thread owns an array
	 * of logically-consecutive items (and consecutive thread ranks own
	 * logically-consecutive arrays).
	 */
	template <typename SizeT>							/// (inferred) Integer counting type
	static __device__ __forceinline__ void Store(
		SmemStorage		&smem_storage,					/// (opaque) Shared memory storage
		T 				items[ITEMS_PER_THREAD],		/// (in) Data to store
		OutputIterator 	itr,							/// (in) Output iterator for storing to
		const SizeT 	&cta_offset)					/// (in) Offset in itr at which to store the tile
	{
		// Write out directly in thread-blocked order
		#pragma unroll
		for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
		{
			int item_offset = (threadIdx.x * ITEMS_PER_THREAD) + ITEM;
			ThreadStore<MODIFIER>(itr + cta_offset + item_offset, items[ITEM]);
		}
	}


	/**
	 * Store tile, guarded by range
	 *
	 * The aggregate set of items is assumed to be ordered across
	 * threads in "blocked" fashion, i.e., each thread owns an array
	 * of logically-consecutive items (and consecutive thread ranks own
	 * logically-consecutive arrays).
	 */
	template <typename SizeT>							/// (inferred) Integer counting type
	static __device__ __forceinline__ void Store(
		SmemStorage		&smem_storage,					/// (opaque) Shared memory storage
		T 				items[ITEMS_PER_THREAD],		/// (in) Data to store
		OutputIterator 	itr,							/// (in) Output iterator for storing to
		const SizeT 	&cta_offset,					/// (in) Offset in itr at which to store the tile
		const SizeT 	&guarded_elements)				/// (in) Number of valid items in the tile
	{
		// Write out directly in thread-blocked order
		#pragma unroll
		for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
		{
			int item_offset = (threadIdx.x * ITEMS_PER_THREAD) + ITEM;
			if (item_offset < guarded_elements)
			{
				ThreadStore<MODIFIER>(itr + cta_offset + item_offset, items[ITEM]);
			}
		}
	}
};


/**
 * CTA store direct interface
 */
template <
	int 			CTA_THREADS,					/// The CTA size in threads
	PtxStoreModifier 	MODIFIER,						/// Cache store modifier Cache modifier (e.g., PTX_STORE_NONE/PTX_STORE_WB/PTX_STORE_CG/PTX_STORE_CS/PTX_STORE_WT/etc.)
	typename 		T,								/// (inferred) Value type
	int				ITEMS_PER_THREAD,				/// (inferred) The number of items per thread
	typename 		InputIterator,					/// (inferred) Input iterator type
	typename 		SizeT>							/// (inferred) Integer counting type
__device__ __forceinline__ void CtaStoreDirect(
	T 				(&items)[ITEMS_PER_THREAD],		/// (in) Data to store
	InputIterator 	itr,							/// (in) Input iterator for storing from
	const SizeT 	&cta_offset)					/// (in) Offset in itr at which to store the tile
{
	typedef CtaStore<InputIterator, CTA_THREADS, ITEMS_PER_THREAD, CTA_STORE_DIRECT, MODIFIER> CtaStore;
	CtaStore::Store(CtaStore::SmemStorage(), items, itr, cta_offset);
}


/**
 * CTA store direct interface
 */
template <
	int 			CTA_THREADS,					/// The CTA size in threads
	typename 		T,								/// (inferred) Value type
	int				ITEMS_PER_THREAD,				/// (inferred) The number of items per thread
	typename 		InputIterator,					/// (inferred) Input iterator type
	typename 		SizeT>							/// (inferred) Integer counting type
__device__ __forceinline__ void CtaStoreDirect(
	T 				(&items)[ITEMS_PER_THREAD],		/// (in) Data to store
	InputIterator 	itr,							/// (in) Input iterator for storing from
	const SizeT 	&cta_offset)					/// (in) Offset in itr at which to store the tile
{
	CtaStoreDirect<CTA_THREADS, PTX_STORE_NONE>(items, itr, cta_offset);
}

/**
 * CTA store direct interface, guarded by range
 */
template <
	int 			CTA_THREADS,					/// The CTA size in threads
	PtxStoreModifier 	MODIFIER,						/// Cache store modifier Cache modifier (e.g., PTX_STORE_NONE/PTX_STORE_WB/PTX_STORE_CG/PTX_STORE_CS/PTX_STORE_WT/etc.)
	typename 		T,								/// (inferred) Value type
	int				ITEMS_PER_THREAD,				/// (inferred) The number of items per thread
	typename 		InputIterator,					/// (inferred) Input iterator type
	typename 		SizeT>							/// (inferred) Integer counting type
__device__ __forceinline__ void CtaStoreDirect(
	T 				(&items)[ITEMS_PER_THREAD],		/// (in) Data to store
	InputIterator 	itr,							/// (in) Input iterator for storing from
	const SizeT 	&cta_offset,					/// (in) Offset in itr at which to store the tile
	const SizeT 	&guarded_elements)				/// (in) Number of valid items in the tile
{
	typedef CtaStore<InputIterator, CTA_THREADS, ITEMS_PER_THREAD, CTA_STORE_DIRECT, MODIFIER> CtaStore;
	CtaStore::Store(CtaStore::SmemStorage(), items, itr, cta_offset, guarded_elements);
}


/**
 * CTA store direct interface, guarded by range
 */
template <
	int 			CTA_THREADS,					/// The CTA size in threads
	typename 		T,								/// (inferred) Value type
	int				ITEMS_PER_THREAD,				/// (inferred) The number of items per thread
	typename 		InputIterator,					/// (inferred) Input iterator type
	typename 		SizeT>							/// (inferred) Integer counting type
__device__ __forceinline__ void CtaStoreDirect(
	T 				(&items)[ITEMS_PER_THREAD],		/// (in) Data to store
	InputIterator 	itr,							/// (in) Input iterator for storing from
	const SizeT 	&cta_offset,					/// (in) Offset in itr at which to store the tile
	const SizeT 	&guarded_elements)				/// (in) Number of valid items in the tile
{
	CtaStoreDirect<CTA_THREADS, PTX_STORE_NONE>(items, itr, cta_offset, guarded_elements);
}

//-----------------------------------------------------------------------------
// CtaStore abstraction specialized for CTA_STORE_TRANSPOSE policy
//-----------------------------------------------------------------------------

/**
 * CTA collective abstraction for storing tiles of items.
 *
 * Specialized to transpose consecutive thread-items into CTA-striped thread-items as
 * a write-coalescing optimization
 */
template <
	typename 		OutputIterator,						/// Output iterator type
	int 			CTA_THREADS,						/// The CTA size in threads
	int				ITEMS_PER_THREAD,					/// The number of items per thread
	PtxStoreModifier 	MODIFIER>							/// (optional) Cache modifier (e.g., PTX_STORE_NONE/PTX_STORE_WB/PTX_STORE_CG/PTX_STORE_CS/PTX_STORE_WT/etc.)
class CtaStore<OutputIterator, CTA_THREADS, ITEMS_PER_THREAD, CTA_STORE_TRANSPOSE, MODIFIER>
{
	//---------------------------------------------------------------------
	// Type definitions and constants
	//---------------------------------------------------------------------

private:

	// Data type of output iterator
	typedef typename std::iterator_traits<OutputIterator>::value_type T;

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
	 * Store tile.
	 *
	 * The aggregate set of items is assumed to be ordered across
	 * threads in "blocked" fashion, i.e., each thread owns an array
	 * of logically-consecutive items (and consecutive thread ranks own
	 * logically-consecutive arrays).
	 */
	template <typename SizeT>							/// (inferred) Integer counting type
	static __device__ __forceinline__ void Store(
		SmemStorage		&smem_storage,					/// (opaque) Shared memory storage
		T 				items[ITEMS_PER_THREAD],		/// (in) Data to store
		OutputIterator 	itr,							/// (in) Output iterator for storing to
		const SizeT 	&cta_offset)					/// (in) Offset in itr at which to store the tile
	{
		// Transpose to CTA-striped order
		CtaExchange::TransposeBlockedStriped(smem_storage, items);

		// Write out in CTA-striped order
		#pragma unroll
		for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
		{
			int item_offset = (ITEM * CTA_THREADS) + threadIdx.x;
			ThreadStore<MODIFIER>(itr + cta_offset + item_offset, items[ITEM]);
		}
	}

	/**
	 * Store tile, guarded by range
	 *
	 * The aggregate set of items is assumed to be ordered across
	 * threads in "blocked" fashion, i.e., each thread owns an array
	 * of logically-consecutive items (and consecutive thread ranks own
	 * logically-consecutive arrays).
	 */
	template <typename SizeT>							/// (inferred) Integer counting type
	static __device__ __forceinline__ void Store(
		SmemStorage		&smem_storage,					/// (opaque) Shared memory storage
		T 				items[ITEMS_PER_THREAD],		/// (in) Data to store
		OutputIterator 	itr,							/// (in) Output iterator for storing to
		const SizeT 	&cta_offset,					/// (in) Offset in itr at which to store the tile
		const SizeT 	&guarded_elements)				/// (in) Number of valid items in the tile
	{
		// Transpose to CTA-striped order
		CtaExchange::TransposeBlockedStriped(smem_storage, items);

		// Write out in CTA-striped order
		#pragma unroll
		for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
		{
			int item_offset = (ITEM * CTA_THREADS) + threadIdx.x;
			if (item_offset < guarded_elements)
			{
				ThreadStore<MODIFIER>(itr + cta_offset + item_offset, items[ITEM]);
			}
		}
	}
};


//-----------------------------------------------------------------------------
// CtaStore abstraction specialized for CTA_STORE_VECTORIZE
//-----------------------------------------------------------------------------

/**
 * CTA collective abstraction for storing tiles of items.
 *
 * Specialized to minimize store instruction overhead using CUDA's built-in
 * vectorized items as a write-coalescing optimization.  This implementation
 * resorts to CTA_STORE_DIRECT behavior if:
 * 		(a) The output iterator type is not a native pointer type
 * 		(b) The output pointer is not not quad-item aligned
 * 		(c) The output is guarded
 */
template <
	typename 		OutputIterator,						/// Output iterator type
	int 			CTA_THREADS,						/// The CTA size in threads
	int				ITEMS_PER_THREAD,					/// The number of items per thread
	PtxStoreModifier 	MODIFIER>							/// (optional) Cache modifier (e.g., PTX_STORE_NONE/PTX_STORE_WB/PTX_STORE_CG/PTX_STORE_CS/PTX_STORE_WT/etc.)
class CtaStore<OutputIterator, CTA_THREADS, ITEMS_PER_THREAD, CTA_STORE_VECTORIZE, MODIFIER>
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
	typedef typename std::iterator_traits<OutputIterator>::value_type T;

	// Vector type
	typedef typename VectorType<T, VEC_SIZE>::Type Vector;

	// CTA_STORE_DIRECT specialization of CtaStore for vector type
	typedef CtaStore<Vector*, CTA_THREADS, VECTORS_PER_THREAD, CTA_STORE_DIRECT, MODIFIER> CtaStoreVector;

	// CTA_STORE_DIRECT specialization of CtaStore for singleton type
	typedef CtaStore<OutputIterator, CTA_THREADS, ITEMS_PER_THREAD, CTA_STORE_DIRECT, MODIFIER> CtaStoreSingly;

public:

	/**
	 * Opaque shared memory storage layout
	 */
	union SmemStorage
	{
		typename CtaStoreVector::SmemStorage vector_storage;
		typename CtaStoreSingly::SmemStorage singly_storage;
	};


	//---------------------------------------------------------------------
	// Interface
	//---------------------------------------------------------------------

public:

	/**
	 * Store tile.
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
	static __device__ __forceinline__ void Store(
		SmemStorage		&smem_storage,					/// (opaque) Shared memory storage
		T 				items[ITEMS_PER_THREAD],		/// (in) Data to store
		T 				*ptr,							/// (in) Output pointer for storing to
		const SizeT 	&cta_offset)					/// (in) Offset in ptr at which to store the tile
	{
		// Vectorize if aligned
		if ((size_t(ptr) & (VEC_SIZE - 1)) == 0)
		{
			// Alias pointers (use "raw" array here which should get optimized away to prevent conservative PTXAS lmem spilling)
			Vector raw_vector[VECTORS_PER_THREAD];
			T *raw_items = reinterpret_cast<T*>(raw_vector);
			Vector *ptr_vectors = reinterpret_cast<Vector*>(ptr + cta_offset);

			// Copy
			#pragma unroll
			for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
			{
				raw_items[ITEM] = items[ITEM];
			}

			// Direct-store using vector types
			CtaStoreVector::Store(smem_storage.vector_storage, raw_vector, ptr_vectors, 0);
		}
		else
		{
			// Unaligned: direct-store of individual items
			CtaStoreSingly::Store(smem_storage.singly_storage, items, ptr, cta_offset);
		}
	}


	/*
	 * Store tile.
	 *
	 * Specialized for opaque output iterators (skips vectorization)
	 *
	 * The aggregate set of items is assumed to be ordered across
	 * threads in "blocked" fashion, i.e., each thread owns an array
	 * of logically-consecutive items (and consecutive thread ranks own
	 * logically-consecutive arrays).
	 */
	template <
		typename _T,									/// (inferred) Value type
		typename _OutputIterator,						/// (inferred) Output iterator type
		typename SizeT>									/// (inferred) Integer counting type
	static __device__ __forceinline__ void Store(
		SmemStorage		&smem_storage,					/// (opaque) Shared memory storage
		_T 				items[ITEMS_PER_THREAD],		/// (in) Data to store
		_OutputIterator itr,							/// (in) Output iterator for storing to
		const SizeT 	&cta_offset)					/// (in) Offset in itr at which to store the tile
	{
		// Direct-store of individual items
		CtaStoreSingly::Store(smem_storage.singly_storage, items, itr, cta_offset);
	}


	/**
	 * Store tile, guarded by range
	 *
	 * The aggregate set of items is assumed to be ordered across
	 * threads in "blocked" fashion, i.e., each thread owns an array
	 * of logically-consecutive items (and consecutive thread ranks own
	 * logically-consecutive arrays).
	 */
	template <
		typename SizeT>									/// (inferred) Integer counting type
	static __device__ __forceinline__ void Store(
		SmemStorage		&smem_storage,					/// (opaque) Shared memory storage
		T 				items[ITEMS_PER_THREAD],		/// (in) Data to store
		OutputIterator 	itr,							/// (in) Output iterator for storing to
		const SizeT 	&cta_offset,					/// (in) Offset in itr at which to store the tile
		const SizeT 	&guarded_elements)				/// (in) Number of valid items in the tile
	{
		// Direct-store of individual items
		CtaStoreSingly::Store(smem_storage.singly_storage, items, itr, cta_offset, guarded_elements);
	}
};










} // namespace cub
CUB_NS_POSTFIX
