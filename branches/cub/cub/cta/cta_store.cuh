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

/**
 * \file
 * The cub::CtaStore type provides store operations for writing global tiles of data from the CTA (in blocked arrangement across threads).
 */

#pragma once

#include <iterator>

#include "../ns_wrapper.cuh"
#include "../macro_utils.cuh"
#include "../thread/thread_store.cuh"
#include "../type_utils.cuh"
#include "../vector_type.cuh"
#include "cta_exchange.cuh"

CUB_NS_PREFIX

/// CUB namespace
namespace cub {

/**
 *  \addtogroup SimtUtils
 * @{
 */


/******************************************************************//**
 * \name CTA direct stores (blocked arrangement)
 *********************************************************************/
//@{

/**
 * \brief Store a tile of items across CTA threads directly using the specified cache modifier.
 *
 * \tparam MODIFIER             cub::PtxStoreModifier cache modifier.
 * \tparam T                    [inferred] The data type to store.
 * \tparam ITEMS_PER_THREAD     [inferred] The number of consecutive items partitioned onto each thread.
 * \tparam InputIterator        [inferred] The input iterator type (may be a simple pointer).
 * \tparam SizeT                [inferred] Integer type for offsets
 *
 * The aggregate tile of items is assumed to be partitioned evenly across
 * threads in <em>blocked</em> arrangement with thread<sub><em>i</em></sub> owning
 * the <em>i</em><sup>th</sup> segment of consecutive elements.
 */
template <
    PtxStoreModifier MODIFIER,
    typename        T,
    int             ITEMS_PER_THREAD,
    typename        InputIterator,
    typename        SizeT>
__device__ __forceinline__ void CtaStoreDirect(
    T               (&items)[ITEMS_PER_THREAD],     ///< [in] Data to store
    InputIterator   itr,                            ///< [in] Input iterator for storing from
    const SizeT     &cta_offset)                    ///< [in] Offset in \p itr at which to store the tile
{
    // Store directly in thread-blocked order
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
    {
        int item_offset = (threadIdx.x * ITEMS_PER_THREAD) + ITEM;
        ThreadStore<MODIFIER>(itr + cta_offset + item_offset, items[ITEM]);
    }
}


/**
 * \brief Store a tile of items across CTA threads directly.
 *
 * \tparam T                    [inferred] The data type to store.
 * \tparam ITEMS_PER_THREAD     [inferred] The number of consecutive items partitioned onto each thread.
 * \tparam InputIterator        [inferred] The input iterator type (may be a simple pointer).
 * \tparam SizeT                [inferred] Integer type for offsets
 *
 * The aggregate tile of items is assumed to be partitioned evenly across
 * threads in <em>blocked</em> arrangement with thread<sub><em>i</em></sub> owning
 * the <em>i</em><sup>th</sup> segment of consecutive elements.
 */
template <
    typename        T,
    int             ITEMS_PER_THREAD,
    typename        InputIterator,
    typename        SizeT>
__device__ __forceinline__ void CtaStoreDirect(
    T               (&items)[ITEMS_PER_THREAD],     ///< [in] Data to store
    InputIterator   itr,                            ///< [in] Input iterator for storing from
    const SizeT     &cta_offset)                    ///< [in] Offset in \p itr at which to store the tile
{
    CtaStoreDirect<PTX_STORE_NONE>(items, itr, cta_offset);
}


/**
 * \brief Store a tile of items across CTA threads directly using the specified cache modifier, guarded by range
 *
 * \tparam MODIFIER             cub::PtxStoreModifier cache modifier.
 * \tparam T                    [inferred] The data type to store.
 * \tparam ITEMS_PER_THREAD     [inferred] The number of consecutive items partitioned onto each thread.
 * \tparam InputIterator        [inferred] The input iterator type (may be a simple pointer).
 * \tparam SizeT                [inferred] Integer type for offsets
 *
 * The aggregate tile of items is assumed to be partitioned evenly across
 * threads in <em>blocked</em> arrangement with thread<sub><em>i</em></sub> owning
 * the <em>i</em><sup>th</sup> segment of consecutive elements.
 */
template <
    PtxStoreModifier MODIFIER,
    typename        T,
    int             ITEMS_PER_THREAD,
    typename        InputIterator,
    typename        SizeT>
__device__ __forceinline__ void CtaStoreDirect(
    T               (&items)[ITEMS_PER_THREAD],     ///< [in] Data to store
    InputIterator   itr,                            ///< [in] Input iterator for storing from
    const SizeT     &cta_offset,                    ///< [in] Offset in \p itr at which to store the tile
    const SizeT     &guarded_elements)              ///< [in] Number of valid items in the tile
{
    // Store directly in thread-blocked order
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


/**
 * \brief Store a tile of items across CTA threads directly, guarded by range
 *
 * \tparam T                    [inferred] The data type to store.
 * \tparam ITEMS_PER_THREAD     [inferred] The number of consecutive items partitioned onto each thread.
 * \tparam InputIterator        [inferred] The input iterator type (may be a simple pointer).
 * \tparam SizeT                [inferred] Integer type for offsets
 *
 * The aggregate tile of items is assumed to be partitioned evenly across
 * threads in <em>blocked</em> arrangement with thread<sub><em>i</em></sub> owning
 * the <em>i</em><sup>th</sup> segment of consecutive elements.
 */
template <
    typename        T,
    int             ITEMS_PER_THREAD,
    typename        InputIterator,
    typename        SizeT>
__device__ __forceinline__ void CtaStoreDirect(
    T               (&items)[ITEMS_PER_THREAD],     ///< [in] Data to store
    InputIterator   itr,                            ///< [in] Input iterator for storing from
    const SizeT     &cta_offset,                    ///< [in] Offset in \p itr at which to store the tile
    const SizeT     &guarded_elements)              ///< [in] Number of valid items in the tile
{
    CtaStoreDirect<PTX_STORE_NONE>(items, itr, cta_offset, guarded_elements);
}


//@}
/******************************************************************//**
 * \name CTA direct stores (striped arrangement)
 *********************************************************************/
//@{



/**
 * \brief Store striped tile directly using the specified cache modifier.
 *
 * \tparam CTA_THREADS          The CTA size in threads
 * \tparam MODIFIER             cub::PtxStoreModifier cache modifier.
 * \tparam T                    [inferred] The data type to store.
 * \tparam ITEMS_PER_THREAD     [inferred] The number of consecutive items partitioned onto each thread.
 * \tparam InputIterator        [inferred] The input iterator type (may be a simple pointer).
 * \tparam SizeT                [inferred] Integer type for offsets
 *
 * The aggregate tile of items is assumed to be partitioned across
 * threads in "striped" arrangement, i.e., the \p ITEMS_PER_THREAD
 * items owned by each thread have logical stride \p CTA_THREADS between them.
 */
template <
    int             CTA_THREADS,
    PtxStoreModifier MODIFIER,
    typename        T,
    int             ITEMS_PER_THREAD,
    typename        InputIterator,
    typename        SizeT>
__device__ __forceinline__ void CtaStoreDirectStriped(
    T               (&items)[ITEMS_PER_THREAD],     ///< [in] Data to store
    InputIterator   itr,                            ///< [in] Input iterator for storing from
    const SizeT     &cta_offset)                    ///< [in] Offset in \p itr at which to store the tile
{
    // Store directly in striped order
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
    {
        int item_offset = (ITEM * CTA_THREADS) + threadIdx.x;
        ThreadStore<MODIFIER>(itr + cta_offset + item_offset, items[ITEM]);
    }
}


/**
 * \brief Store striped tile directly.
 *
 * \tparam CTA_THREADS          The CTA size in threads
 * \tparam MODIFIER             cub::PtxStoreModifier cache modifier.
 * \tparam T                    [inferred] The data type to store.
 * \tparam ITEMS_PER_THREAD     [inferred] The number of consecutive items partitioned onto each thread.
 * \tparam InputIterator        [inferred] The input iterator type (may be a simple pointer).
 * \tparam SizeT                [inferred] Integer type for offsets
 *
 * The aggregate tile of items is assumed to be partitioned across
 * threads in <em>striped</em> arrangement, i.e., the \p ITEMS_PER_THREAD
 * items owned by each thread have logical stride \p CTA_THREADS between them.
 */
template <
    int             CTA_THREADS,
    typename        T,
    int             ITEMS_PER_THREAD,
    typename        InputIterator,
    typename        SizeT>
__device__ __forceinline__ void CtaStoreDirectStriped(
    T               (&items)[ITEMS_PER_THREAD],     ///< [in] Data to store
    InputIterator   itr,                            ///< [in] Input iterator for storing from
    const SizeT     &cta_offset)                    ///< [in] Offset in \p itr at which to store the tile
{
    CtaStoreDirectStriped<CTA_THREADS, PTX_STORE_NONE>(items, itr, cta_offset);
}


/**
 * Store striped directly tile using the specified cache modifier, guarded by range
 *
 * \tparam CTA_THREADS          The CTA size in threads
 * \tparam MODIFIER             cub::PtxStoreModifier cache modifier.
 * \tparam T                    [inferred] The data type to store.
 * \tparam ITEMS_PER_THREAD     [inferred] The number of consecutive items partitioned onto each thread.
 * \tparam InputIterator        [inferred] The input iterator type (may be a simple pointer).
 * \tparam SizeT                [inferred] Integer type for offsets
 *
 * The aggregate tile of items is assumed to be partitioned across
 * threads in <em>striped</em> arrangement, i.e., the \p ITEMS_PER_THREAD
 * items owned by each thread have logical stride \p CTA_THREADS between them.
 */
template <
    int             CTA_THREADS,
    PtxStoreModifier MODIFIER,
    typename        T,
    int             ITEMS_PER_THREAD,
    typename        InputIterator,
    typename        SizeT>
__device__ __forceinline__ void CtaStoreDirectStriped(
    T               (&items)[ITEMS_PER_THREAD],     ///< [in] Data to store
    InputIterator   itr,                            ///< [in] Input iterator for storing from
    const SizeT     &cta_offset,                    ///< [in] Offset in \p itr at which to store the tile
    const SizeT     &guarded_elements)              ///< [in] Number of valid items in the tile
{
    // Store directly in striped order
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


/**
 * \brief Store striped tile directly, guarded by range
 *
 * \tparam CTA_THREADS          The CTA size in threads
 * \tparam T                    [inferred] The data type to store.
 * \tparam ITEMS_PER_THREAD     [inferred] The number of consecutive items partitioned onto each thread.
 * \tparam InputIterator        [inferred] The input iterator type (may be a simple pointer).
 * \tparam SizeT                [inferred] Integer type for offsets
 *
 * The aggregate tile of items is assumed to be partitioned across
 * threads in <em>striped</em> arrangement, i.e., the \p ITEMS_PER_THREAD
 * items owned by each thread have logical stride \p CTA_THREADS between them.
 */
template <
    int             CTA_THREADS,
    typename        T,
    int             ITEMS_PER_THREAD,
    typename        InputIterator,
    typename        SizeT>
__device__ __forceinline__ void CtaStoreDirectStriped(
    T               (&items)[ITEMS_PER_THREAD],     ///< [in] Data to store
    InputIterator   itr,                            ///< [in] Input iterator for storing from
    const SizeT     &cta_offset,                    ///< [in] Offset in \p itr at which to store the tile
    const SizeT     &guarded_elements)              ///< [in] Number of valid items in the tile
{
    CtaStoreDirectStriped<CTA_THREADS, PTX_STORE_NONE>(items, itr, cta_offset, guarded_elements);
}


//@}
/******************************************************************//**
 * \name CTA vectorized stores (blocked arrangement)
 *********************************************************************/
//@{

/**
 * \brief Store a tile of items across CTA threads directly using the specified cache modifier.
 *
 * \tparam MODIFIER             cub::PtxStoreModifier cache modifier.
 * \tparam T                    [inferred] The data type to store.
 * \tparam ITEMS_PER_THREAD     [inferred] The number of consecutive items partitioned onto each thread.
 * \tparam SizeT                [inferred] Integer type for offsets
 *
 * The aggregate tile of items is assumed to be partitioned evenly across
 * threads in <em>blocked</em> arrangement with thread<sub><em>i</em></sub> owning
 * the <em>i</em><sup>th</sup> segment of consecutive elements.
 *
 * The following conditions will prevent vectorization and storing will fall back to cub::CTA_STORE_DIRECT:
 * - \p ITEMS_PER_THREAD is odd
 * - The \p InputIterator is not a simple pointer type
 * - The input offset (\p ptr + \p cta_offset) is not quad-aligned
 * - The data type \p T is not a built-in primitive or CUDA vector type (e.g., \p short, \p int2, \p double, \p float2, etc.)
 */
template <
    PtxStoreModifier MODIFIER,
    typename        T,
    int             ITEMS_PER_THREAD,
    typename        SizeT>
__device__ __forceinline__ void CtaStoreVectorized(
    T               (&items)[ITEMS_PER_THREAD],     ///< [in] Data to store
    T               *ptr,                           ///< [in] Input pointer for storing from
    const SizeT     &cta_offset)                    ///< [in] Offset in ptr at which to store the tile
{
    enum
    {
        // Maximum CUDA vector size is 4 elements
        MAX_VEC_SIZE = CUB_MIN(4, ITEMS_PER_THREAD),

        // Vector size must be a power of two and an even divisor of the items per thread
        VEC_SIZE = ((((MAX_VEC_SIZE - 1) & MAX_VEC_SIZE) == 0) && ((ITEMS_PER_THREAD % MAX_VEC_SIZE) == 0)) ?
            MAX_VEC_SIZE :
            1,

        VECTORS_PER_THREAD     = ITEMS_PER_THREAD / VEC_SIZE,
    };

    // Vector type
    typedef typename VectorType<T, VEC_SIZE>::Type Vector;

    // Alias global pointer
    Vector *ptr_vectors = reinterpret_cast<Vector *>(ptr + cta_offset);

    // Vectorize if aligned
    if ((size_t(ptr_vectors) & (VEC_SIZE - 1)) == 0)
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
        CtaStoreDirect<MODIFIER>(raw_vector, ptr_vectors, 0);
    }
    else
    {
        // Unaligned: direct-store of individual items
        CtaStoreDirect<MODIFIER>(items, ptr, cta_offset);
    }
}


/**
 * \brief Store a tile of items across CTA threads directly.
 *
 * \tparam T                    [inferred] The data type to store.
 * \tparam ITEMS_PER_THREAD     [inferred] The number of consecutive items partitioned onto each thread.
 * \tparam SizeT                [inferred] Integer type for offsets
 *
 * The aggregate tile of items is assumed to be partitioned evenly across
 * threads in <em>blocked</em> arrangement with thread<sub><em>i</em></sub> owning
 * the <em>i</em><sup>th</sup> segment of consecutive elements.
 *
 * The following conditions will prevent vectorization and storing will fall back to cub::CTA_STORE_DIRECT:
 * - \p ITEMS_PER_THREAD is odd
 * - The \p InputIterator is not a simple pointer type
 * - The input offset (\p ptr + \p cta_offset) is not quad-aligned
 * - The data type \p T is not a built-in primitive or CUDA vector type (e.g., \p short, \p int2, \p double, \p float2, etc.)
 */
template <
    typename        T,
    int             ITEMS_PER_THREAD,
    typename        SizeT>
__device__ __forceinline__ void CtaStoreVectorized(
    T               (&items)[ITEMS_PER_THREAD],     ///< [in] Data to store
    T               *ptr,                           ///< [in] Input pointer for storing from
    const SizeT     &cta_offset)                    ///< [in] Offset in ptr at which to store the tile
{
    CtaStoreVectorized<PTX_STORE_NONE>(items, ptr, cta_offset);
}

//@}


/** @} */       // end of SimtUtils group


//-----------------------------------------------------------------------------
// Generic CtaStore abstraction
//-----------------------------------------------------------------------------

/// Tuning policy for cub::CtaStore
enum CtaStorePolicy
{
    CTA_STORE_DIRECT,        ///< Stores consecutive thread-items directly from the input
    CTA_STORE_VECTORIZE,     ///< Attempts to use CUDA's built-in vectorized items as a coalescing optimization
    CTA_STORE_TRANSPOSE,     ///< Stores striped inputs as a coalescing optimization and then transposes them through shared memory into the desired blocks of thread-consecutive items
};



/**
 * \addtogroup SimtCoop
 * @{
 */


/**
 * \brief The CtaStore type provides store operations for writing global tiles of data from the CTA (in blocked arrangement across threads). ![](cta_store_logo.png)
 *
 * <b>Overview</b>
 * \par
 * CtaStore can be configured to use one of three alternative algorithms:
 *   -# <b>cub::CTA_STORE_DIRECT</b>.  Stores consecutive thread-items
 *      directly from the input.
 *   <br><br>
 *   -# <b>cub::CTA_STORE_VECTORIZE</b>.  Attempts to use CUDA's
 *      built-in vectorized items as a coalescing optimization.  For
 *      example, <tt>st.global.v4.s32</tt> will be generated when
 *      \p T = \p int and \p ITEMS_PER_THREAD > 4.
 *   <br><br>
 *   -# <b>cub::CTA_STORE_TRANSPOSE</b>.  Stores striped inputs as
 *      a coalescing optimization and then transposes them through
 *      shared memory into the desired blocks of thread-consecutive items
 *
 * \par
 * The data movement operations exposed by this type assume a blocked
 * arrangement of data amongst threads, i.e., an <em>n</em>-element list (or
 * <em>tile</em>) that is partitioned evenly among \p CTA_THREADS threads,
 * with thread<sub><em>i</em></sub> owning the <em>i</em><sup>th</sup> segment of
 * consecutive elements.
 *
 * \tparam InputIterator        The input iterator type (may be a simple pointer).
 * \tparam CTA_THREADS          The CTA size in threads.
 * \tparam ITEMS_PER_THREAD     The number of consecutive items partitioned onto each thread.
 * \tparam POLICY               <b>[optional]</b> cub::CtaStorePolicy tuning policy enumeration.  Default = cub::CTA_STORE_DIRECT.
 * \tparam MODIFIER             <b>[optional]</b> cub::PtxStoreModifier cache modifier.  Default = cub::PTX_STORE_NONE.
 *
 * <b>Important Features and Considerations</b>
 * \par
 * - After any operation, a subsequent CTA barrier (<tt>__syncthreads()</tt>) is
 *   required if the supplied CtaScan::SmemStorage is to be reused/repurposed by the CTA.
 * - The following conditions will prevent vectorization and storing will fall back to cub::CTA_STORE_DIRECT:
 *   - \p ITEMS_PER_THREAD is odd
 *   - The \p InputIterator is not a simple pointer type
 *   - The input offset (\p ptr + \p cta_offset) is not quad-aligned
 *   - The data type \p T is not a built-in primitive or CUDA vector type (e.g., \p short, \p int2, \p double, \p float2, etc.) *
 *
 * <b>Examples</b>
 * \par
 * - <b>Example 1:</b> Store four consecutive integers per thread:
 * \code
 * #include <cub.cuh>
 *
 * template <int CTA_THREADS>
 * __global__ void SomeKernel(int *d_in, ...)
 * {
 *      // Declare a parameterized CtaStore type for the kernel configuration
 *      typedef cub::CtaStore<int, CTA_THREADS, 4> CtaStore;
 *
 *      // Declare shared memory for CtaStore
 *      __shared__ typename CtaStore::SmemStorage smem_storage;
 *
 *      // A segment of four input items per thread
 *      int data[4];
 *
 *      // Store a tile of data
 *      CtaStore::Store(data, d_in, blockIdx.x * CTA_THREADS * 4);
 *
 *      ...
 * }
 * \endcode
 * - <b>Example 2:</b> Store four consecutive integers per thread using vectorized stores and global-only caching:
 * \code
 * #include <cub.cuh>
 *
 * template <int CTA_THREADS>
 * __global__ void SomeKernel(int *d_in, ...)
 * {
 *      const int ITEMS_PER_THREAD = 4;
 *
 *      // Declare a parameterized CtaStore type for the kernel configuration
 *      typedef cub::CtaStore<int, CTA_THREADS, 4, CTA_STORE_VECTORIZE, PTX_STORE_CG> CtaStore;
 *
 *      // Declare shared memory for CtaStore
 *      __shared__ typename CtaStore::SmemStorage smem_storage;
 *
 *      // A segment of four input items per thread
 *      int data[4];
 *
 *      // Store a tile of data using vector-store instructions if possible
 *      CtaStore::Store(data, d_in, blockIdx.x * CTA_THREADS * 4);
 *
 *      ...
 * }
 * \endcode
 * <br>
 */
template <
    typename            InputIterator,
    int                 CTA_THREADS,
    int                 ITEMS_PER_THREAD,
    CtaStorePolicy       POLICY = CTA_STORE_DIRECT,
    PtxStoreModifier     MODIFIER = PTX_STORE_NONE>
class CtaStore
{
    //---------------------------------------------------------------------
    // Type definitions and constants
    //---------------------------------------------------------------------

private:

    // Data type of input iterator
    typedef typename std::iterator_traits<InputIterator>::value_type T;


    /// Store helper
    template <CtaStorePolicy POLICY, int DUMMY = 0>
    struct StoreInternal;


    /**
     * CTA_STORE_DIRECT store helper
     */
    template <int DUMMY>
    struct StoreInternal<CTA_STORE_DIRECT, DUMMY>
    {
        /// Shared memory storage layout type
        typedef NullType SmemStorage;

        /// Store a tile of items across CTA threads
        template <typename SizeT>
        static __device__ __forceinline__ void Store(
            SmemStorage     &smem_storage,              ///< [in] Shared reference to opaque SmemStorage layout
            T               (&items)[ITEMS_PER_THREAD], ///< [in] Data to store
            InputIterator   itr,                        ///< [in] Input iterator for storing from
            const SizeT     &cta_offset)                ///< [in] Offset in \p itr at which to store the tile
        {
            CtaStoreDirect<MODIFIER>(items, itr, cta_offset);
        }

        /// Store a tile of items across CTA threads, guarded by range
        template <typename SizeT>
        static __device__ __forceinline__ void Store(
            SmemStorage     &smem_storage,              ///< [in] Shared reference to opaque SmemStorage layout
            T               (&items)[ITEMS_PER_THREAD], ///< [in] Data to store
            InputIterator   itr,                        ///< [in] Input iterator for storing from
            const SizeT     &cta_offset,                ///< [in] Offset in \p itr at which to store the tile
            const SizeT     &guarded_elements)          ///< [in] Number of valid items in the tile
        {
            CtaStoreDirect<PTX_STORE_NONE>(items, itr, cta_offset, guarded_elements);
        }
    };


    /**
     * CTA_STORE_VECTORIZE store helper
     */
    template <int DUMMY>
    struct StoreInternal<CTA_STORE_VECTORIZE, DUMMY>
    {
        /// Shared memory storage layout type
        typedef NullType SmemStorage;

        /// Store a tile of items across CTA threads, specialized for native pointer types (attempts vectorization)
        template <typename SizeT>
        static __device__ __forceinline__ void Store(
            SmemStorage     &smem_storage,              ///< [in] Shared reference to opaque SmemStorage layout
            T               (&items)[ITEMS_PER_THREAD], ///< [in] Data to store
            T               *ptr,                       ///< [in] Input iterator for storing from
            const SizeT     &cta_offset)                ///< [in] Offset in ptr at which to store the tile
        {
            CtaStoreVectorized<MODIFIER>(items, ptr, cta_offset);
        }

        /// Store a tile of items across CTA threads, specialized for opaque input iterators (skips vectorization)
        template <
            typename T,
            typename InputIterator,
            typename SizeT>
        static __device__ __forceinline__ void Store(
            SmemStorage     &smem_storage,              ///< [in] Shared reference to opaque SmemStorage layout
            T               (&items)[ITEMS_PER_THREAD], ///< [in] Data to store
            InputIterator   itr,                        ///< [in] Input iterator for storing from
            const SizeT     &cta_offset)                ///< [in] Offset in \p itr at which to store the tile
        {
            CtaStoreDirect<MODIFIER>(items, itr, cta_offset);
        }

        /// Store a tile of items across CTA threads, guarded by range
        template <typename SizeT>
        static __device__ __forceinline__ void Store(
            SmemStorage     &smem_storage,              ///< [in] Shared reference to opaque SmemStorage layout
            T               (&items)[ITEMS_PER_THREAD], ///< [in] Data to store
            InputIterator   itr,                        ///< [in] Input iterator for storing from
            const SizeT     &cta_offset,                ///< [in] Offset in \p itr at which to store the tile
            const SizeT     &guarded_elements)          ///< [in] Number of valid items in the tile
        {
            CtaStoreDirect<PTX_STORE_NONE>(items, itr, cta_offset, guarded_elements);
        }
    };


    /**
     * CTA_STORE_TRANSPOSE store helper
     */
    template <int DUMMY>
    struct StoreInternal<CTA_STORE_TRANSPOSE, DUMMY>
    {
        // CtaExchange utility type for keys
        typedef CtaExchange<T, CTA_THREADS, ITEMS_PER_THREAD> CtaExchange;

        /// Shared memory storage layout type
        typedef typename CtaExchange::SmemStorage SmemStorage;

        /// Store a tile of items across CTA threads
        template <typename SizeT>
        static __device__ __forceinline__ void Store(
            SmemStorage     &smem_storage,              ///< [in] Shared reference to opaque SmemStorage layout
            T               (&items)[ITEMS_PER_THREAD], ///< [in] Data to store
            InputIterator   itr,                        ///< [in] Input iterator for storing from
            const SizeT     &cta_offset)                ///< [in] Offset in \p itr at which to store the tile
        {
            // Transpose to striped order
            CtaExchange::BlockedToStriped(smem_storage, items);

            CtaStoreDirectStriped<CTA_THREADS, MODIFIER>(items, itr, cta_offset);
        }

        /// Store a tile of items across CTA threads, guarded by range
        template <typename SizeT>
        static __device__ __forceinline__ void Store(
            SmemStorage     &smem_storage,              ///< [in] Shared reference to opaque SmemStorage layout
            T               (&items)[ITEMS_PER_THREAD], ///< [in] Data to store
            InputIterator   itr,                        ///< [in] Input iterator for storing from
            const SizeT     &cta_offset,                ///< [in] Offset in \p itr at which to store the tile
            const SizeT     &guarded_elements)          ///< [in] Number of valid items in the tile
        {
            // Transpose to striped order
            CtaExchange::BlockedToStriped(smem_storage, items);

            CtaStoreDirectStriped<CTA_THREADS, PTX_STORE_NONE>(items, itr, cta_offset, guarded_elements);
        }

    };

    /// Shared memory storage layout type
    typedef typename StoreInternal<POLICY>::SmemStorage SmemLayout;

public:

    /// The operations exposed by CtaStore require shared memory of this
    /// type.  This opaque storage can be allocated directly using the
    /// <tt>__shared__</tt> keyword.  Alternatively, it can be aliased to
    /// externally allocated shared memory or <tt>union</tt>'d with other types
    /// to facilitate shared memory reuse.
    typedef SmemLayout SmemStorage;



    //---------------------------------------------------------------------
    // Interface
    //---------------------------------------------------------------------

    /**
     * \brief Store a tile of items across CTA threads.
     *
     * \tparam SizeT                [inferred] Integer type for offsets
     */
    template <typename SizeT>
    static __device__ __forceinline__ void Store(
        SmemStorage     &smem_storage,              ///< [in] Shared reference to opaque SmemStorage layout
        T               (&items)[ITEMS_PER_THREAD], ///< [in] Data to store
        InputIterator   itr,                        ///< [in] Input iterator for storing from
        const SizeT     &cta_offset)                ///< [in] Offset in \p itr at which to store the tile
    {
        StoreInternal<POLICY>::Store(smem_storage, items, itr, cta_offset);
    }

    /**
     * \brief Store a tile of items across CTA threads, guarded by range.
     *
     * \tparam SizeT                [inferred] Integer type for offsets
     */
    template <typename SizeT>
    static __device__ __forceinline__ void Store(
        SmemStorage     &smem_storage,              ///< [in] Shared reference to opaque SmemStorage layout
        T               (&items)[ITEMS_PER_THREAD], ///< [in] Data to store
        InputIterator   itr,                        ///< [in] Input iterator for storing from
        const SizeT     &cta_offset,                ///< [in] Offset in \p itr at which to store the tile
        const SizeT     &guarded_elements)          ///< [in] Number of valid items in the tile
    {
        StoreInternal<POLICY>::Store(smem_storage, items, itr, cta_offset, guarded_elements);
    }
};

/** @} */       // end of SimtCoop group

} // namespace cub
CUB_NS_POSTFIX
