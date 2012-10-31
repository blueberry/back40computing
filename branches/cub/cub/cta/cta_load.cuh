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
 * The cub::CtaLoad type provides global data movement operations for loading tiles of items across threads within a CUDA CTA.
 */

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

/// CUB namespace
namespace cub {


//-----------------------------------------------------------------------------
// Policy
//-----------------------------------------------------------------------------

/// Tuning policy for cub::CtaLoad
enum CtaLoadPolicy
{
    CTA_LOAD_DIRECT,        ///< Reads consecutive thread-items directly from the input
    CTA_LOAD_VECTORIZE,     ///< Attempts to use CUDA's built-in vectorized items as a coalescing optimization
    CTA_LOAD_TRANSPOSE,     ///< Reads CTA-striped inputs as a coalescing optimization and then transposes them through shared memory into the desired blocks of thread-consecutive items
};



//-----------------------------------------------------------------------------
// CtaLoadDirect() methods
//-----------------------------------------------------------------------------


/**
 * Load a tile of items across CTA threads directly using the specified cache modifier.
 *
 * \tparam MODIFIER             cub::PtxLoadModifier cache modifier.
 * \tparam T                    [inferred] The data type to load.
 * \tparam ITEMS_PER_THREAD     [inferred] The number of consecutive items contributed by each thread.
 * \tparam InputIterator        [inferred] The input iterator type (may be a simple pointer).
 * \tparam SizeT                [inferred] Integer type for offsets
 *
 * The aggregate set of items is assumed to be ordered across
 * threads in "blocked" fashion, i.e., each thread owns an array
 * of logically-consecutive items (and consecutive thread ranks own
 * logically-consecutive arrays).
 */
template <
    PtxLoadModifier MODIFIER,
    typename        T,
    int             ITEMS_PER_THREAD,
    typename        InputIterator,
    typename        SizeT>
__device__ __forceinline__ void CtaLoadDirect(
    T               (&items)[ITEMS_PER_THREAD],     ///< [out] Data to load
    InputIterator   itr,                            ///< [in] Input iterator for loading from
    const SizeT     &cta_offset)                    ///< [in] Offset in itr at which to load the tile
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
 * Load a tile of items across CTA threads directly.
 *
 * \tparam T                    [inferred] The data type to load.
 * \tparam ITEMS_PER_THREAD     [inferred] The number of consecutive items contributed by each thread.
 * \tparam InputIterator        [inferred] The input iterator type (may be a simple pointer).
 * \tparam SizeT                [inferred] Integer type for offsets
 *
 * The aggregate set of items is assumed to be ordered across
 * threads in "blocked" fashion, i.e., each thread owns an array
 * of logically-consecutive items (and consecutive thread ranks own
 * logically-consecutive arrays).
 */
template <
    typename        T,
    int             ITEMS_PER_THREAD,
    typename        InputIterator,
    typename        SizeT>
__device__ __forceinline__ void CtaLoadDirect(
    T               (&items)[ITEMS_PER_THREAD],     ///< [out] Data to load
    InputIterator   itr,                            ///< [in] Input iterator for loading from
    const SizeT     &cta_offset)                    ///< [in] Offset in itr at which to load the tile
{
    CtaLoadDirect<PTX_LOAD_NONE>(items, itr, cta_offset);
}


/**
 * Load striped tile directly using the specified cache modifier.
 *
 * \tparam CTA_THREADS          The CTA size in threads
 * \tparam MODIFIER             cub::PtxLoadModifier cache modifier.
 * \tparam T                    [inferred] The data type to load.
 * \tparam ITEMS_PER_THREAD     [inferred] The number of consecutive items contributed by each thread.
 * \tparam InputIterator        [inferred] The input iterator type (may be a simple pointer).
 * \tparam SizeT                [inferred] Integer type for offsets
 *
 * The aggregate set of items is assumed to be ordered across
 * threads in "striped" fashion, i.e., the \p ITEMS_PER_THREAD
 * items owned by each thread have logical stride CTA_THREADS between them.
 */
template <
    int             CTA_THREADS,
    PtxLoadModifier MODIFIER,
    typename        T,
    int             ITEMS_PER_THREAD,
    typename        InputIterator,
    typename        SizeT>
__device__ __forceinline__ void CtaLoadDirectStriped(
    T               (&items)[ITEMS_PER_THREAD],     ///< [out] Data to load
    InputIterator   itr,                            ///< [in] Input iterator for loading from
    const SizeT     &cta_offset)                    ///< [in] Offset in itr at which to load the tile
{
    // Read directly in CTA-striped order
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
    {
        int item_offset = (ITEM * CTA_THREADS) + threadIdx.x;
        items[ITEM] = ThreadLoad<MODIFIER>(itr + cta_offset + item_offset);
    }
}


/**
 * Load striped tile directly.
 *
 * \tparam CTA_THREADS          The CTA size in threads
 * \tparam MODIFIER             cub::PtxLoadModifier cache modifier.
 * \tparam T                    [inferred] The data type to load.
 * \tparam ITEMS_PER_THREAD     [inferred] The number of consecutive items contributed by each thread.
 * \tparam InputIterator        [inferred] The input iterator type (may be a simple pointer).
 * \tparam SizeT                [inferred] Integer type for offsets
 *
 * The aggregate set of items is assumed to be ordered across
 * threads in "striped" fashion, i.e., the \p ITEMS_PER_THREAD
 * items owned by each thread have logical stride CTA_THREADS between them.
 */
template <
    int             CTA_THREADS,
    typename        T,
    int             ITEMS_PER_THREAD,
    typename        InputIterator,
    typename        SizeT>
__device__ __forceinline__ void CtaLoadDirectStriped(
    T               (&items)[ITEMS_PER_THREAD],     ///< [out] Data to load
    InputIterator   itr,                            ///< [in] Input iterator for loading from
    const SizeT     &cta_offset)                    ///< [in] Offset in itr at which to load the tile
{
    CtaLoadDirectStriped<CTA_THREADS, PTX_LOAD_NONE>(items, itr, cta_offset);
}


/**
 * Load a tile of items across CTA threads directly using the specified cache modifier, guarded by range
 *
 * \tparam MODIFIER             cub::PtxLoadModifier cache modifier.
 * \tparam T                    [inferred] The data type to load.
 * \tparam ITEMS_PER_THREAD     [inferred] The number of consecutive items contributed by each thread.
 * \tparam InputIterator        [inferred] The input iterator type (may be a simple pointer).
 * \tparam SizeT                [inferred] Integer type for offsets
 *
 * The aggregate set of items is assumed to be ordered across
 * threads in "blocked" fashion, i.e., each thread owns an array
 * of logically-consecutive items (and consecutive thread ranks own
 * logically-consecutive arrays).
 */
template <
    PtxLoadModifier MODIFIER,
    typename        T,
    int             ITEMS_PER_THREAD,
    typename        InputIterator,
    typename        SizeT>
__device__ __forceinline__ void CtaLoadDirect(
    T               (&items)[ITEMS_PER_THREAD],     ///< [out] Data to load
    InputIterator   itr,                            ///< [in] Input iterator for loading from
    const SizeT     &cta_offset,                    ///< [in] Offset in itr at which to load the tile
    const SizeT     &guarded_elements)              ///< [in] Number of valid items in the tile
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


/**
 * Load a tile of items across CTA threads directly, guarded by range
 *
 * \tparam T                    [inferred] The data type to load.
 * \tparam ITEMS_PER_THREAD     [inferred] The number of consecutive items contributed by each thread.
 * \tparam InputIterator        [inferred] The input iterator type (may be a simple pointer).
 * \tparam SizeT                [inferred] Integer type for offsets
 *
 * The aggregate set of items is assumed to be ordered across
 * threads in "blocked" fashion, i.e., each thread owns an array
 * of logically-consecutive items (and consecutive thread ranks own
 * logically-consecutive arrays).
 */
template <
    typename        T,
    int             ITEMS_PER_THREAD,
    typename        InputIterator,
    typename        SizeT>
__device__ __forceinline__ void CtaLoadDirect(
    T               (&items)[ITEMS_PER_THREAD],     ///< [out] Data to load
    InputIterator   itr,                            ///< [in] Input iterator for loading from
    const SizeT     &cta_offset,                    ///< [in] Offset in itr at which to load the tile
    const SizeT     &guarded_elements)              ///< [in] Number of valid items in the tile
{
    CtaLoadDirect<PTX_LOAD_NONE>(items, itr, cta_offset, guarded_elements);
}


/**
 * Load striped directly tile using the specified cache modifier, guarded by range
 *
 * \tparam CTA_THREADS          The CTA size in threads
 * \tparam MODIFIER             cub::PtxLoadModifier cache modifier.
 * \tparam T                    [inferred] The data type to load.
 * \tparam ITEMS_PER_THREAD     [inferred] The number of consecutive items contributed by each thread.
 * \tparam InputIterator        [inferred] The input iterator type (may be a simple pointer).
 * \tparam SizeT                [inferred] Integer type for offsets
 *
 * The aggregate set of items is assumed to be ordered across
 * threads in "blocked" fashion, i.e., each thread owns an array
 * of logically-consecutive items (and consecutive thread ranks own
 * logically-consecutive arrays).
 */
template <
    int             CTA_THREADS,
    PtxLoadModifier MODIFIER,
    typename        T,
    int             ITEMS_PER_THREAD,
    typename        InputIterator,
    typename        SizeT>
__device__ __forceinline__ void CtaLoadDirectStriped(
    T               (&items)[ITEMS_PER_THREAD],     ///< [out] Data to load
    InputIterator   itr,                            ///< [in] Input iterator for loading from
    const SizeT     &cta_offset,                    ///< [in] Offset in itr at which to load the tile
    const SizeT     &guarded_elements)              ///< [in] Number of valid items in the tile
{
    // Read directly in CTA-striped order
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
    {
        int item_offset = (ITEM * CTA_THREADS) + threadIdx.x;
        if (item_offset < guarded_elements)
        {
            items[ITEM] = ThreadLoad<MODIFIER>(itr + cta_offset + item_offset);
        }
    }
}


/**
 * Load striped tile directly, guarded by range
 *
 * \tparam CTA_THREADS          The CTA size in threads
 * \tparam T                    [inferred] The data type to load.
 * \tparam ITEMS_PER_THREAD     [inferred] The number of consecutive items contributed by each thread.
 * \tparam InputIterator        [inferred] The input iterator type (may be a simple pointer).
 * \tparam SizeT                [inferred] Integer type for offsets
 *
 * The aggregate set of items is assumed to be ordered across
 * threads in "blocked" fashion, i.e., each thread owns an array
 * of logically-consecutive items (and consecutive thread ranks own
 * logically-consecutive arrays).
 */
template <
    int             CTA_THREADS,
    typename        T,
    int             ITEMS_PER_THREAD,
    typename        InputIterator,
    typename        SizeT>
__device__ __forceinline__ void CtaLoadDirectStriped(
    T               (&items)[ITEMS_PER_THREAD],     ///< [out] Data to load
    InputIterator   itr,                            ///< [in] Input iterator for loading from
    const SizeT     &cta_offset,                    ///< [in] Offset in itr at which to load the tile
    const SizeT     &guarded_elements)              ///< [in] Number of valid items in the tile
{
    CtaLoadDirectStriped<CTA_THREADS, PTX_LOAD_NONE>(items, itr, cta_offset, guarded_elements);
}




//-----------------------------------------------------------------------------
// CtaLoadVectorized() methods
//-----------------------------------------------------------------------------

/**
 * Load a tile of items across CTA threads directly using the specified cache modifier.
 *
 * \tparam MODIFIER             cub::PtxLoadModifier cache modifier.
 * \tparam T                    [inferred] The data type to load.
 * \tparam ITEMS_PER_THREAD     [inferred] The number of consecutive items contributed by each thread.
 * \tparam SizeT                [inferred] Integer type for offsets
 *
 * The aggregate set of items is assumed to be ordered across
 * threads in "blocked" fashion, i.e., each thread owns an array
 * of logically-consecutive items (and consecutive thread ranks own
 * logically-consecutive arrays).
 */
template <
    PtxLoadModifier MODIFIER,
    typename        T,
    int             ITEMS_PER_THREAD,
    typename        SizeT>
__device__ __forceinline__ void CtaLoadVectorized(
    T               (&items)[ITEMS_PER_THREAD],     ///< [out] Data to load
    T               *ptr,                           ///< [in] Input pointer for loading from
    const SizeT     &cta_offset)                    ///< [in] Offset in ptr at which to load the tile
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

    // Vectorize if aligned
    if ((size_t(ptr) & (VEC_SIZE - 1)) == 0)
    {
        // Alias pointers (use "raw" array here which should get optimized away to prevent conservative PTXAS lmem spilling)
        T raw_items[ITEMS_PER_THREAD];
        Vector *item_vectors = reinterpret_cast<Vector *>(raw_items);
        Vector *ptr_vectors = reinterpret_cast<Vector *>(ptr + cta_offset);

        // Direct-load using vector types
        CtaLoadDirect<MODIFIER>(item_vectors, ptr_vectors, 0);

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
        CtaLoadDirect<MODIFIER>(items, ptr, cta_offset);
    }
}


/**
 * Load a tile of items across CTA threads directly.
 *
 * \tparam T                    [inferred] The data type to load.
 * \tparam ITEMS_PER_THREAD     [inferred] The number of consecutive items contributed by each thread.
 * \tparam SizeT                [inferred] Integer type for offsets
 *
 * The aggregate set of items is assumed to be ordered across
 * threads in "blocked" fashion, i.e., each thread owns an array
 * of logically-consecutive items (and consecutive thread ranks own
 * logically-consecutive arrays).
 */
template <
    typename        T,
    int             ITEMS_PER_THREAD,
    typename        SizeT>
__device__ __forceinline__ void CtaLoadVectorized(
    T               (&items)[ITEMS_PER_THREAD],     ///< [out] Data to load
    T               *ptr,                           ///< [in] Input pointer for loading from
    const SizeT     &cta_offset)                    ///< [in] Offset in ptr at which to load the tile
{
    CtaLoadVectorized<PTX_LOAD_NONE>(items, ptr, cta_offset);
}



//-----------------------------------------------------------------------------
// Generic CtaLoad abstraction
//-----------------------------------------------------------------------------

/**
 * \brief The CtaLoad type provides global data movement operations for loading tiles of items across threads within a CUDA CTA.
 *
 * \tparam InputIterator        The input iterator type (may be a simple pointer).
 * \tparam CTA_THREADS          The CTA size in threads.
 * \tparam ITEMS_PER_THREAD     The number of consecutive items contributed by each thread.
 * \tparam POLICY               [optional] cub::CtaLoadPolicy tuning policy enumeration.  Default = cub::CTA_LOAD_DIRECT.
 * \tparam MODIFIER             [optional] cub::PtxLoadModifier Cache modifier.  Default = cub::PTX_LOAD_NONE.
 */
template <
    typename            InputIterator,
    int                 CTA_THREADS,
    int                 ITEMS_PER_THREAD,
    CtaLoadPolicy       POLICY = CTA_LOAD_DIRECT,
    PtxLoadModifier     MODIFIER = PTX_LOAD_NONE>
class CtaLoad
{
    //---------------------------------------------------------------------
    // Type definitions and constants
    //---------------------------------------------------------------------

private:

    // Data type of input iterator
    typedef typename std::iterator_traits<InputIterator>::value_type T;


    /// Load helper
    template <CtaLoadPolicy POLICY, int DUMMY = 0>
    struct LoadInternal;


    /**
     * CTA_LOAD_DIRECT load helper
     */
    template <int DUMMY>
    struct LoadInternal<CTA_LOAD_VECTORIZE, DUMMY>
    {
        /// Shared memory storage layout type
        typedef NullType SmemStorage;

        /// Load a tile of items across CTA threads
        template <typename SizeT>
        static __device__ __forceinline__ void Load(
            SmemStorage     &smem_storage,              ///< [in] Shared reference to opaque SmemStorage layout
            _T              items[ITEMS_PER_THREAD],    ///< [out] Data to load
            _InputIterator  itr,                        ///< [in] Input iterator for loading from
            const SizeT     &cta_offset)                ///< [in] Offset in itr at which to load the tile
        {
            CtaLoadDirect<MODIFIER>(items, ptr, cta_offset);
        }

        /// Load a tile of items across CTA threads, guarded by range
        template <typename SizeT>
        static __device__ __forceinline__ void Load(
            SmemStorage     &smem_storage,              ///< [in] Shared reference to opaque SmemStorage layout
            T               (&items)[ITEMS_PER_THREAD], ///< [out] Data to load
            InputIterator   itr,                        ///< [in] Input iterator for loading from
            const SizeT     &cta_offset,                ///< [in] Offset in itr at which to load the tile
            const SizeT     &guarded_elements)          ///< [in] Number of valid items in the tile
        {
            CtaLoadDirect<PTX_LOAD_NONE>(items, itr, cta_offset, guarded_elements);
        }
    };


    /**
     * CTA_LOAD_VECTORIZE load helper
     */
    template <int DUMMY>
    struct LoadInternal<CTA_LOAD_VECTORIZE, DUMMY>
    {
        /// Shared memory storage layout type
        typedef NullType SmemStorage;

        /// Load a tile of items across CTA threads, specialized for native pointer types (attempts vectorization)
        template <typename SizeT>
        static __device__ __forceinline__ void Load(
            SmemStorage     &smem_storage,              ///< [in] Shared reference to opaque SmemStorage layout
            T               items[ITEMS_PER_THREAD],    ///< [out] Data to load
            T               *ptr,                       ///< [in] Input iterator for loading from
            const SizeT     &cta_offset)                ///< [in] Offset in ptr at which to load the tile
        {
            CtaLoadVectorized<MODIFIER>(items, ptr, cta_offset);
        }

        /// Load a tile of items across CTA threads, specialized for opaque input iterators (skips vectorization)
        template <
            typename _T,
            typename _InputIterator,
            typename SizeT>
        static __device__ __forceinline__ void Load(
            SmemStorage     &smem_storage,              ///< [in] Shared reference to opaque SmemStorage layout
            _T              items[ITEMS_PER_THREAD],    ///< [out] Data to load
            _InputIterator  itr,                        ///< [in] Input iterator for loading from
            const SizeT     &cta_offset)                ///< [in] Offset in itr at which to load the tile
        {
            CtaLoadDirect<MODIFIER>(items, ptr, cta_offset);
        }

        /// Load a tile of items across CTA threads, guarded by range
        template <typename SizeT>
        static __device__ __forceinline__ void Load(
            SmemStorage     &smem_storage,              ///< [in] Shared reference to opaque SmemStorage layout
            T               (&items)[ITEMS_PER_THREAD], ///< [out] Data to load
            InputIterator   itr,                        ///< [in] Input iterator for loading from
            const SizeT     &cta_offset,                ///< [in] Offset in itr at which to load the tile
            const SizeT     &guarded_elements)          ///< [in] Number of valid items in the tile
        {
            CtaLoadDirect<PTX_LOAD_NONE>(items, itr, cta_offset, guarded_elements);
        }
    };


    /**
     * CTA_LOAD_VECTORIZE load helper
     */
    template <int DUMMY>
    struct LoadInternal<CTA_LOAD_VECTORIZE, DUMMY>
    {
        // CtaExchange utility type for keys
        typedef CtaExchange<T, CTA_THREADS, ITEMS_PER_THREAD> CtaExchange;

        /// Shared memory storage layout type
        typedef typename CtaExchange::SmemStorage SmemStorage;

        /// Load a tile of items across CTA threads
        template <typename SizeT>
        static __device__ __forceinline__ void Load(
            SmemStorage     &smem_storage,              ///< [in] Shared reference to opaque SmemStorage layout
            _T              items[ITEMS_PER_THREAD],    ///< [out] Data to load
            _InputIterator  itr,                        ///< [in] Input iterator for loading from
            const SizeT     &cta_offset)                ///< [in] Offset in itr at which to load the tile
        {
            CtaLoadDirectStriped<CTA_THREADS, MODIFIER>(items, ptr, cta_offset);

            // Transpose to CTA-striped order
            CtaExchange::TransposeStripedBlocked(smem_storage, items);
        }

        /// Load a tile of items across CTA threads, guarded by range
        template <typename SizeT>
        static __device__ __forceinline__ void Load(
            SmemStorage     &smem_storage,              ///< [in] Shared reference to opaque SmemStorage layout
            T               (&items)[ITEMS_PER_THREAD], ///< [out] Data to load
            InputIterator   itr,                        ///< [in] Input iterator for loading from
            const SizeT     &cta_offset,                ///< [in] Offset in itr at which to load the tile
            const SizeT     &guarded_elements)          ///< [in] Number of valid items in the tile
        {
            CtaLoadDirectStriped<CTA_THREADS, PTX_LOAD_NONE>(items, itr, cta_offset, guarded_elements);

            // Transpose to CTA-striped order
            CtaExchange::TransposeStripedBlocked(smem_storage, items);
        }

    };

    /// Shared memory storage layout type
    typedef typename LoadInternal<POLICY>::SmemStorage SmemStorage;

public:

    /// The operations exposed by CtaLoad require shared memory of this
    /// type.  This opaque storage can be allocated directly using the
    /// <tt>__shared__</tt> keyword.  Alternatively, it can be aliased to
    /// externally allocated shared memory or <tt>union</tt>'d with other types
    /// to facilitate shared memory reuse.
    typedef SmemStorage SmemStorage;


    //---------------------------------------------------------------------
    // Interface
    //---------------------------------------------------------------------

    /**
     * \brief Load a tile of items across CTA threads.
     *
     * \tparam SizeT                [inferred] Integer type for offsets
     *
     * The aggregate set of items is assumed to be ordered across
     * threads in "blocked" fashion, i.e., each thread owns an array
     * of logically-consecutive items (and consecutive thread ranks own
     * logically-consecutive arrays).
     */
    template <typename SizeT>
    static __device__ __forceinline__ void Load(
        SmemStorage     &smem_storage,              ///< [in] Shared reference to opaque SmemStorage layout
        T               items[ITEMS_PER_THREAD],    ///< [out] Data to load
        InputIterator   itr,                        ///< [in] Input iterator for loading from
        const SizeT     &cta_offset)                ///< [in] Offset in itr at which to load the tile
    {
        LoadInternal<POLICY>::template Load(smem_storage, items, itr, cta_offset);
    }

    /**
     * \brief Load a tile of items across CTA threads, guarded by range.
     *
     * \tparam SizeT                [inferred] Integer type for offsets
     *
     * The aggregate set of items is assumed to be ordered across
     * threads in "blocked" fashion, i.e., each thread owns an array
     * of logically-consecutive items (and consecutive thread ranks own
     * logically-consecutive arrays).
     */
    template <typename SizeT>
    static __device__ __forceinline__ void Load(
        SmemStorage     &smem_storage,              ///< [in] Shared reference to opaque SmemStorage layout
        T               (&items)[ITEMS_PER_THREAD], ///< [out] Data to load
        InputIterator   itr,                        ///< [in] Input iterator for loading from
        const SizeT     &cta_offset,                ///< [in] Offset in itr at which to load the tile
        const SizeT     &guarded_elements)          ///< [in] Number of valid items in the tile
    {
        LoadInternal<POLICY>::template Load(smem_storage, items, itr, cta_offset, guarded_elements);
    }
};


} // namespace cub
CUB_NS_POSTFIX
