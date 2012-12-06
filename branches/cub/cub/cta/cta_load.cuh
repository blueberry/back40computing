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
 * The cub::CtaLoad type provides operations for reading global tiles of data into the CTA (in blocked arrangement across threads).
 */

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

/**
 *  \addtogroup SimtUtils
 * @{
 */


/******************************************************************//**
 * \name CTA direct loads (blocked arrangement)
 *********************************************************************/
//@{

/**
 * \brief Load a tile of items across CTA threads directly using the specified cache modifier.
 *
 * \tparam MODIFIER             cub::PtxLoadModifier cache modifier.
 * \tparam T                    <b>[inferred]</b> The data type to load.
 * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
 * \tparam InputIterator        <b>[inferred]</b> The input iterator type (may be a simple pointer).
 * \tparam SizeT                <b>[inferred]</b> Integer type for offsets
 *
 * The aggregate tile of items is assumed to be partitioned evenly across
 * threads in "blocked" fashion with thread<sub><em>i</em></sub> owning
 * the <em>i</em><sup>th</sup> segment of consecutive elements.
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
    const SizeT     &cta_offset)                    ///< [in] Offset in \p itr at which to load the tile
{
    // Load directly in thread-blocked order
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
    {
        int item_offset = (threadIdx.x * ITEMS_PER_THREAD) + ITEM;
        items[ITEM] = ThreadLoad<MODIFIER>(itr + cta_offset + item_offset);
    }
}


/**
 * \brief Load a tile of items across CTA threads directly.
 *
 * \tparam T                    <b>[inferred]</b> The data type to load.
 * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
 * \tparam InputIterator        <b>[inferred]</b> The input iterator type (may be a simple pointer).
 * \tparam SizeT                <b>[inferred]</b> Integer type for offsets
 *
 * The aggregate tile of items is assumed to be partitioned evenly across
 * threads in "blocked" fashion with thread<sub><em>i</em></sub> owning
 * the <em>i</em><sup>th</sup> segment of consecutive elements.
 */
template <
    typename        T,
    int             ITEMS_PER_THREAD,
    typename        InputIterator,
    typename        SizeT>
__device__ __forceinline__ void CtaLoadDirect(
    T               (&items)[ITEMS_PER_THREAD],     ///< [out] Data to load
    InputIterator   itr,                            ///< [in] Input iterator for loading from
    const SizeT     &cta_offset)                    ///< [in] Offset in \p itr at which to load the tile
{
    CtaLoadDirect<PTX_LOAD_NONE>(items, itr, cta_offset);
}



/**
 * \brief Load a tile of items across CTA threads directly using the specified cache modifier, guarded by range
 *
 * \tparam MODIFIER             cub::PtxLoadModifier cache modifier.
 * \tparam T                    <b>[inferred]</b> The data type to load.
 * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
 * \tparam InputIterator        <b>[inferred]</b> The input iterator type (may be a simple pointer).
 * \tparam SizeT                <b>[inferred]</b> Integer type for offsets
 *
 * The aggregate tile of items is assumed to be partitioned evenly across
 * threads in "blocked" fashion with thread<sub><em>i</em></sub> owning
 * the <em>i</em><sup>th</sup> segment of consecutive elements.
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
    const SizeT     &cta_offset,                    ///< [in] Offset in \p itr at which to load the tile
    const SizeT     &guarded_items)              ///< [in] Number of valid items in the tile
{
    // Load directly in thread-blocked order
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
    {
        int item_offset = (threadIdx.x * ITEMS_PER_THREAD) + ITEM;
        if (item_offset < guarded_items)
        {
            items[ITEM] = ThreadLoad<MODIFIER>(itr + cta_offset + item_offset);
        }
    }
}


/**
 * \brief Load a tile of items across CTA threads directly, guarded by range
 *
 * \tparam T                    <b>[inferred]</b> The data type to load.
 * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
 * \tparam InputIterator        <b>[inferred]</b> The input iterator type (may be a simple pointer).
 * \tparam SizeT                <b>[inferred]</b> Integer type for offsets
 *
 * The aggregate tile of items is assumed to be partitioned evenly across
 * threads in "blocked" fashion with thread<sub><em>i</em></sub> owning
 * the <em>i</em><sup>th</sup> segment of consecutive elements.
 */
template <
    typename        T,
    int             ITEMS_PER_THREAD,
    typename        InputIterator,
    typename        SizeT>
__device__ __forceinline__ void CtaLoadDirect(
    T               (&items)[ITEMS_PER_THREAD],     ///< [out] Data to load
    InputIterator   itr,                            ///< [in] Input iterator for loading from
    const SizeT     &cta_offset,                    ///< [in] Offset in \p itr at which to load the tile
    const SizeT     &guarded_items)              ///< [in] Number of valid items in the tile
{
    CtaLoadDirect<PTX_LOAD_NONE>(items, itr, cta_offset, guarded_items);
}


/**
 * \brief Load a tile of items across CTA threads directly using the specified cache modifier, guarded by range, with assignment for out-of-bound elements
 *
 * \tparam MODIFIER             cub::PtxLoadModifier cache modifier.
 * \tparam T                    <b>[inferred]</b> The data type to load.
 * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
 * \tparam InputIterator        <b>[inferred]</b> The input iterator type (may be a simple pointer).
 * \tparam SizeT                <b>[inferred]</b> Integer type for offsets
 *
 * The aggregate tile of items is assumed to be partitioned evenly across
 * threads in "blocked" fashion with thread<sub><em>i</em></sub> owning
 * the <em>i</em><sup>th</sup> segment of consecutive elements.
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
    const SizeT     &cta_offset,                    ///< [in] Offset in \p itr at which to load the tile
    const SizeT     &guarded_items,              ///< [in] Number of valid items in the tile
    T               oob_default)                    ///< [in] Default value to assign out-of-bound items
{
    // Load directly in thread-blocked order
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
    {
        int item_offset = (threadIdx.x * ITEMS_PER_THREAD) + ITEM;
        items[ITEM] =  (item_offset < guarded_items) ?
            ThreadLoad<MODIFIER>(itr + cta_offset + item_offset) :
            oob_default;
    }
}


/**
 * \brief Load a tile of items across CTA threads directly, guarded by range, with assignment for out-of-bound elements
 *
 * \tparam T                    <b>[inferred]</b> The data type to load.
 * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
 * \tparam InputIterator        <b>[inferred]</b> The input iterator type (may be a simple pointer).
 * \tparam SizeT                <b>[inferred]</b> Integer type for offsets
 *
 * The aggregate tile of items is assumed to be partitioned evenly across
 * threads in "blocked" fashion with thread<sub><em>i</em></sub> owning
 * the <em>i</em><sup>th</sup> segment of consecutive elements.
 */
template <
    typename        T,
    int             ITEMS_PER_THREAD,
    typename        InputIterator,
    typename        SizeT>
__device__ __forceinline__ void CtaLoadDirect(
    T               (&items)[ITEMS_PER_THREAD],     ///< [out] Data to load
    InputIterator   itr,                            ///< [in] Input iterator for loading from
    const SizeT     &cta_offset,                    ///< [in] Offset in \p itr at which to load the tile
    const SizeT     &guarded_items,              ///< [in] Number of valid items in the tile
    T               oob_default)                    ///< [in] Default value to assign out-of-bound items
{
    CtaLoadDirect<PTX_LOAD_NONE>(items, itr, cta_offset, guarded_items, oob_default);
}


//@}
/******************************************************************//**
 * \name CTA direct loads (striped arrangement)
 *********************************************************************/
//@{


/**
 * \brief Load striped tile directly using the specified cache modifier.
 *
 * \tparam MODIFIER             cub::PtxLoadModifier cache modifier.
 * \tparam T                    <b>[inferred]</b> The data type to load.
 * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
 * \tparam InputIterator        <b>[inferred]</b> The input iterator type (may be a simple pointer).
 * \tparam SizeT                <b>[inferred]</b> Integer type for offsets
 *
 * The aggregate tile of items is assumed to be partitioned across
 * threads in "striped" fashion, i.e., the \p ITEMS_PER_THREAD
 * items owned by each thread have logical stride \p CTA_THREADS between them.
 */
template <
    PtxLoadModifier MODIFIER,
    typename        T,
    int             ITEMS_PER_THREAD,
    typename        InputIterator,
    typename        SizeT>
__device__ __forceinline__ void CtaLoadDirectStriped(
    T               (&items)[ITEMS_PER_THREAD],     ///< [out] Data to load
    InputIterator   itr,                            ///< [in] Input iterator for loading from
    const SizeT     &cta_offset)                    ///< [in] Offset in \p itr at which to load the tile
{
    // Load directly in striped order
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
    {
        int item_offset = (ITEM * blockDim.x) + threadIdx.x;
        items[ITEM] = ThreadLoad<MODIFIER>(itr + cta_offset + item_offset);
    }
}


/**
 * \brief Load striped tile directly.
 *
 * \tparam MODIFIER             cub::PtxLoadModifier cache modifier.
 * \tparam T                    <b>[inferred]</b> The data type to load.
 * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
 * \tparam InputIterator        <b>[inferred]</b> The input iterator type (may be a simple pointer).
 * \tparam SizeT                <b>[inferred]</b> Integer type for offsets
 *
 * The aggregate tile of items is assumed to be partitioned across
 * threads in "striped" fashion, i.e., the \p ITEMS_PER_THREAD
 * items owned by each thread have logical stride \p CTA_THREADS between them.
 */
template <
    typename        T,
    int             ITEMS_PER_THREAD,
    typename        InputIterator,
    typename        SizeT>
__device__ __forceinline__ void CtaLoadDirectStriped(
    T               (&items)[ITEMS_PER_THREAD],     ///< [out] Data to load
    InputIterator   itr,                            ///< [in] Input iterator for loading from
    const SizeT     &cta_offset)                    ///< [in] Offset in \p itr at which to load the tile
{
    CtaLoadDirectStriped<PTX_LOAD_NONE>(items, itr, cta_offset);
}

/**
 * \brief Load striped directly tile using the specified cache modifier, guarded by range
 *
 * \tparam CTA_THREADS          The CTA size in threads
 * \tparam MODIFIER             cub::PtxLoadModifier cache modifier.
 * \tparam T                    <b>[inferred]</b> The data type to load.
 * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
 * \tparam InputIterator        <b>[inferred]</b> The input iterator type (may be a simple pointer).
 * \tparam SizeT                <b>[inferred]</b> Integer type for offsets
 *
 * The aggregate tile of items is assumed to be partitioned across
 * threads in "striped" fashion, i.e., the \p ITEMS_PER_THREAD
 * items owned by each thread have logical stride \p CTA_THREADS between them.
 */
template <
    PtxLoadModifier MODIFIER,
    typename        T,
    int             ITEMS_PER_THREAD,
    typename        InputIterator,
    typename        SizeT>
__device__ __forceinline__ void CtaLoadDirectStriped(
    T               (&items)[ITEMS_PER_THREAD],     ///< [out] Data to load
    InputIterator   itr,                            ///< [in] Input iterator for loading from
    const SizeT     &cta_offset,                    ///< [in] Offset in \p itr at which to load the tile
    const SizeT     &guarded_items)              ///< [in] Number of valid items in the tile
{
    // Load directly in striped order
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
    {
        int item_offset = (ITEM * blockDim.x) + threadIdx.x;
        if (item_offset < guarded_items)
        {
            items[ITEM] = ThreadLoad<MODIFIER>(itr + cta_offset + item_offset);
        }
    }
}


/**
 * \brief Load striped tile directly, guarded by range
 *
 * \tparam T                    <b>[inferred]</b> The data type to load.
 * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
 * \tparam InputIterator        <b>[inferred]</b> The input iterator type (may be a simple pointer).
 * \tparam SizeT                <b>[inferred]</b> Integer type for offsets
 *
 * The aggregate tile of items is assumed to be partitioned across
 * threads in "striped" fashion, i.e., the \p ITEMS_PER_THREAD
 * items owned by each thread have logical stride \p CTA_THREADS between them.
 */
template <
    typename        T,
    int             ITEMS_PER_THREAD,
    typename        InputIterator,
    typename        SizeT>
__device__ __forceinline__ void CtaLoadDirectStriped(
    T               (&items)[ITEMS_PER_THREAD],     ///< [out] Data to load
    InputIterator   itr,                            ///< [in] Input iterator for loading from
    const SizeT     &cta_offset,                    ///< [in] Offset in \p itr at which to load the tile
    const SizeT     &guarded_items)              ///< [in] Number of valid items in the tile
{
    CtaLoadDirectStriped<PTX_LOAD_NONE>(items, itr, cta_offset, guarded_items);
}


/**
 * \brief Load striped directly tile using the specified cache modifier, guarded by range, with assignment for out-of-bound elements
 *
 * \tparam MODIFIER             cub::PtxLoadModifier cache modifier.
 * \tparam T                    <b>[inferred]</b> The data type to load.
 * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
 * \tparam InputIterator        <b>[inferred]</b> The input iterator type (may be a simple pointer).
 * \tparam SizeT                <b>[inferred]</b> Integer type for offsets
 *
 * The aggregate tile of items is assumed to be partitioned across
 * threads in "striped" fashion, i.e., the \p ITEMS_PER_THREAD
 * items owned by each thread have logical stride \p CTA_THREADS between them.
 */
template <
    PtxLoadModifier MODIFIER,
    typename        T,
    int             ITEMS_PER_THREAD,
    typename        InputIterator,
    typename        SizeT>
__device__ __forceinline__ void CtaLoadDirectStriped(
    T               (&items)[ITEMS_PER_THREAD],     ///< [out] Data to load
    InputIterator   itr,                            ///< [in] Input iterator for loading from
    const SizeT     &cta_offset,                    ///< [in] Offset in \p itr at which to load the tile
    const SizeT     &guarded_items,                 ///< [in] Number of valid items in the tile
    T               oob_default)                    ///< [in] Default value to assign out-of-bound items
{
    // Load directly in striped order
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
    {
        int item_offset = (ITEM * blockIdx.x) + threadIdx.x;
        items[ITEM] = (item_offset < guarded_items) ?
             ThreadLoad<MODIFIER>(itr + cta_offset + item_offset) :
             oob_default;
    }
}


/**
 * \brief Load striped tile directly, guarded by range, with assignment for out-of-bound elements
 *
 * \tparam T                    <b>[inferred]</b> The data type to load.
 * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
 * \tparam InputIterator        <b>[inferred]</b> The input iterator type (may be a simple pointer).
 * \tparam SizeT                <b>[inferred]</b> Integer type for offsets
 *
 * The aggregate tile of items is assumed to be partitioned across
 * threads in "striped" fashion, i.e., the \p ITEMS_PER_THREAD
 * items owned by each thread have logical stride \p CTA_THREADS between them.
 */
template <
    typename        T,
    int             ITEMS_PER_THREAD,
    typename        InputIterator,
    typename        SizeT>
__device__ __forceinline__ void CtaLoadDirectStriped(
    T               (&items)[ITEMS_PER_THREAD],     ///< [out] Data to load
    InputIterator   itr,                            ///< [in] Input iterator for loading from
    const SizeT     &cta_offset,                    ///< [in] Offset in \p itr at which to load the tile
    const SizeT     &guarded_items,                  ///< [in] Number of valid items in the tile
    T               oob_default)                    ///< [in] Default value to assign out-of-bound items
{
    CtaLoadDirectStriped<PTX_LOAD_NONE>(items, itr, cta_offset, guarded_items, oob_default);
}

//@}
/******************************************************************//**
 * \name CTA vectorized loads (blocked arrangement)
 *********************************************************************/
//@{

/**
 * \brief Load a tile of items across CTA threads directly using the specified cache modifier.
 *
 * \tparam MODIFIER             cub::PtxLoadModifier cache modifier.
 * \tparam T                    <b>[inferred]</b> The data type to load.
 * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
 * \tparam SizeT                <b>[inferred]</b> Integer type for offsets
 *
 * The aggregate tile of items is assumed to be partitioned evenly across
 * threads in "blocked" fashion with thread<sub><em>i</em></sub> owning
 * the <em>i</em><sup>th</sup> segment of consecutive elements.
 *
 * \par
 * The following conditions will prevent vectorization and loading will fall back to cub::CTA_LOAD_DIRECT:
 *   - \p ITEMS_PER_THREAD is odd
 *   - The \p InputIterator is not a simple pointer type
 *   - The input offset (\p ptr + \p cta_offset) is not quad-aligned
 *   - The data type \p T is not a built-in primitive or CUDA vector type (e.g., \p short, \p int2, \p double, \p float2, etc.)
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

    // Alias global pointer
    Vector *ptr_vectors = reinterpret_cast<Vector *>(ptr + cta_offset);

    // Vectorize if aligned
    if ((size_t(ptr_vectors) & (VEC_SIZE - 1)) == 0)
    {
        // Alias local data (use raw_items array here which should get optimized away to prevent conservative PTXAS lmem spilling)
        T raw_items[ITEMS_PER_THREAD];

        // Direct-load using vector types
        CtaLoadDirect<MODIFIER>(
            reinterpret_cast<Vector (&)[VECTORS_PER_THREAD]>(raw_items),
            ptr_vectors,
            0);

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
 * \brief Load a tile of items across CTA threads directly.
 *
 * \tparam T                    <b>[inferred]</b> The data type to load.
 * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
 * \tparam SizeT                <b>[inferred]</b> Integer type for offsets
 *
 * The aggregate tile of items is assumed to be partitioned evenly across
 * threads in "blocked" fashion with thread<sub><em>i</em></sub> owning
 * the <em>i</em><sup>th</sup> segment of consecutive elements.
 *
 * \par
 * The following conditions will prevent vectorization and loading will fall back to cub::CTA_LOAD_DIRECT:
 *   - \p ITEMS_PER_THREAD is odd
 *   - The \p InputIterator is not a simple pointer type
 *   - The input offset (\p ptr + \p cta_offset) is not quad-aligned
 *   - The data type \p T is not a built-in primitive or CUDA vector type (e.g., \p short, \p int2, \p double, \p float2, etc.)
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

//@}

/** @} */       // end of SimtUtils group



//-----------------------------------------------------------------------------
// Generic CtaLoad abstraction
//-----------------------------------------------------------------------------

/// Tuning policy for cub::CtaLoad
enum CtaLoadPolicy
{
    CTA_LOAD_DIRECT,        ///< Loads consecutive thread-items directly from the input
    CTA_LOAD_VECTORIZE,     ///< Attempts to use CUDA's built-in vectorized items as a coalescing optimization
    CTA_LOAD_TRANSPOSE,     ///< Loads striped inputs as a coalescing optimization and then transposes them through shared memory into the desired blocks of thread-consecutive items
};


/**
 * \addtogroup SimtCoop
 * @{
 */


/**
 * \brief The CtaLoad type provides operations for reading global tiles of data into the CTA (in blocked arrangement across threads). ![](cta_load_logo.png)
 *
 * <b>Overview</b>
 * \par
 * CtaLoad can be configured to use one of three alternative algorithms:
 *   -# <b>cub::CTA_LOAD_DIRECT</b>.  Loads consecutive thread-items
 *      directly from the input.
 *   <br><br>
 *   -# <b>cub::CTA_LOAD_VECTORIZE</b>.  Attempts to use CUDA's
 *      built-in vectorized items as a coalescing optimization.  For
 *      example, <tt>ld.global.v4.s32</tt> will be generated when
 *      \p T = \p int and \p ITEMS_PER_THREAD > 4.
 *   <br><br>
 *   -# <b>cub::CTA_LOAD_TRANSPOSE</b>.  Loads striped inputs as
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
 * \tparam POLICY               <b>[optional]</b> cub::CtaLoadPolicy tuning policy enumeration.  Default = cub::CTA_LOAD_DIRECT.
 * \tparam MODIFIER             <b>[optional]</b> cub::PtxLoadModifier cache modifier.  Default = cub::PTX_LOAD_NONE.
 *
 * <b>Important Features and Considerations</b>
 * \par
 * - After any operation, a subsequent CTA barrier (<tt>__syncthreads()</tt>) is
 *   required if the supplied CtaLoad::SmemStorage is to be reused/repurposed by the CTA.
 * - The following conditions will prevent vectorization and loading will fall back to cub::CTA_LOAD_DIRECT:
 *   - \p ITEMS_PER_THREAD is odd
 *   - The \p InputIterator is not a simple pointer type
 *   - The input offset (\p ptr + \p cta_offset) is not quad-aligned
 *   - The data type \p T is not a built-in primitive or CUDA vector type (e.g., \p short, \p int2, \p double, \p float2, etc.)
 *
 * <b>Examples</b>
 * \par
 * - <b>Example 1:</b> Have a 128-thread CTA load four consecutive integers per thread (blocked arrangement):
 *      \code
 *      #include <cub.cuh>
 *
 *      __global__ void SomeKernel(int *d_in, ...)
 *      {
 *          // Parameterize a CtaLoad type for use in the current problem context
 *          typedef cub::CtaLoad<int, CTA_THREADS, 4> CtaLoad;
 *
 *          // Declare shared memory for CtaLoad
 *          __shared__ typename CtaLoad::SmemStorage smem_storage;
 *
 *          // A segment of consecutive input items per thread
 *          int data[4];
 *
 *          // Load a tile of data at this block's offset
 *          CtaLoad::Load(data, d_in, blockIdx.x * 128 * 4);
 *
 *      \endcode
 *
 * \par
 * - <b>Example 2:</b> Have a CTA load consecutive integers per thread (blocked arrangement) using vectorized loads and global-only caching:
 *      \code
 *      #include <cub.cuh>
 *
 *      template <int CTA_THREADS, int ITEMS_PER_THREAD>
 *      __global__ void SomeKernel(int *d_in, ...)
 *      {
 *          // Parameterize a CtaLoad type for use in the current problem context
 *          typedef cub::CtaLoad<int, CTA_THREADS, ITEMS_PER_THREAD, CTA_LOAD_VECTORIZE, PTX_LOAD_CG> CtaLoad;
 *
 *          // Declare shared memory for CtaLoad
 *          __shared__ typename CtaLoad::SmemStorage smem_storage;
 *
 *          // A segment of consecutive input items per thread
 *          int data[ITEMS_PER_THREAD];
 *
 *          // Load a tile of data at this block's offset
 *          CtaLoad::Load(data, d_in, blockIdx.x * CTA_THREADS * ITEMS_PER_THREAD);
 *
 *      \endcode
 * <br>
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
    struct LoadInternal<CTA_LOAD_DIRECT, DUMMY>
    {
        /// Shared memory storage layout type
        typedef NullType SmemStorage;

        /// Load a tile of items across CTA threads
        template <typename SizeT>
        static __device__ __forceinline__ void Load(
            SmemStorage     &smem_storage,              ///< [in] Shared reference to opaque SmemStorage layout
            T               (&items)[ITEMS_PER_THREAD], ///< [out] Data to load
            InputIterator   itr,                        ///< [in] Input iterator for loading from
            const SizeT     &cta_offset)                ///< [in] Offset in \p itr at which to load the tile
        {
            CtaLoadDirect<MODIFIER>(items, itr, cta_offset);
        }

        /// Load a tile of items across CTA threads, guarded by range
        template <typename SizeT>
        static __device__ __forceinline__ void Load(
            SmemStorage     &smem_storage,              ///< [in] Shared reference to opaque SmemStorage layout
            T               (&items)[ITEMS_PER_THREAD], ///< [out] Data to load
            InputIterator   itr,                        ///< [in] Input iterator for loading from
            const SizeT     &cta_offset,                ///< [in] Offset in \p itr at which to load the tile
            const SizeT     &guarded_items)          ///< [in] Number of valid items in the tile
        {
            CtaLoadDirect<PTX_LOAD_NONE>(items, itr, cta_offset, guarded_items);
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
            T               (&items)[ITEMS_PER_THREAD], ///< [out] Data to load
            T               *ptr,                       ///< [in] Input iterator for loading from
            const SizeT     &cta_offset)                ///< [in] Offset in ptr at which to load the tile
        {
            CtaLoadVectorized<MODIFIER>(items, ptr, cta_offset);
        }

        /// Load a tile of items across CTA threads, specialized for opaque input iterators (skips vectorization)
        template <
            typename T,
            typename InputIterator,
            typename SizeT>
        static __device__ __forceinline__ void Load(
            SmemStorage     &smem_storage,              ///< [in] Shared reference to opaque SmemStorage layout
            T               (&items)[ITEMS_PER_THREAD], ///< [out] Data to load
            InputIterator   itr,                        ///< [in] Input iterator for loading from
            const SizeT     &cta_offset)                ///< [in] Offset in \p itr at which to load the tile
        {
            CtaLoadDirect<MODIFIER>(items, itr, cta_offset);
        }

        /// Load a tile of items across CTA threads, guarded by range
        template <typename SizeT>
        static __device__ __forceinline__ void Load(
            SmemStorage     &smem_storage,              ///< [in] Shared reference to opaque SmemStorage layout
            T               (&items)[ITEMS_PER_THREAD], ///< [out] Data to load
            InputIterator   itr,                        ///< [in] Input iterator for loading from
            const SizeT     &cta_offset,                ///< [in] Offset in \p itr at which to load the tile
            const SizeT     &guarded_items)          ///< [in] Number of valid items in the tile
        {
            CtaLoadDirect<PTX_LOAD_NONE>(items, itr, cta_offset, guarded_items);
        }
    };


    /**
     * CTA_LOAD_TRANSPOSE load helper
     */
    template <int DUMMY>
    struct LoadInternal<CTA_LOAD_TRANSPOSE, DUMMY>
    {
        // CtaExchange utility type for keys
        typedef CtaExchange<T, CTA_THREADS, ITEMS_PER_THREAD> CtaExchange;

        /// Shared memory storage layout type
        typedef typename CtaExchange::SmemStorage SmemStorage;

        /// Load a tile of items across CTA threads
        template <typename SizeT>
        static __device__ __forceinline__ void Load(
            SmemStorage     &smem_storage,              ///< [in] Shared reference to opaque SmemStorage layout
            T               (&items)[ITEMS_PER_THREAD], ///< [out] Data to store
            InputIterator   itr,                        ///< [in] Input iterator for loading from
            const SizeT     &cta_offset)                ///< [in] Offset in \p itr at which to load the tile
        {
            CtaLoadDirectStriped<MODIFIER>(items, itr, cta_offset);

            // Transpose to blocked order
            CtaExchange::StripedToBlocked(smem_storage, items);
        }

        /// Load a tile of items across CTA threads, guarded by range
        template <typename SizeT>
        static __device__ __forceinline__ void Load(
            SmemStorage     &smem_storage,              ///< [in] Shared reference to opaque SmemStorage layout
            T               (&items)[ITEMS_PER_THREAD], ///< [out] Data to load
            InputIterator   itr,                        ///< [in] Input iterator for loading from
            const SizeT     &cta_offset,                ///< [in] Offset in \p itr at which to load the tile
            const SizeT     &guarded_items)             ///< [in] Number of valid items in the tile
        {
            CtaLoadDirectStriped<MODIFIER>(items, itr, cta_offset, guarded_items);

            // Transpose to blocked order
            CtaExchange::StripedToBlocked(smem_storage, items);
        }

    };

    /// Shared memory storage layout type
    typedef typename LoadInternal<POLICY>::SmemStorage SmemLayout;

public:


    /// The operations exposed by CtaLoad require shared memory of this
    /// type.  This opaque storage can be allocated directly using the
    /// <tt>__shared__</tt> keyword.  Alternatively, it can be aliased to
    /// externally allocated shared memory or <tt>union</tt>'d with other types
    /// to facilitate shared memory reuse.
    typedef SmemLayout SmemStorage;


    //---------------------------------------------------------------------
    // Interface
    //---------------------------------------------------------------------

    /**
     * \brief Load a tile of items across CTA threads.
     *
     * \tparam SizeT                <b>[inferred]</b> Integer type for offsets
     */
    template <typename SizeT>
    static __device__ __forceinline__ void Load(
        SmemStorage     &smem_storage,              ///< [in] Shared reference to opaque SmemStorage layout
        T               (&items)[ITEMS_PER_THREAD], ///< [out] Data to load
        InputIterator   itr,                        ///< [in] Input iterator for loading from
        const SizeT     &cta_offset)                ///< [in] Offset in \p itr at which to load the tile
    {
        LoadInternal<POLICY>::Load(smem_storage, items, itr, cta_offset);
    }

    /**
     * \brief Load a tile of items across CTA threads, guarded by range.
     *
     * \tparam SizeT                <b>[inferred]</b> Integer type for offsets
     */
    template <typename SizeT>
    static __device__ __forceinline__ void Load(
        SmemStorage     &smem_storage,              ///< [in] Shared reference to opaque SmemStorage layout
        T               (&items)[ITEMS_PER_THREAD], ///< [out] Data to load
        InputIterator   itr,                        ///< [in] Input iterator for loading from
        const SizeT     &cta_offset,                ///< [in] Offset in \p itr at which to load the tile
        const SizeT     &guarded_items)             ///< [in] Number of valid items in the tile
    {
        LoadInternal<POLICY>::Load(smem_storage, items, itr, cta_offset, guarded_items);
    }
};

/** @} */       // end of SimtCoop group

} // namespace cub
CUB_NS_POSTFIX
