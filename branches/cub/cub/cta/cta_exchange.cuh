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
 * The cub::CtaExchange type provides operations for reorganizing the partitioning of logical lists across CTA threads.
 */

/******************************************************************************
 * CTA abstractions for commonplace all-to-all exchanges between threads
 ******************************************************************************/

#pragma once

#include "../ns_wrapper.cuh"
#include "../device_props.cuh"
#include "../ptx_intrinsics.cuh"
#include "../type_utils.cuh"

CUB_NS_PREFIX

/// CUB namespace
namespace cub {

/**
 *  \addtogroup SimtCoop
 *  @{
 */

/**
 * \brief The CtaExchange type provides operations for reorganizing the partitioning of logical lists across CTA threads. ![](transpose_logo.png)
 *
 * <b>Overview</b>
 * \par
 * The operations exposed by CtaExchange allow CTAs to reorganize data items between
 * threads, converting between (or scattering to) the following partitioning arrangements:
 * -# <b><em>CTA-blocked</em> arrangement</b>.  The aggregate tile of items is partitioned
 *   evenly across threads in "blocked" fashion with thread<sub><em>i</em></sub>
 *   owning the <em>i</em><sup>th</sup> segment of consecutive elements.
 * -# <b><em>CTA-striped</em> arrangement</b>.  The aggregate tile of items is partitioned across
 *   threads in "striped" fashion, i.e., the \p ITEMS_PER_THREAD items owned by
 *   each thread have logical stride \p CTA_THREADS between them.
 *
 * \tparam T                    The data type to be exchanged.
 * \tparam CTA_THREADS          The CTA size in threads.
 * \tparam ITEMS_PER_THREAD     The number of items partitioned onto each thread.
 *
 * <b>Important Features and Considerations</b>
 * \par
 * - After any operation, a subsequent CTA barrier (<tt>__syncthreads()</tt>) is
 *   required if the supplied CtaScan::CtaExchange is to be reused/repurposed by the CTA.
 * - Zero bank conflicts for most types.
 *
 * <b>Algorithm</b>
 * \par
 * Regardless of the initial blocked/striped arrangement, CTA threads scatter
 * items into shared memory in <em>CTA-blocked</em>, taking care to include
 * one item of padding for every shared memory bank's worth of items.  After a
 * barrier, items are gathered in the desired blocked/striped arrangement.
 * <br>
 * <br>
 * \image html raking.png
 * <center><b>A CTA of 16 threads performing a conflict-free <em>CTA-blocked</em> gathering of 64 exchanged items.</b></center>
 * <br>
 *
 */
template <
    typename        T,
    int             CTA_THREADS,
    int             ITEMS_PER_THREAD>
class CtaExchange
{
    //---------------------------------------------------------------------
    // Type definitions and constants
    //---------------------------------------------------------------------

private:

    enum
    {
        TILE_ITEMS          = CTA_THREADS * ITEMS_PER_THREAD,

        LOG_SMEM_BANKS      = DeviceProps::LOG_SMEM_BANKS,
        SMEM_BANKS          = 1 << LOG_SMEM_BANKS,

        // Insert padding if the number of items per thread is a power of two
        PADDING             = ((ITEMS_PER_THREAD & (ITEMS_PER_THREAD - 1)) == 0),
        PADDING_ELEMENTS    = (PADDING) ? (TILE_ITEMS >> LOG_SMEM_BANKS) : 0,
    };

    /// Shared memory storage layout type
    struct SmemStorage
    {
        T exchange[TILE_ITEMS + PADDING_ELEMENTS];
    };

public:

    /// The operations exposed by CtaExchange require shared memory of this
    /// type.  This opaque storage can be allocated directly using the
    /// <tt>__shared__</tt> keyword.  Alternatively, it can be aliased to
    /// externally allocated shared memory or <tt>union</tt>'d with other types
    /// to facilitate shared memory reuse.
    typedef SmemStorage SmemStorage;


private:

    static __device__ __forceinline__ void ScatterBlocked(
        T items[ITEMS_PER_THREAD],
        T *buffer)
    {
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            int item_offset = (threadIdx.x * ITEMS_PER_THREAD) + ITEM;
            if (PADDING) item_offset = SHR_ADD(item_offset, LOG_SMEM_BANKS, item_offset);
            buffer[item_offset] = items[ITEM];
        }
    }

    static __device__ __forceinline__ void ScatterStriped(
        T items[ITEMS_PER_THREAD],
        T *buffer)
    {
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            int item_offset = (ITEM * CTA_THREADS) + threadIdx.x;
            if (PADDING) item_offset = SHR_ADD(item_offset, LOG_SMEM_BANKS, item_offset);
            buffer[item_offset] = items[ITEM];
        }
    }

    static __device__ __forceinline__ void GatherBlocked(
        T items[ITEMS_PER_THREAD],
        T *buffer)
    {
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            int item_offset = (threadIdx.x * ITEMS_PER_THREAD) + ITEM;
            if (PADDING) item_offset = SHR_ADD(item_offset, LOG_SMEM_BANKS, item_offset);
            items[ITEM] = buffer[item_offset];
        }
    }

    static __device__ __forceinline__ void GatherStriped(
        T items[ITEMS_PER_THREAD],
        T *buffer)
    {
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            int item_offset = (ITEM * CTA_THREADS) + threadIdx.x;
            if (PADDING) item_offset = SHR_ADD(item_offset, LOG_SMEM_BANKS, item_offset);
            items[ITEM] = buffer[item_offset];
        }
    }

    static __device__ __forceinline__ void ScatterRanked(
        T                 items[ITEMS_PER_THREAD],
        unsigned int     ranks[ITEMS_PER_THREAD],
        T                 *buffer)
    {
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            int item_offset = ranks[ITEM];
            if (PADDING) item_offset = SHR_ADD(item_offset, LOG_SMEM_BANKS, item_offset);

            buffer[item_offset] = items[ITEM];
        }
    }


    //---------------------------------------------------------------------
    // Interface
    //---------------------------------------------------------------------

public:

    /******************************************************************//**
     * \name Transpose exchanges
     *********************************************************************/
    //@{

    /**
     * \brief Transposes data items from <em>CTA-blocked</em> arrangement to <em>CTA-striped</em> arrangement.
     */
    static __device__ __forceinline__ void BlockedToStriped(
        SmemStorage     &smem_storage,              ///< [in] Shared reference to opaque SmemStorage layout
        T               items[ITEMS_PER_THREAD])    ///< [in-out] Items to exchange, converting between <em>CTA-blocked</em> and <em>CTA-striped</em> arrangements.
    {
        // Scatter items to shared memory
        ScatterBlocked(items, smem_storage.exchange);

        __syncthreads();

        // Gather items from shared memory
        GatherStriped(items, smem_storage.exchange);
    }


    /**
     * \brief Transposes data items from <em>CTA-striped</em> arrangement to <em>CTA-blocked</em> arrangement.
     */
    static __device__ __forceinline__ void StripedToBlocked(
        SmemStorage      &smem_storage,             ///< [in] Shared reference to opaque SmemStorage layout
        T                items[ITEMS_PER_THREAD])   ///< [in-out] Items to exchange, converting between <em>CTA-striped</em> and <em>CTA-blocked</em> arrangements.
    {
        // Scatter items to shared memory
        ScatterStriped(items, smem_storage.exchange);

        __syncthreads();

        // Gather items from shared memory
        GatherBlocked(items, smem_storage.exchange);
    }

    //@}
    /******************************************************************//**
     * \name Scatter exchanges
     *********************************************************************/
    //@{

    /**
     * \brief Exchanges data items annotated by rank into <em>CTA-blocked</em> arrangement.
     */
    static __device__ __forceinline__ void ScatterToBlocked(
        SmemStorage     &smem_storage,              ///< [in] Shared reference to opaque SmemStorage layout
        T               items[ITEMS_PER_THREAD],    ///< [in-out] Items to exchange
        unsigned int    ranks[ITEMS_PER_THREAD])    ///< [in] Corresponding scatter ranks
    {
        // Scatter items to shared memory
        Scatter(items, ranks, smem_storage.exchange);

        __syncthreads();

        // Gather items from shared memory
        GatherBlocked(items, smem_storage.exchange);
    }


    /**
     * \brief Exchanges data items annotated by rank into <em>CTA-striped</em> arrangement.
     */
    static __device__ __forceinline__ void ScatterToStriped(
        SmemStorage     &smem_storage,              ///< [in] Shared reference to opaque SmemStorage layout
        T               items[ITEMS_PER_THREAD],    ///< [in-out] Items to exchange
        unsigned int    ranks[ITEMS_PER_THREAD])    ///< [in] Corresponding scatter ranks
    {
        // Scatter items to shared memory
        Scatter(items, ranks, smem_storage.exchange);

        __syncthreads();

        // Gather items from shared memory
        GatherStriped(items, smem_storage.exchange);
    }

    //@}


};

/** @} */       // end of SimtCoop group

} // namespace cub
CUB_NS_POSTFIX
