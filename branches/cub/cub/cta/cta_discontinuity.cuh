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
 * The cub::CtaDiscontinuity type provides operations for flagging discontinuities within a list of data items partitioned across CTA threads.
 */

#pragma once

#include "../device_props.cuh"
#include "../type_utils.cuh"
#include "../operators.cuh"
#include "../ns_wrapper.cuh"

CUB_NS_PREFIX

/// CUB namespace
namespace cub {

/**
 * \addtogroup SimtCoop
 * @{
 */


/**
 * \brief The CtaDiscontinuity type provides operations for flagging discontinuities within a list of data items partitioned across CTA threads. ![](discont_logo.png)
 *
 * <b>Overview</b>
 * \par
 * The operations exposed by CtaDiscontinuity allow CTAs to set "head flags" for data elements that
 * are different from their predecessor (as specified by a binary boolean operator).
 *
 * \tparam T                    The data type to be exchanged.
 * \tparam CTA_THREADS          The CTA size in threads.
 *
 * <b>Important Features and Considerations</b>
 * \par
 * - After any operation, a subsequent CTA barrier (<tt>__syncthreads()</tt>) is
 *   required if the supplied CtaDiscontinuity::SmemStorage is to be reused/repurposed by the CTA.
 * - Zero bank conflicts for most types.
 *
 */
template <
    typename    T,
	int 		CTA_THREADS>
class CtaDiscontinuity
{
private:

    //---------------------------------------------------------------------
    // Type definitions and constants
    //---------------------------------------------------------------------

    /// Shared memory storage layout type
    struct SmemLayout
    {
        T last_items[CTA_THREADS];      ///< Last element from each thread's input
    };

public:

    /// The operations exposed by CtaDiscontinuity require shared memory of this
    /// type.  This opaque storage can be allocated directly using the
    /// <tt>__shared__</tt> keyword.  Alternatively, it can be aliased to
    /// externally allocated shared memory or <tt>union</tt>'d with other types
    /// to facilitate shared memory reuse.
    typedef SmemLayout SmemStorage;


    /**
     * \brief Sets discontinuity flags for a tile of CTA items, for which the first item has no reference (and is always flagged).  The last tile item of the last thread is also returned to thread<sub>0</sub>.
     *
     * Assuming a <em>blocked</em> arrangement of elements across threads,
     * <tt>flags</tt><sub><em>i</em></sub> is set non-zero for item
     * <tt>input</tt><sub><em>i</em></sub> when <tt>scan_op(</tt><em>previous-item</em>, <tt>input<sub><em>i</em></sub>)</tt>
     * is \p true (where <em>previous-item</em> is either <tt>input<sub><em>i-1</em></sub></tt>,
     * or <tt>input<sub><em>ITEMS_PER_THREAD</em>-1</sub></tt> in the previous thread).  Furthermore,
     * <tt>flags</tt><sub><em>i</em></sub> is always non-zero for <tt>input<sub>0</sub></tt>
     * in <em>thread</em><sub>0</sub>.
     *
     * The \p last_tile_item is undefined in threads other than <em>thread</em><sub>0</sub>.
     *
     * \smemreuse
     *
     * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
     * \tparam FlagT                 <b>[inferred]</b> The flag type (must be an integer type)
     * \tparam FlagOp               <b>[inferred]</b> Binary boolean functor type, having input parameters <tt>(const T &a, const T &b)</tt> and returning \p true if a discontinuity exists between \p a and \p b, otherwise \p false.
     */
    template <
        int             ITEMS_PER_THREAD,
        typename        FlagT,
        typename        FlagOp>
    static __device__ __forceinline__ void Flag(
        SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
        T               (&input)[ITEMS_PER_THREAD],     ///< [in] Input items
        FlagOp          flag_op,                        ///< [in] Binary boolean flag predicate
        FlagT           (&flags)[ITEMS_PER_THREAD],     ///< [out] Discontinuity flags
        T               &last_tile_item)                ///< [out] <b>[<em>thread</em><sub>0</sub> only]</b> The last tile item (<tt>input<sub><em>ITEMS_PER_THREAD</em>-1</sub></tt> from thread<sub><tt><em>CTA_THREADS</em></tt>-1</sub>)
    {
        // Share last item
        smem_storage.last_items[threadIdx.x] = input[ITEMS_PER_THREAD - 1];

        __syncthreads();

        // Set flag for first item
        if (threadIdx.x == 0)
        {
            flags[0] = 1;
            last_tile_item = smem_storage.last_items[CTA_THREADS - 1];
        }
        else
        {
            flags[0] = flag_op(smem_storage.last_items[threadIdx.x - 1], input[0]);
        }


        // Set flags for remaining items
        #pragma unroll
        for (int ITEM = 1; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            flags[ITEM] = flag_op(input[ITEM - 1], input[ITEM]);
        }
    }


    /**
     * \brief Sets discontinuity flags for a tile of CTA items, for which the first item has no reference (and is always flagged).
     *
     * Assuming a <em>blocked</em> arrangement of elements across threads,
     * <tt>flags</tt><sub><em>i</em></sub> is set non-zero for item
     * <tt>input</tt><sub><em>i</em></sub> when <tt>scan_op(</tt><em>previous-item</em>, <tt>input<sub><em>i</em></sub>)</tt>
     * is \p true (where <em>previous-item</em> is either <tt>input<sub><em>i-1</em></sub></tt>,
     * or <tt>input<sub><em>ITEMS_PER_THREAD</em>-1</sub></tt> in the previous thread).  Furthermore,
     * <tt>flags</tt><sub><em>i</em></sub> is always non-zero for <tt>input<sub>0</sub></tt>
     * in <em>thread</em><sub>0</sub>.
     *
     * The \p last_tile_item is undefined in threads other than <em>thread</em><sub>0</sub>.
     *
     * \smemreuse
     *
     * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
     * \tparam FlagT                 <b>[inferred]</b> The flag type (must be an integer type)
     * \tparam FlagOp               <b>[inferred]</b> Binary boolean functor type, having input parameters <tt>(const T &a, const T &b)</tt> and returning \p true if a discontinuity exists between \p a and \p b, otherwise \p false.
     */
    template <
        int             ITEMS_PER_THREAD,
        typename        FlagT,
        typename        FlagOp>
    static __device__ __forceinline__ void Flag(
        SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
        T               (&input)[ITEMS_PER_THREAD],     ///< [in] Input items
        FlagOp          flag_op,                        ///< [in] Binary boolean flag predicate
        FlagT           (&flags)[ITEMS_PER_THREAD])     ///< [out] Discontinuity flags
    {
        T last_tile_item;   // discard
        Flag(smem_storage, input, flag_op, flags, last_tile_item);
    }



    /**
     * \brief Sets discontinuity flags for a tile of CTA items.  The last tile item of the last thread is also returned to thread<sub>0</sub>.
     *
     * Assuming a <em>blocked</em> arrangement of elements across threads,
     * <tt>flags</tt><sub><em>i</em></sub> is set non-zero for item
     * <tt>input</tt><sub><em>i</em></sub> when <tt>scan_op(</tt><em>previous-item</em>, <tt>input<sub><em>i</em></sub>)</tt>
     * is \p true (where <em>previous-item</em> is either <tt>input<sub><em>i-1</em></sub></tt>,
     * or <tt>input<sub><em>ITEMS_PER_THREAD</em>-1</sub></tt> in the previous thread).  For
     * <em>thread</em><sub>0</sub>, item <tt>input<sub>0</sub></tt> is compared
     * against /p tile_prefix.
     *
     * The \p tile_prefix and \p last_tile_item are undefined in threads other than <em>thread</em><sub>0</sub>.
     *
     * \smemreuse
     *
     * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
     * \tparam FlagT                 <b>[inferred]</b> The flag type (must be an integer type)
     * \tparam FlagOp               <b>[inferred]</b> Binary boolean functor type, having input parameters <tt>(const T &a, const T &b)</tt> and returning \p true if a discontinuity exists between \p a and \p b, otherwise \p false.
     */
    template <
        int             ITEMS_PER_THREAD,
        typename        FlagT,
        typename        FlagOp>
    static __device__ __forceinline__ void Flag(
        SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
        T               (&input)[ITEMS_PER_THREAD],     ///< [in] Input items
        T               tile_prefix,                    ///< [in] <b>[<em>thread</em><sub>0</sub> only]</b> Item with which to compare the first tile item (<tt>input<sub>0</sub></tt>from <em>thread</em><sub>0</sub>).
        FlagOp          flag_op,                        ///< [in] Binary boolean flag predicate
        FlagT           (&flags)[ITEMS_PER_THREAD],     ///< [out] Discontinuity flags
        T               &last_tile_item)                ///< [out] <b>[<em>thread</em><sub>0</sub> only]</b> The last tile item (<tt>input<sub><em>ITEMS_PER_THREAD</em>-1</sub></tt> from <em>thread</em><sub><tt><em>CTA_THREADS</em></tt>-1</sub>)
    {
        // Share last item
        smem_storage.last_items[threadIdx.x] = input[ITEMS_PER_THREAD - 1];

        __syncthreads();

        // Set flag for first item
        int prefix;
        if (threadIdx.x == 0)
        {
            prefix = tile_prefix;
            last_tile_item = smem_storage.last_items[CTA_THREADS - 1];
        }
        else
        {
            prefix = smem_storage.last_items[threadIdx.x - 1];
        }
        flags[0] = flag_op(prefix, input[0]);


        // Set flags for remaining items
        #pragma unroll
        for (int ITEM = 1; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            flags[ITEM] = flag_op(input[ITEM - 1], input[ITEM]);
        }
    }


    /**
     * \brief Sets discontinuity flags for a tile of CTA items.
     *
     * Assuming a <em>blocked</em> arrangement of elements across threads,
     * <tt>flags</tt><sub><em>i</em></sub> is set non-zero for item
     * <tt>input</tt><sub><em>i</em></sub> when <tt>scan_op(</tt><em>previous-item</em>, <tt>input<sub><em>i</em></sub>)</tt>
     * is \p true (where <em>previous-item</em> is either <tt>input<sub><em>i-1</em></sub></tt>,
     * or <tt>input<sub><em>ITEMS_PER_THREAD</em>-1</sub></tt> in the previous thread).  For
     * <em>thread</em><sub>0</sub>, item <tt>input<sub>0</sub></tt> is compared
     * against /p tile_prefix.
     *
     * The \p tile_prefix and \p last_tile_item are undefined in threads other than <em>thread</em><sub>0</sub>.
     *
     * \smemreuse
     *
     * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
     * \tparam FlagT                 <b>[inferred]</b> The flag type (must be an integer type)
     * \tparam FlagOp               <b>[inferred]</b> Binary boolean functor type, having input parameters <tt>(const T &a, const T &b)</tt> and returning \p true if a discontinuity exists between \p a and \p b, otherwise \p false.
     */
    template <
        int             ITEMS_PER_THREAD,
        typename        FlagT,
        typename        FlagOp>
    static __device__ __forceinline__ void Flag(
        SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
        T               (&input)[ITEMS_PER_THREAD],     ///< [in] Input items
        T               tile_prefix,                    ///< [in] <b>[<em>thread</em><sub>0</sub> only]</b> Item with which to compare the first tile item (<tt>input<sub>0</sub></tt>from <em>thread</em><sub>0</sub>).
        FlagOp          flag_op,                        ///< [in] Binary boolean flag predicate
        FlagT           (&flags)[ITEMS_PER_THREAD])     ///< [out] Discontinuity flags
    {
        T last_tile_item;   // discard
        Flag(smem_storage, input, tile_prefix, flag_op, flags, last_tile_item);
    }

};


/** @} */       // end of SimtCoop group

} // namespace cub
CUB_NS_POSTFIX
