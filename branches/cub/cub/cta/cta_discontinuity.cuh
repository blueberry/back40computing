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
 * \brief The CtaDiscontinuity type provides operations for flagging discontinuities within a list of data items partitioned across CTA threads.
 *
 * <b>Overview</b>
 * \par
 * The operations exposed by CtaDiscontinuity allow CTAs to set "head flags" for data elements that
 * are different from their predecessor (as specified by a binary boolean operator).
 *
 * \tparam T                    The data type to be exchanged.
 * \tparam CTA_THREADS          The CTA size in threads.
 * \tparam ITEMS_PER_THREAD     The number of items partitioned onto each thread.
 *
 * <b>Important Features and Considerations</b>
 * \par
 * - After any operation, a subsequent CTA barrier (<tt>__syncthreads()</tt>) is
 *   required if the supplied CtaDiscontinuity::SmemStorage is to be reused/repurposed by the CTA.
 * - Zero bank conflicts for most types.
 *
 */
template <
	int 		CTA_THREADS,			// The CTA size in threads
	typename 	T>						// The input type for which we are detecting duplicates
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
     * \brief Sets discontinuity flags for a tile of CTA items.
     *
     * Assuming a <em>blocked</em> arrangement of elements across threads,
     * <tt>flags</tt><sub><em>i</em></sub> is set non-zero for item
     * <tt>input<sub><em>i</em></sub></tt> when <tt>scan_op(</tt><em>previous-item</em>, <tt>input<sub><em>i</em></sub>)</tt>
     * is \p true (where <em>previous-item</em> is either <tt>input<sub><em>i-1</em></sub></tt>,
     * or <tt>input<sub><em>ITEMS_PER_THREAD</em>-1</sub></tt> in the previous thread).  Furthermore,
     * <tt>flags</tt><sub><em>i</em></sub> is always zero for <tt>input<sub>0</sub></tt>
     * in thread<sub>0</sub>.  The last item of the last thread is also returned to thread<sub>0</sub>.
     *
     * \smemreuse
     *
     * \tparam ITEMS_PER_THREAD     [inferred] The number of consecutive items partitioned onto each thread.
     * \tparam Flag                 [inferred] The flag type (must be an integer type)
     * \tparam FlagOp               [inferred] Binary boolean functor type, having input parameters <tt>T a, b</tt> and returning \p true if a discontinuity exists between \p a and \p b, otherwise \p false.
     */
    template <
        int             ITEMS_PER_THREAD,
        typename        Flag,
        typename        FlagOp>
    static __device__ __forceinline__ void Flag(
        SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
        T               (&input)[ITEMS_PER_THREAD],     ///< [in] Input items
        FlagOp          flag_op,                        ///< [in] Binary boolean flag predicate
        Flag            (&flags)[ITEMS_PER_THREAD],     ///< [out] Discontinuity flags
        T               &last_item)                     ///< [out] Last item of last thread (valid only in thread<sub>0</sub>)
    {
    }


};


/** @} */       // end of SimtCoop group

} // namespace cub
CUB_NS_POSTFIX
