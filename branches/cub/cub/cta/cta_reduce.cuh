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
 * The cub::CtaReduce type provides variants of parallel reduction across threads within a CTA
 */

#pragma once

#include "../cta/cta_raking_grid.cuh"
#include "../device_props.cuh"
#include "../type_utils.cuh"
#include "../operators.cuh"
#include "../thread/thread_reduce.cuh"
#include "../thread/thread_load.cuh"
#include "../thread/thread_store.cuh"
#include "../ns_wrapper.cuh"

CUB_NS_PREFIX

/// CUB namespace
namespace cub {

/**
 *  \addtogroup SimtCoop
 *  @{
 */

/**
 * \brief The CtaReduce type provides variants of parallel reduction across threads within a CTA. ![](reduce_logo.png)
 *
 * <b>Overview</b>
 * \par
 * A <em>reduction</em> (or <em>fold</em>) uses a binary combining operator to
 * compute a single aggregate from a list of input elements.  The parallel
 * operations exposed by this type assume <em>n</em>-element
 * lists that are partitioned evenly across \p CTA_THREADS threads,
 * with thread<sub><em>i</em></sub> owning the <em>i</em><sup>th</sup>
 * element (or <em>i</em><sup>th</sup> segment of consecutive elements).
 * To minimize synchronization overhead, these operations only produce a
 * valid cumulative aggregate for thread<sub>0</sub>.
 *
 * \tparam T                        The reduction input/output element type
 * \tparam CTA_THREADS              The CTA size in threads
 *
 * <b>Important Features and Considerations</b>
 * \par
 * - Supports non-commutative reduction operators.
 * - Supports partially-full CTAs (i.e., high-order threads having undefined values).
 * - Very efficient (only one synchronization barrier).
 * - Zero bank conflicts for most types.
 * - After any operation, a subsequent CTA barrier (<tt>__syncthreads()</tt>) is
 *   required if the supplied CtaReduce::SmemStorage is to be reused/repurposed by the CTA.
 * - The operations are most efficient (lowest instruction overhead) when:
 *      - The data type \p T is a built-in primitive or CUDA vector type (e.g.,
 *        \p short, \p int2, \p double, \p float2, etc.)  Otherwise the implementation may use memory
 *        fences to prevent reference reordering of non-primitive types.
 *      - \p CTA_THREADS is a multiple of the architecture's warp size
 *      - Every thread has a valid input (i.e., unguarded reduction)
 *
 * <b>Algorithm</b>
 * \par
 * These parallel reduction variants have <em>O</em>(<em>n</em>) work complexity and are implemented in three phases:
 * -# Sequential reduction in registers (if threads contribute more than one input each).  Each thread then places the partial reduction of its item(s) into shared memory.
 * -# A single-warp performs a raking upsweep across partial reductions shared each thread in the CTA.
 * -# A warp-synchronous Kogge-Stone style reduction within the raking warp to produce the total aggregate.
 * <br>
 * <br>
 * \image html cta_reduce.png
 * <center><b>Data flow for a hypothetical 16-thread CTA and 4-thread raking warp.</b></center>
 * <br>
 *
 * <b>Examples</b>
 * \par
 * - <b>Example 1:</b> Simple reduction
 * \code
 * #include <cub.cuh>
 *
 * template <int CTA_THREADS>
 * __global__ void SomeKernel(...)
 * {
 *      // Parameterize CtaReduce for use with CTA_THREADS threads on type int.
 *      typedef cub::CtaReduce<int, CTA_THREADS> CtaReduce;
 *
 *      // Declare shared memory for CtaReduce
 *      __shared__ typename CtaReduce::SmemStorage smem_storage;
 *
 *      // A segment of four input items per thread
 *      int data[4];
 *
 *      ...
 *
 *      // Compute the CTA-wide sum for thread0.
 *      int aggregate = CtaReduce::Reduce(smem_storage, data);
 *
 *      ...
 * }
 * \endcode
 * <br>
 *
 * \par
 * - <b>Example 2:</b> Guarded reduction
 * \code
 * #include <cub.cuh>
 *
 * template <int CTA_THREADS>
 * __global__ void SomeKernel(..., int num_elements)
 * {
 *      // Parameterize CtaReduce for use with CTA_THREADS threads on type int.
 *      typedef cub::CtaReduce<int, CTA_THREADS> CtaReduce;
 *
 *      // Declare shared memory for CtaReduce
 *      __shared__ typename CtaReduce::SmemStorage smem_storage;
 *
 *      // Guarded load
 *      int data;
 *      if (threadIdx.x < num_elements) data = ...;
 *
 *      // Compute the CTA-wide sum of valid elements in thread0.
 *      int aggregate = CtaReduce::Reduce(smem_storage, data, num_elements);
 * \endcode
 */
template <
    typename     T,
    int         CTA_THREADS>
class CtaReduce
{
private:

    //---------------------------------------------------------------------
    // Constants and typedefs
    //---------------------------------------------------------------------

    /**
     * Layout type for padded CTA raking grid
     */
    typedef CtaRakingGrid<CTA_THREADS, T, 1> CtaRakingGrid;


    enum
    {
        /// Number of raking threads
        RAKING_THREADS = CtaRakingGrid::RAKING_THREADS,

        /// Number of raking elements per warp synchronous raking thread
        RAKING_LENGTH = CtaRakingGrid::RAKING_LENGTH,

        /// Number of warp-synchronous steps
        WARP_SYNCH_STEPS = Log2<RAKING_THREADS>::VALUE,

        /// Cooperative work can be entirely warp synchronous
        WARP_SYNCHRONOUS = (RAKING_THREADS == CTA_THREADS),

        /// Whether or not warp-synchronous reduction should be unguarded (i.e., the warp-reduction elements is a power of two
        WARP_SYNCHRONOUS_UNGUARDED = ((RAKING_THREADS & (RAKING_THREADS - 1)) == 0),

        /// Whether or not accesses into smem are unguarded
        RAKING_UNGUARDED = CtaRakingGrid::UNGUARDED,

    };

    /// Shared memory storage layout type
    struct SmemStorage
    {
        T                                       warp_buffer[RAKING_THREADS];    ///< Buffer for warp-synchronous reduction
        typename CtaRakingGrid::SmemStorage     raking_grid;                    ///< Padded CTA raking grid
    };

public:

    /// The operations exposed by CtaReduce require shared memory of this
    /// type.  This opaque storage can be allocated directly using the
    /// <tt>__shared__</tt> keyword.  Alternatively, it can be aliased to
    /// externally allocated shared memory or <tt>union</tt>'d with other types
    /// to facilitate shared memory reuse.
    typedef SmemStorage SmemStorage;

private:

    //---------------------------------------------------------------------
    // Utility methods
    //---------------------------------------------------------------------

    /**
     * Warp reduction
     */
    template <
        bool                FULL_TILE,
        int                 RAKING_LENGTH,
        typename            ReductionOp>
    static __device__ __forceinline__ T WarpReduce(
        SmemStorage         &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
        T                   partial,            ///< [in] Calling thread's input partial reduction
        const unsigned int  &valid_threads,     ///< [in] Number valid threads (may be less than CTA_THREADS)
        ReductionOp         reduction_op)       ///< [in] Reduction operator
    {
        for (int STEP = 0; STEP < WARP_SYNCH_STEPS; STEP++)
        {
            const int OFFSET = 1 << STEP;

            // Share partial into buffer
            ThreadStore<PTX_STORE_VS>(&smem_storage.warp_buffer[threadIdx.x], partial);

            // Update partial if addend is in range
            if ((FULL_TILE && WARP_SYNCHRONOUS_UNGUARDED) || ((threadIdx.x + OFFSET) * RAKING_LENGTH < valid_threads))
            {
                T addend = ThreadLoad<PTX_LOAD_VS>(&smem_storage.warp_buffer[threadIdx.x + OFFSET]);
                partial = reduction_op(partial, addend);
            }
        }

        return partial;
    }



    /**
     * Perform a cooperative, CTA-wide reduction. The first valid_threads
     * threads each contribute one reduction partial.
     *
     * The return value is only valid for thread<sub>0</sub> (and is undefined for
     * other threads).
     */
    template <
        bool                FULL_TILE,
        typename            ReductionOp>
    static __device__ __forceinline__ T ReduceInternal(
        SmemStorage         &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
        T                   partial,            ///< [in] Calling thread's input partial reductions
        const unsigned int  &valid_threads,     ///< [in] Number of valid elements (may be less than CTA_THREADS)
        ReductionOp         reduction_op)       ///< [in] Reduction operator
    {
        if (WARP_SYNCHRONOUS)
        {
            // Short-circuit directly to warp synchronous reduction (unguarded if active threads is a power-of-two)
            partial = WarpReduce<FULL_TILE, 1>(
                smem_storage,
                partial,
                valid_threads,
                reduction_op);
        }
        else
        {
            // Place partial into shared memory grid.
            *CtaRakingGrid::PlacementPtr(smem_storage.raking_grid) = partial;

            __syncthreads();

            // Reduce parallelism to one warp
            if (threadIdx.x < RAKING_THREADS)
            {
                // Raking reduction in grid
                T *raking_segment = CtaRakingGrid::RakingPtr(smem_storage.raking_grid);
                partial = raking_segment[0];

                #pragma unroll
                for (int ITEM = 1; ITEM < RAKING_LENGTH; ITEM++)
                {
                    // Update partial if addend is in range
                    if ((FULL_TILE && RAKING_UNGUARDED) || ((threadIdx.x * RAKING_LENGTH) + ITEM < valid_threads))
                    {
                        partial = reduction_op(partial, raking_segment[ITEM]);
                    }
                }

                // Warp synchronous reduction
                partial = WarpReduce<(FULL_TILE && RAKING_UNGUARDED), RAKING_LENGTH>(
                    smem_storage,
                    partial,
                    valid_threads,
                    reduction_op);
            }
        }

        return partial;
    }

public:

    /******************************************************************//**
     * \name Summation reductions
     *********************************************************************/
    //@{

    /**
     * \brief Computes a CTA-wide reduction for thread<sub>0</sub> using addition (+) as the reduction operator.  Each thread contributes one input element.
     *
     * The return value is undefined in threads other than thread<sub>0</sub>.
     *
     * \smemreuse
     */
    static __device__ __forceinline__ T Reduce(
        SmemStorage     &smem_storage,              ///< [in] Shared reference to opaque SmemStorage layout
        T               input)                      ///< [in] Calling thread's input
    {
        Sum<T> reduction_op;
        return Reduce(smem_storage, input, reduction_op);
    }

    /**
     * \brief Computes a CTA-wide reduction for thread<sub>0</sub> using addition (+) as the reduction operator.  Each thread contributes an array of consecutive input elements.
     *
     * The return value is undefined in threads other than thread<sub>0</sub>.
     *
     * \smemreuse
     *
     * \tparam ITEMS_PER_THREAD     [inferred] The number of consecutive items partitioned onto each thread.
     */
    template <int ITEMS_PER_THREAD>
    static __device__ __forceinline__ T Reduce(
        SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
        T               (&inputs)[ITEMS_PER_THREAD])    ///< [in] Calling thread's input segment
    {
        Sum<T> reduction_op;
        return Reduce(smem_storage, inputs, reduction_op);
    }


    /**
     * \brief Computes a CTA-wide reduction for thread<sub>0</sub> using addition (+) as the reduction operator.  The first \p valid_threads threads each contribute one input element.
     *
     * \smemreuse
     *
     * The return value is undefined in threads other than thread<sub>0</sub>.
     */
    static __device__ __forceinline__ T Reduce(
        SmemStorage         &smem_storage,          ///< [in] Shared reference to opaque SmemStorage layout
        T                   input,                   ///< [in] Calling thread's input
        const unsigned int  &valid_threads)         ///< [in] Number of threads containing valid elements (may be less than CTA_THREADS)
    {
        Sum<T> reduction_op;
        Reduce(smem_storage, input, valid_threads);
    }


    //@}
    /******************************************************************//**
     * \name Generic reductions
     *********************************************************************/
    //@{


    /**
     * \brief Computes a CTA-wide reduction for thread<sub>0</sub> using the specified binary reduction functor.  Each thread contributes one input element.
     *
     * The return value is undefined in threads other than thread<sub>0</sub>.
     *
     * \smemreuse
     *
     * \tparam ReductionOp          [inferred] Binary reduction functor type (a model of <a href="http://www.sgi.com/tech/stl/BinaryFunction.html">Binary Function</a>).
     */
    template <typename ReductionOp>
    static __device__ __forceinline__ T Reduce(
        SmemStorage     &smem_storage,              ///< [in] Shared reference to opaque SmemStorage layout
        T               input,                      ///< [in] Calling thread's input
        ReductionOp     reduction_op)               ///< [in] Binary associative reduction functor
    {
        return Reduce(smem_storage, input, CTA_THREADS, reduction_op);
    }


    /**
     * \brief Computes a CTA-wide reduction for thread<sub>0</sub> using the specified binary reduction functor.  Each thread contributes an array of consecutive input elements.
     *
     * The return value is undefined in threads other than thread<sub>0</sub>.
     *
     * \smemreuse
     *
     * \tparam ITEMS_PER_THREAD     [inferred] The number of consecutive items partitioned onto each thread.
     * \tparam ReductionOp          [inferred] Binary reduction functor type (a model of <a href="http://www.sgi.com/tech/stl/BinaryFunction.html">Binary Function</a>).
     */
    template <
        int ITEMS_PER_THREAD,
        typename ReductionOp>
    static __device__ __forceinline__ T Reduce(
        SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
        T               (&inputs)[ITEMS_PER_THREAD],    ///< [in] Calling thread's input segment
        ReductionOp     reduction_op)                   ///< [in] Binary associative reduction functor
    {
        // Reduce partials
        T partial = ThreadReduce(inputs, reduction_op);
        return Reduce(smem_storage, partial, CTA_THREADS, reduction_op);
    }


    /**
     * \brief Computes a CTA-wide reduction for thread<sub>0</sub> using the specified binary reduction functor.  The first \p valid_threads threads each contribute one input element.
     *
     * The return value is undefined in threads other than thread<sub>0</sub>.
     *
     * \smemreuse
     *
     * \tparam ReductionOp          [inferred] Binary reduction functor type (a model of <a href="http://www.sgi.com/tech/stl/BinaryFunction.html">Binary Function</a>).
     */
    template <typename ReductionOp>
    static __device__ __forceinline__ T Reduce(
        SmemStorage         &smem_storage,          ///< [in] Shared reference to opaque SmemStorage layout
        T                   input,                  ///< [in] Calling thread's input
        ReductionOp         reduction_op,           ///< [in] Binary associative reduction functor
        const unsigned int  &valid_threads)         ///< [in] Number of threads containing valid elements (may be less than CTA_THREADS)
    {
        // Determine if we don't need bounds checking
        if (valid_threads == CTA_THREADS)
        {
            return ReduceInternal<true>(smem_storage, input, valid_threads, reduction_op);
        }
        else
        {
            return ReduceInternal<false>(smem_storage, input, valid_threads, reduction_op);
        }
    }

    //@}

};

/** @} */       // end of SimtCoop group

} // namespace cub
CUB_NS_POSTFIX
