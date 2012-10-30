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
 * The cub::CtaScan type provides variants of parallel prefix scan across threads within a CUDA CTA.
 */

#pragma once

#include "../device_props.cuh"
#include "../type_utils.cuh"
#include "../operators.cuh"
#include "../warp/warp_scan.cuh"
#include "../thread/thread_reduce.cuh"
#include "../thread/thread_scan.cuh"
#include "../ns_wrapper.cuh"

CUB_NS_PREFIX

/// CUB namespace
namespace cub {


//-----------------------------------------------------------------------------
// Policy
//-----------------------------------------------------------------------------

/// Tuning policy for cub::CtaScan
enum CtaScanPolicy
{
    CTA_SCAN_RAKING,        ///< Use an work-efficient, but longer-latency algorithm (raking reduce-then-scan).  Useful when the GPU is fully occupied.
    CTA_SCAN_WARPSCANS,     ///< Use an work-inefficient, but shorter-latency algorithm (tiled warpscans).  Useful when the GPU is under-occupied.
};


/**
 * \brief The CtaScan type provides variants of parallel prefix scan across threads within a CUDA CTA.  ![](scan_logo.png)
 *
 * \tparam T                The reduction input/output element type
 * \tparam CTA_THREADS      The CTA size in threads
 * \tparam POLICY           [optional] cub::CtaScanPolicy tuning policy enumeration.  Default = cub::CTA_SCAN_RAKING.
 *
 * <b>Overview</b>
 * \par
 * Given a list of input elements and a binary reduction operator, <em>prefix scan</em>
 * produces an output list where each element is computed to be the reduction
 * of the elements occurring earlier in the input list.  <em>Prefix sum</em>
 * connotes a prefix scan with the addition operator. The term \em inclusive means
 * that each result includes the corresponding input operand in the partial sum.
 * The term \em exclusive means that each result does not include the corresponding
 * input operand in the partial reduction.
 *
 * \par
 * The parallel operations exposed by this type assume <em>n</em>-element
 * lists that are partitioned evenly among \p CTA_THREADS threads,
 * with thread<sub><em>i</em></sub> owning the <em>i</em><sup>th</sup>
 * element (or <em>i</em><sup>th</sup> segment of consecutive elements).
 *
 * <b>Features</b>
 * \par
 * - Supports non-commutative scan operators.
 * - Very efficient (only two synchronization barriers).
 * - Zero bank conflicts for most types.
 *
 * <b>Algorithm</b>
 * \par
 * The CtaScan class can be configured to use one of two alternative algorithms:
 *
 * \par
 *   -# <b>Algorithm cub::CTA_SCAN_RAKING</b>:
 *
 *     These variants have <em>O</em>(<em>n</em>) work complexity and are comprised of five phases:
 *     -# Upsweep sequential reduction in registers (if threads contribute more than one input each).  Each thread then places the partial reduction of its item(s) into shared memory.
 *     -# Upsweep sequential reduction in shared memory.  Threads within a single warp rake across segments of shared partial reductions.
 *     -# A warp-synchronous Kogge-Stone style exclusive scan within the raking warp.
 *     -# Downsweep sequential exclusive scan in shared memory.  Threads within a single warp rake across segments of shared partial reductions, seeded with the warp-scan output.
 *     -# Downsweep sequential scan in registers (if threads contribute more than one input), seeded with the raking scan output.
 *     <br>
 *     <br>
 *     \image html cta_scan.png
 *     <center><b>\p CTA_SCAN_RAKING data flow for a hypothetical 16-thread CTA and 4-thread raking warp.</b></center>
 *     <br>
 *   -# <b>Algorithm cub::CTA_SCAN_WARPSCANS</b>:
 *
 *     These variants have <em>O</em>(<em>n</em>log<em>n</em>) work complexity and are comprised of five phases:
 *     <br>
 *
 * <b>Important Considerations</b>
 * \par
 * - After any CtaScan operation, a subsequent CTA barrier (<tt>__syncthreads</tt>) is
 *   required if the supplied CtaScan::SmemStorage is to be reused/repurposed by the CTA.
 * - The operations are most efficient (lowest instruction overhead) when:
 *      - The data type \p T is a built-in primitive or CUDA vector type (e.g.,
 *        \p short, \p int2, \p double, \p float2, etc.)  Otherwise the implementation may use memory
 *        fences to prevent reference reordering of non-primitive types.
 *      - \p CTA_THREADS is a multiple of the architecture's warp size
 * - To minimize synchronization overhead for operations involving the cumulative
 *   \p aggregate and \p cta_prefix, these values are only valid in thread<sub>0</sub>.
 *
 * <b>Examples</b>
 * \par
 * - <b>Example 1:</b> Simple exclusive prefix sum
 *      \code
 *      #include <cub.cuh>
 *
 *      template <int CTA_THREADS>
 *      __global__ void SomeKernel(...)
 *      {
 *
 *      \endcode
 *
 * \par
 * - <b>Example 2:</b> More sophisticated exclusive prefix sum
 *      \code
 *      #include <cub.cuh>
 *
 *      template <int CTA_THREADS>
 *      __global__ void SomeKernel(...)
 *      {
 *
 *      \endcode
 */
template <
    typename        T,
    int             CTA_THREADS,
    CtaScanPolicy   POLICY = CTA_SCAN_RAKING>
class CtaScan
{
private:

    //---------------------------------------------------------------------
    // Type definitions and constants
    //---------------------------------------------------------------------

    /**
     * Layout type for padded CTA raking grid
     */
    typedef CtaRakingGrid<CTA_THREADS, T> CtaRakingGrid;

    enum
    {
        /// Number of active warps
        WARPS = (CTA_THREADS + DeviceProps::WARP_THREADS - 1) / DeviceProps::WARP_THREADS,

        /// Number of raking threads
        RAKING_THREADS = CtaRakingGrid::RAKING_THREADS,

        /// Number of raking elements per warp synchronous raking thread
        RAKING_LENGTH = CtaRakingGrid::RAKING_LENGTH,

        /// Cooperative work can be entirely warp synchronous
        WARP_SYNCHRONOUS = (CTA_THREADS == RAKING_THREADS),
    };

    ///  Raking warp-scan utility type
    typedef WarpScan<T, 1, RAKING_THREADS> WarpScan;

    /// Shared memory storage layout type
    struct SmemStorage
    {
        typename WarpScan::SmemStorage          warp_scan;      ///< Buffer for warp-synchronous scan
        typename CtaRakingGrid::SmemStorage     raking_grid;    ///< Padded CTA raking grid
    };

public:

    /// The operations exposed by CtaScan require shared memory of this
    /// type.  This opaque storage can be allocated directly using the
    /// <tt>__shared__</tt> keyword.  Alternatively, it can be aliased to
    /// externally allocated shared memory or <tt>union</tt>'d with other types
    /// to facilitate shared memory reuse.
    typedef SmemStorage SmemStorage;


    /******************************************************************//**
     * @name Exclusive prefix scan variants
     *********************************************************************/
    //@{


    /**
     * \brief Computes an exclusive CTA-wide prefix scan using the specified binary scan functor.  Each thread contributes one input element.  Also computes the CTA-wide \p aggregate of all inputs for thread<sub>0</sub>.
     *
     * The \p aggregate is undefined in threads other than thread<sub>0</sub>.
     *
     * \smemreuse
     *
     * \tparam ScanOp   [inferred] Binary scan functor type (a model of <a href="http://www.sgi.com/tech/stl/BinaryFunction.html">Binary Function</a>).
     */
    template <typename ScanOp>
    static __device__ __forceinline__ void ExclusiveScan(
        SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
        T               input,              ///< [in] Input
        T               &output,            ///< [out] Output (may be aliased to input)
        T               identity,           ///< [in] Identity value.
        ScanOp          scan_op,            ///< [in] Binary scan operator
        T               &aggregate)         ///< [out] Total aggregate (valid in lane-0).
    {
        if (WARP_SYNCHRONOUS)
        {
            // Short-circuit directly to warp scan
            WarpScan::ExclusiveScan(
                smem_storage.warp_scan,
                input,
                output,
                identity,
                scan_op,
                aggregate);
        }
        else
        {
            // Place thread partial into shared memory raking grid
            T *placement_ptr = CtaRakingGrid::PlacementPtr(smem_storage.raking_grid);
            *placement_ptr = input;

            __syncthreads();

            // Reduce parallelism down to just raking threads
            if (threadIdx.x < RAKING_THREADS)
            {
                // Raking upsweep reduction in grid
                T *raking_ptr = CtaRakingGrid::RakingPtr(smem_storage.raking_grid);
                T raking_partial = ThreadReduce<RAKING_LENGTH>(raking_ptr, scan_op);

                // Exclusive warp synchronous scan
                WarpScan::ExclusiveScan(
                    smem_storage.warp_scan,
                    raking_partial,
                    raking_partial,
                    identity,
                    scan_op,
                    aggregate);

                // Exclusive raking downsweep scan
                ThreadScanExclusive<RAKING_LENGTH>(raking_ptr, raking_ptr, scan_op, raking_partial);

                if (!CtaRakingGrid::UNGUARDED)
                {
                    // CTA size isn't a multiple of warp size, so grab aggregate from the appropriate raking cell
                    aggregate = *CtaRakingGrid::PlacementPtr(smem_storage.raking_grid, 0, CTA_THREADS);
                }
            }

            __syncthreads();

            // Grab thread prefix from shared memory
            output = *placement_ptr;

        }
    }


    /**
     * \brief Computes an exclusive CTA-wide prefix scan using the specified binary scan functor.  Each thread contributes an array of consecutive input elements.  Also computes the CTA-wide \p aggregate of all inputs for thread<sub>0</sub>.
     *
     * The \p aggregate is undefined in threads other than thread<sub>0</sub>.
     *
     * \smemreuse
     *
     * \tparam ITEMS_PER_THREAD     [inferred] The number of consecutive items contributed by each thread.
     * \tparam ScanOp               [inferred] Binary scan functor type (a model of <a href="http://www.sgi.com/tech/stl/BinaryFunction.html">Binary Function</a>).
     */
    template <
        int             ITEMS_PER_THREAD,
        typename        ScanOp>
    static __device__ __forceinline__ void ExclusiveScan(
        SmemStorage       &smem_storage,                ///< [in] Shared reference to opaque SmemStorage layout
        T                 (&input)[ITEMS_PER_THREAD],   ///< [in] Input
        T                 (&output)[ITEMS_PER_THREAD],  ///< [out] Output (may be aliased to input)
        T                 identity,                     ///< [in] Identity value.
        ScanOp            scan_op,                      ///< [in] Binary scan operator
        T                 &aggregate)                   ///< [out] Total aggregate (valid in lane-0).
    {
        // Reduce consecutive thread items in registers
        T thread_partial = ThreadReduce(input, scan_op);

        // Exclusive CTA-scan
        ExclusiveScan(smem_storage, thread_partial, thread_partial, identity, scan_op, aggregate);

        // Exclusive scan in registers with prefix
        ThreadScanExclusive(input, output, scan_op, thread_partial);
    }


    /**
     * \brief Computes an exclusive CTA-wide prefix scan using the specified binary scan functor.  Each thread contributes one input element.  The \p cta_prefix value from thread<sub>0</sub> is applied to all scan outputs.  Also computes the CTA-wide \p aggregate of all inputs for thread<sub>0</sub>.  The \p cta_prefix is further updated by the value of \p aggregate.
     *
     * The \p aggregate and \p cta_prefix are undefined in threads other than thread<sub>0</sub>.
     *
     * \smemreuse
     *
     * \tparam ScanOp               [inferred] Binary scan functor type (a model of <a href="http://www.sgi.com/tech/stl/BinaryFunction.html">Binary Function</a>).
     */
    template <typename ScanOp>
    static __device__ __forceinline__ void ExclusiveScan(
        SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
        T               input,                          ///< [in] Input
        T               &output,                        ///< [out] Output (may be aliased to input)
        T               identity,                       ///< [in] Identity value.
        ScanOp          scan_op,                        ///< [in] Binary scan operator
        T               &aggregate,                     ///< [out] Total aggregate (valid in lane-0).
        T               &cta_prefix)                    ///< [in-out] Cta-wide prefix to scan (valid in lane-0).
    {
        if (WARP_SYNCHRONOUS)
        {
            // Short-circuit directly to warp scan
            WarpScan::ExclusiveScan(
                smem_storage.warp_scan,
                input,
                output,
                identity,
                scan_op,
                aggregate,
                cta_prefix);
        }
        else
        {
            // Place thread partial into shared memory raking grid
            T *placement_ptr = CtaRakingGrid::PlacementPtr(smem_storage.raking_grid);
            *placement_ptr = input;

            __syncthreads();

            // Reduce parallelism down to just raking threads
            if (threadIdx.x < RAKING_THREADS)
            {
                // Raking upsweep reduction in grid
                T *raking_ptr = CtaRakingGrid::RakingPtr(smem_storage.raking_grid);
                T raking_partial = ThreadReduce<RAKING_LENGTH>(raking_ptr, scan_op);

                // Exclusvie warp synchronous scan
                WarpScan::ExclusiveScan(
                    smem_storage.warp_scan,
                    raking_partial,
                    raking_partial,
                    identity,
                    scan_op,
                    aggregate,
                    cta_prefix);

                // Exclusive raking downsweep scan
                ThreadScanExclusive<RAKING_LENGTH>(raking_ptr, raking_ptr, scan_op, raking_partial);

                if (!CtaRakingGrid::UNGUARDED)
                {
                    // CTA size isn't a multiple of warp size, so grab aggregate from the appropriate raking cell
                    aggregate = *CtaRakingGrid::PlacementPtr(smem_storage.raking_grid, 0, CTA_THREADS);
                    cta_prefix = scan_op(cta_prefix, aggregate);
                }
            }

            __syncthreads();

            // Grab thread prefix from shared memory
            output = *placement_ptr;
        }
    }


    /**
     * \brief Computes an exclusive CTA-wide prefix scan using the specified binary scan functor.  Each thread contributes an array of consecutive input elements.  The \p cta_prefix value from thread<sub>0</sub> is applied to all scan outputs.  Also computes the CTA-wide \p aggregate of all inputs for thread<sub>0</sub>.  The \p cta_prefix is further updated by the value of \p aggregate.
     *
     * The \p aggregate and \p cta_prefix are undefined in threads other than thread<sub>0</sub>.
     *
     * \smemreuse
     *
     * \tparam ITEMS_PER_THREAD     [inferred] The number of consecutive items contributed by each thread.
     * \tparam ScanOp               [inferred] Binary scan functor type (a model of <a href="http://www.sgi.com/tech/stl/BinaryFunction.html">Binary Function</a>).
     */
    template <
        int             ITEMS_PER_THREAD,
        typename         ScanOp>
    static __device__ __forceinline__ void ExclusiveScan(
        SmemStorage        &smem_storage,               ///< [in] Shared reference to opaque SmemStorage layout
        T                 (&input)[ITEMS_PER_THREAD],   ///< [in] Input
        T                 (&output)[ITEMS_PER_THREAD],  ///< [out] Output (may be aliased to input)
        T                 identity,                     ///< [in] Identity value.
        ScanOp            scan_op,                      ///< [in] Binary scan operator
        T                 &aggregate,                   ///< [out] Total aggregate (valid in lane-0).
        T                 &cta_prefix)                  ///< [in-out] Cta-wide prefix to scan (valid in lane-0).
    {
        // Reduce consecutive thread items in registers
        T thread_partial = ThreadReduce(input, scan_op);

        // Exclusive CTA-scan
        ExclusiveScan(smem_storage, thread_partial, thread_partial, identity, scan_op, aggregate, cta_prefix);

        // Exclusive scan in registers with prefix
        ThreadScanExclusive(input, output, scan_op, thread_partial);
    }


    /**
     * \brief Computes an exclusive CTA-wide prefix scan using the specified binary scan functor.  Each thread contributes one input element.
     *
     * \smemreuse
     *
     * \tparam ScanOp               [inferred] Binary scan functor type (a model of <a href="http://www.sgi.com/tech/stl/BinaryFunction.html">Binary Function</a>).
     */
    template <typename ScanOp>
    static __device__ __forceinline__ void ExclusiveScan(
        SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
        T               input,                          ///< [in] Input
        T               &output,                        ///< [out] Output (may be aliased to input)
        T               identity,                       ///< [in] Identity value.
        ScanOp          scan_op)                        ///< [in] Binary scan operator
    {
        T aggregate;
        ExclusiveScan(smem_storage, input, output, identity, scan_op, aggregate);
    }



    /**
     * \brief Computes an exclusive CTA-wide prefix scan using the specified binary scan functor.  Each thread contributes an array of consecutive input elements.
     *
     * \smemreuse
     *
     * \tparam ITEMS_PER_THREAD     [inferred] The number of consecutive items contributed by each thread.
     * \tparam ScanOp               [inferred] Binary scan functor type (a model of <a href="http://www.sgi.com/tech/stl/BinaryFunction.html">Binary Function</a>).
     */
    template <
        int             ITEMS_PER_THREAD,
        typename         ScanOp>
    static __device__ __forceinline__ void ExclusiveScan(
        SmemStorage       &smem_storage,                ///< [in] Shared reference to opaque SmemStorage layout
        T                 (&input)[ITEMS_PER_THREAD],   ///< [in] Input
        T                 (&output)[ITEMS_PER_THREAD],  ///< [out] Output (may be aliased to input)
        T                 identity,                     ///< [in] Identity value.
        ScanOp            scan_op)                      ///< [in] Binary scan operator
    {
        // Reduce consecutive thread items in registers
        T thread_partial = ThreadReduce(input, scan_op);

        // Exclusive CTA-scan
        ExclusiveScan(smem_storage, thread_partial, thread_partial, identity, scan_op);

        // Exclusive scan in registers with prefix
        ThreadScanExclusive(input, output, scan_op, thread_partial);
    }


    //@}
    /******************************************************************//**
     * @name Exclusive prefix scan variants (without supplied identity)
     *********************************************************************/
    //@{


    /**
     * \brief Exclusive CTA-wide prefix scan also producing aggregate. With no identity value, the output computed for thread<sub>0</sub> is invalid.
     *
     * \smemreuse
     *
     * \tparam ScanOp               [inferred] Binary scan functor type (a model of <a href="http://www.sgi.com/tech/stl/BinaryFunction.html">Binary Function</a>).
     */
    template <typename ScanOp>
    static __device__ __forceinline__ void ExclusiveScan(
        SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
        T               input,                          ///< [in] Input
        T               &output,                        ///< [out] Output (may be aliased to input)
        ScanOp          scan_op,                        ///< [in] Binary scan operator
        T               &aggregate)                     ///< [out] Total aggregate (valid in lane-0).
    {
        if (WARP_SYNCHRONOUS)
        {
            // Short-circuit directly to warp scan
            WarpScan::ExclusiveScan(
                smem_storage.warp_scan,
                input,
                output,
                scan_op,
                aggregate);
        }
        else
        {
            // Place thread partial into shared memory raking grid
            T *placement_ptr = CtaRakingGrid::PlacementPtr(smem_storage.raking_grid);
            *placement_ptr = input;

            __syncthreads();

            // Reduce parallelism down to just raking threads
            if (threadIdx.x < RAKING_THREADS)
            {
                // Raking upsweep reduction in grid
                T *raking_ptr = CtaRakingGrid::RakingPtr(smem_storage.raking_grid);
                T raking_partial = ThreadReduce<RAKING_LENGTH>(raking_ptr, scan_op);

                // Exclusive warp synchronous scan
                WarpScan::ExclusiveScan(
                    smem_storage.warp_scan,
                    raking_partial,
                    raking_partial,
                    scan_op,
                    aggregate);

                // Exclusive raking downsweep scan
                ThreadScanExclusive<RAKING_LENGTH>(raking_ptr, raking_ptr, scan_op, raking_partial, (threadIdx.x != 0));

                if (!CtaRakingGrid::UNGUARDED)
                {
                    // CTA size isn't a multiple of warp size, so grab aggregate from the appropriate raking cell
                    aggregate = *CtaRakingGrid::PlacementPtr(smem_storage.raking_grid, 0, CTA_THREADS);
                }
            }

            __syncthreads();

            // Grab thread prefix from shared memory
            output = *placement_ptr;
        }
    }


    /**
     * \brief 2D exclusive CTA-wide prefix scan also producing aggregate.  With no identity value, the first output element computed for thread<sub>0</sub> is invalid.
     *
     * \smemreuse
     *
     * \tparam ITEMS_PER_THREAD     [inferred] The number of consecutive items contributed by each thread.
     * \tparam ScanOp               [inferred] Binary scan functor type (a model of <a href="http://www.sgi.com/tech/stl/BinaryFunction.html">Binary Function</a>).
     */
    template <
        int             ITEMS_PER_THREAD,
        typename         ScanOp>
    static __device__ __forceinline__ void ExclusiveScan(
        SmemStorage        &smem_storage,               ///< [in] Shared reference to opaque SmemStorage layout
        T                 (&input)[ITEMS_PER_THREAD],   ///< [in] Input
        T                 (&output)[ITEMS_PER_THREAD],  ///< [out] Output (may be aliased to input)
        ScanOp          scan_op,                        ///< [in] Binary scan operator
        T               &aggregate)                     ///< [out] Total aggregate (valid in lane-0).
    {
        // Reduce consecutive thread items in registers
        T thread_partial = ThreadReduce(input, scan_op);

        // Exclusive CTA-scan
        ExclusiveScan(smem_storage, thread_partial, thread_partial, scan_op, aggregate);

        // Exclusive scan in registers with prefix
        ThreadScanExclusive(input, output, scan_op, thread_partial, (threadIdx.x != 0));
    }


    /**
     * \brief Exclusive CTA-wide prefix scan also producing aggregate and consuming/producing cta_prefix.  With no identity value, the first output element computed for thread<sub>0</sub> is invalid.
     *
     * \smemreuse
     *
     * \tparam ScanOp               [inferred] Binary scan functor type (a model of <a href="http://www.sgi.com/tech/stl/BinaryFunction.html">Binary Function</a>).
     */
    template <typename ScanOp>
    static __device__ __forceinline__ void ExclusiveScan(
        SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
        T               input,                          ///< [in] Input
        T               &output,                        ///< [out] Output (may be aliased to input)
        ScanOp          scan_op,                        ///< [in] Binary scan operator
        T               &aggregate,                     ///< [out] Total aggregate (valid in lane-0).
        T               &cta_prefix)                    ///< [in-out] Cta-wide prefix to scan (valid in lane-0).
    {
        if (WARP_SYNCHRONOUS)
        {
            // Short-circuit directly to warp scan
            WarpScan::ExclusiveScan(
                smem_storage.warp_scan,
                input,
                output,
                scan_op,
                aggregate,
                cta_prefix);
        }
        else
        {
            // Place thread partial into shared memory raking grid
            T *placement_ptr = CtaRakingGrid::PlacementPtr(smem_storage.raking_grid);
            *placement_ptr = input;

            __syncthreads();

            // Reduce parallelism down to just raking threads
            if (threadIdx.x < RAKING_THREADS)
            {
                // Raking upsweep reduction in grid
                T *raking_ptr = CtaRakingGrid::RakingPtr(smem_storage.raking_grid);
                T raking_partial = ThreadReduce<RAKING_LENGTH>(raking_ptr, scan_op);

                // Exclusive warp synchronous scan
                WarpScan::ExclusiveScan(
                    smem_storage.warp_scan,
                    raking_partial,
                    raking_partial,
                    scan_op,
                    aggregate,
                    cta_prefix);

                // Exclusive raking downsweep scan
                ThreadScanExclusive<RAKING_LENGTH>(raking_ptr, raking_ptr, scan_op, raking_partial);

                if (!CtaRakingGrid::UNGUARDED)
                {
                    // CTA size isn't a multiple of warp size, so grab aggregate from the appropriate raking cell
                    aggregate = *CtaRakingGrid::PlacementPtr(smem_storage.raking_grid, 0, CTA_THREADS);
                    cta_prefix = scan_op(cta_prefix, aggregate);
                }
            }

            __syncthreads();

            // Grab thread prefix from shared memory
            output = *placement_ptr;
        }
    }


    /**
     * \brief 2D exclusive CTA-wide prefix scan also producing aggregate and consuming/producing cta_prefix.  With no identity value, the first output element computed for thread<sub>0</sub> is invalid.
     *
     * \smemreuse
     *
     * \tparam ITEMS_PER_THREAD     [inferred] The number of consecutive items contributed by each thread.
     * \tparam ScanOp               [inferred] Binary scan functor type (a model of <a href="http://www.sgi.com/tech/stl/BinaryFunction.html">Binary Function</a>).
     */
    template <
        int             ITEMS_PER_THREAD,
        typename         ScanOp>
    static __device__ __forceinline__ void ExclusiveScan(
        SmemStorage        &smem_storage,               ///< [in] Shared reference to opaque SmemStorage layout
        T                 (&input)[ITEMS_PER_THREAD],   ///< [in] Input
        T                 (&output)[ITEMS_PER_THREAD],  ///< [out] Output (may be aliased to input)
        ScanOp            scan_op,                      ///< [in] Binary scan operator
        T                 &aggregate,                   ///< [out] Total aggregate (valid in lane-0).
        T                 &cta_prefix)                  ///< [in-out] Cta-wide prefix to scan (valid in lane-0).
    {
        // Reduce consecutive thread items in registers
        T thread_partial = ThreadReduce(input, scan_op);

        // Exclusive CTA-scan
        ExclusiveScan(smem_storage, thread_partial, thread_partial, scan_op, aggregate, cta_prefix);

        // Exclusive scan in registers with prefix
        ThreadScanExclusive(input, output, scan_op, thread_partial);
    }


    /**
     * \brief Exclusive CTA-wide prefix scan.  With no identity value, the output computed for thread<sub>0</sub> is invalid.
     *
     * \smemreuse
     *
     * \tparam ScanOp               [inferred] Binary scan functor type (a model of <a href="http://www.sgi.com/tech/stl/BinaryFunction.html">Binary Function</a>).
     */
    template <typename ScanOp>
    static __device__ __forceinline__ void ExclusiveScan(
        SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
        T               input,                          ///< [in] Input
        T               &output,                        ///< [out] Output (may be aliased to input)
        ScanOp          scan_op)                        ///< [in] Binary scan operator
    {
        T aggregate;
        ExclusiveScan(smem_storage, input, output, scan_op, aggregate);
    }


    /**
     * \brief 2D exclusive CTA-wide prefix scan.  With no identity value, the first output element computed for thread<sub>0</sub> is invalid.
     *
     * \smemreuse
     *
     * \tparam ITEMS_PER_THREAD     [inferred] The number of consecutive items contributed by each thread.
     * \tparam ScanOp               [inferred] Binary scan functor type (a model of <a href="http://www.sgi.com/tech/stl/BinaryFunction.html">Binary Function</a>).
     */
    template <
        int             ITEMS_PER_THREAD,
        typename         ScanOp>
    static __device__ __forceinline__ void ExclusiveScan(
        SmemStorage        &smem_storage,               ///< [in] Shared reference to opaque SmemStorage layout
        T                 (&input)[ITEMS_PER_THREAD],   ///< [in] Input
        T                 (&output)[ITEMS_PER_THREAD],  ///< [out] Output (may be aliased to input)
        ScanOp            scan_op)                      ///< [in] Binary scan operator
    {
        // Reduce consecutive thread items in registers
        T thread_partial = ThreadReduce(input, scan_op);

        // Exclusive CTA-scan
        ExclusiveScan(smem_storage, thread_partial, thread_partial, scan_op);

        // Exclusive scan in registers with prefix
        ThreadScanExclusive(input, output, scan_op, thread_partial, (threadIdx.x != 0));
    }


    //@}
    /******************************************************************//**
     * @name Exclusive prefix sum variants
     *********************************************************************/
    //@{


    /**
     * \brief Exclusive CTA-wide prefix sum also producing aggregate
     *
     * \smemreuse
     */
    static __device__ __forceinline__ void ExclusiveSum(
        SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
        T               input,                          ///< [in] Input
        T               &output,                        ///< [out] Output (may be aliased to input)
        T               &aggregate)                     ///< [out] Total aggregate (valid in lane-0).
    {
        if (WARP_SYNCHRONOUS)
        {
            // Short-circuit directly to warp scan
            WarpScan::ExclusiveSum(
                smem_storage.warp_scan,
                input,
                output,
                aggregate);
        }
        else
        {
            // Raking scan
            Sum<T> scan_op;

            // Place thread partial into shared memory raking grid
            T *placement_ptr = CtaRakingGrid::PlacementPtr(smem_storage.raking_grid);
            *placement_ptr = input;

            __syncthreads();

            // Reduce parallelism down to just raking threads
            if (threadIdx.x < RAKING_THREADS)
            {
                // Raking upsweep reduction in grid
                T *raking_ptr = CtaRakingGrid::RakingPtr(smem_storage.raking_grid);
                T raking_partial = ThreadReduce<RAKING_LENGTH>(raking_ptr, scan_op);

                // Exclusive warp synchronous scan
                WarpScan::ExclusiveSum(
                    smem_storage.warp_scan,
                    raking_partial,
                    raking_partial,
                    aggregate);

                // Exclusive raking downsweep scan
                ThreadScanExclusive<RAKING_LENGTH>(raking_ptr, raking_ptr, scan_op, raking_partial, (threadIdx.x != 0));

                if (!CtaRakingGrid::UNGUARDED)
                {
                    // CTA size isn't a multiple of warp size, so grab aggregate from the appropriate raking cell
                    aggregate = *CtaRakingGrid::PlacementPtr(smem_storage.raking_grid, 0, CTA_THREADS);
                }
            }

            __syncthreads();

            // Grab thread prefix from shared memory
            output = *placement_ptr;
        }
    }


    /**
     * \brief 2D exclusive CTA-wide prefix sum also producing aggregate.
     *
     * \smemreuse
     *
     * \tparam ITEMS_PER_THREAD     [inferred] The number of consecutive items contributed by each thread.
     */
    template <int ITEMS_PER_THREAD>
    static __device__ __forceinline__ void ExclusiveSum(
        SmemStorage        &smem_storage,               ///< [in] Shared reference to opaque SmemStorage layout
        T                 (&input)[ITEMS_PER_THREAD],   ///< [in] Input
        T                 (&output)[ITEMS_PER_THREAD],  ///< [out] Output (may be aliased to input)
        T                 &aggregate)                   ///< [out] Total aggregate (valid in lane-0).
    {
        // Reduce consecutive thread items in registers
        Sum<T> scan_op;
        T thread_partial = ThreadReduce(input, scan_op);

        // Exclusive CTA-scan
        ExclusiveSum(smem_storage, thread_partial, thread_partial, aggregate);

        // Exclusive scan in registers with prefix
        ThreadScanExclusive(input, output, scan_op, thread_partial, (threadIdx.x != 0));
    }


    /**
     * \brief Exclusive CTA-wide prefix sum also producing aggregate and consuming/producing cta_prefix.
     *
     * \smemreuse
     */
    static __device__ __forceinline__ void ExclusiveSum(
        SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
        T               input,                          ///< [in] Input
        T               &output,                        ///< [out] Output (may be aliased to input)
        T               &aggregate,                     ///< [out] Total aggregate (valid in lane-0).
        T               &cta_prefix)                    ///< [in-out] Cta-wide prefix to scan (valid in lane-0).
    {
        if (WARP_SYNCHRONOUS)
        {
            // Short-circuit directly to warp scan
            WarpScan::ExclusiveSum(
                smem_storage.warp_scan,
                input,
                output,
                aggregate,
                cta_prefix);
        }
        else
        {
            // Raking scan
            Sum<T> scan_op;

            // Place thread partial into shared memory raking grid
            T *placement_ptr = CtaRakingGrid::PlacementPtr(smem_storage.raking_grid);
            *placement_ptr = input;

            __syncthreads();

            // Reduce parallelism down to just raking threads
            if (threadIdx.x < RAKING_THREADS)
            {
                // Raking upsweep reduction in grid
                T *raking_ptr = CtaRakingGrid::RakingPtr(smem_storage.raking_grid);
                T raking_partial = ThreadReduce<RAKING_LENGTH>(raking_ptr, scan_op);

                // Exclusive warp synchronous scan
                WarpScan::ExclusiveSum(
                    smem_storage.warp_scan,
                    raking_partial,
                    raking_partial,
                    aggregate,
                    cta_prefix);

                // Exclusive raking downsweep scan
                ThreadScanExclusive<RAKING_LENGTH>(raking_ptr, raking_ptr, scan_op, raking_partial);

                if (!CtaRakingGrid::UNGUARDED)
                {
                    // CTA size isn't a multiple of warp size, so grab aggregate from the appropriate raking cell
                    aggregate = *CtaRakingGrid::PlacementPtr(smem_storage.raking_grid, 0, CTA_THREADS);
                    cta_prefix = scan_op(cta_prefix, aggregate);
                }
            }

            __syncthreads();

            // Grab thread prefix from shared memory
            output = *placement_ptr;
        }
    }


    /**
     * \brief 2D exclusive CTA-wide prefix sum also producing aggregate and consuming/producing cta_prefix.
     *
     * \smemreuse
     *
     * \tparam ITEMS_PER_THREAD     [inferred] The number of consecutive items contributed by each thread.
     */
    template <int ITEMS_PER_THREAD>
    static __device__ __forceinline__ void ExclusiveSum(
        SmemStorage        &smem_storage,               ///< [in] Shared reference to opaque SmemStorage layout
        T                 (&input)[ITEMS_PER_THREAD],   ///< [in] Input
        T                 (&output)[ITEMS_PER_THREAD],  ///< [out] Output (may be aliased to input)
        T                 &aggregate,                   ///< [out] Total aggregate (valid in lane-0).
        T                 &cta_prefix)                  ///< [in-out] Cta-wide prefix to scan (valid in lane-0).
    {
        // Reduce consecutive thread items in registers
        Sum<T> scan_op;
        T thread_partial = ThreadReduce(input, scan_op);

        // Exclusive CTA-scan
        ExclusiveSum(smem_storage, thread_partial, thread_partial, aggregate, cta_prefix);

        // Exclusive scan in registers with prefix
        ThreadScanExclusive(input, output, scan_op, thread_partial);
    }


    /**
     * \brief Exclusive CTA-wide prefix sum.
     *
     * \smemreuse
     */
    static __device__ __forceinline__ void ExclusiveSum(
        SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
        T               input,                          ///< [in] Input
        T               &output)                        ///< [out] Output (may be aliased to input)
    {
        T aggregate;
        ExclusiveSum(smem_storage, input, output, aggregate);
    }


    /**
     * \brief 2D exclusive CTA-wide prefix sum.
     *
     * \smemreuse
     *
     * \tparam ITEMS_PER_THREAD     [inferred] The number of consecutive items contributed by each thread.
     */
    template <int ITEMS_PER_THREAD>
    static __device__ __forceinline__ void ExclusiveSum(
        SmemStorage        &smem_storage,               ///< [in] Shared reference to opaque SmemStorage layout
        T                 (&input)[ITEMS_PER_THREAD],   ///< [in] Input
        T                 (&output)[ITEMS_PER_THREAD])  ///< [out] Output (may be aliased to input)
    {
        // Reduce consecutive thread items in registers
        Sum<T> scan_op;
        T thread_partial = ThreadReduce(input, scan_op);

        // Exclusive CTA-scan
        ExclusiveSum(smem_storage, thread_partial, thread_partial);

        // Exclusive scan in registers with prefix
        ThreadScanExclusive(input, output, scan_op, thread_partial, (threadIdx.x != 0));
    }


    //@}
    /******************************************************************//**
     * @name Inclusive prefix scan variants
     *********************************************************************/
    //@{


    /**
     * \brief Inclusive CTA-wide prefix scan also producing aggregate
     *
     * \smemreuse
     *
     * \tparam ScanOp               [inferred] Binary scan functor type (a model of <a href="http://www.sgi.com/tech/stl/BinaryFunction.html">Binary Function</a>).
     */
    template <typename ScanOp>
    static __device__ __forceinline__ void InclusiveScan(
        SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
        T               input,                          ///< [in] Input
        T               &output,                        ///< [out] Output (may be aliased to input)
        ScanOp          scan_op,                        ///< [in] Binary scan operator
        T               &aggregate)                     ///< [out] Total aggregate (valid in lane-0).
    {
        if (WARP_SYNCHRONOUS)
        {
            // Short-circuit directly to warp scan
            WarpScan::InclusiveScan(
                smem_storage.warp_scan,
                input,
                output,
                scan_op,
                aggregate);
        }
        else
        {
            // Place thread partial into shared memory raking grid
            T *placement_ptr = CtaRakingGrid::PlacementPtr(smem_storage.raking_grid);
            *placement_ptr = input;

            __syncthreads();

            // Reduce parallelism down to just raking threads
            if (threadIdx.x < RAKING_THREADS)
            {
                // Raking upsweep reduction in grid
                T *raking_ptr = CtaRakingGrid::RakingPtr(smem_storage.raking_grid);
                T raking_partial = ThreadReduce<RAKING_LENGTH>(raking_ptr, scan_op);

                // Exclusive warp synchronous scan
                WarpScan::ExclusiveScan(
                    smem_storage.warp_scan,
                    raking_partial,
                    raking_partial,
                    scan_op,
                    aggregate);

                // Exclusive raking downsweep scan
                ThreadScanExclusive<RAKING_LENGTH>(raking_ptr, raking_ptr, scan_op, raking_partial, (threadIdx.x != 0));

                if (!CtaRakingGrid::UNGUARDED)
                {
                    // CTA size isn't a multiple of warp size, so grab aggregate from the appropriate raking cell
                    aggregate = *CtaRakingGrid::PlacementPtr(smem_storage.raking_grid, 0, CTA_THREADS);
                }
            }

            __syncthreads();

            // Grab thread prefix from shared memory
            output = *placement_ptr;
        }
    }


    /**
     * \brief Inclusive CTA-wide prefix scan also producing aggregate
     *
     * \smemreuse
     *
     * \tparam ITEMS_PER_THREAD     [inferred] The number of consecutive items contributed by each thread.
     * \tparam ScanOp               [inferred] Binary scan functor type (a model of <a href="http://www.sgi.com/tech/stl/BinaryFunction.html">Binary Function</a>).
     */
    template <
        int             ITEMS_PER_THREAD,
        typename         ScanOp>
    static __device__ __forceinline__ void InclusiveScan(
        SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
        T               (&input)[ITEMS_PER_THREAD],     ///< [in] Input
        T               (&output)[ITEMS_PER_THREAD],    ///< [out] Output (may be aliased to input)
        ScanOp          scan_op,                        ///< [in] Binary scan operator
        T               &aggregate)                     ///< [out] Total aggregate (valid in lane-0).
    {
        // Reduce consecutive thread items in registers
        T thread_partial = ThreadReduce(input, scan_op);

        // Exclusive CTA-scan
        ExclusiveScan(smem_storage, thread_partial, thread_partial, scan_op, aggregate);

        // Inclusive scan in registers with prefix
        ThreadScanInclusive(input, output, scan_op, thread_partial, (threadIdx.x != 0));
    }


    /**
     * \brief Inclusive CTA-wide prefix scan also producing aggregate, consuming/producing cta_prefix.
     *
     * \smemreuse
     *
     * \tparam ScanOp               [inferred] Binary scan functor type (a model of <a href="http://www.sgi.com/tech/stl/BinaryFunction.html">Binary Function</a>).
     */
    template <typename ScanOp>
    static __device__ __forceinline__ void InclusiveScan(
        SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
        T               input,                          ///< [in] Input
        T               &output,                        ///< [out] Output (may be aliased to input)
        ScanOp          scan_op,                        ///< [in] Binary scan operator
        T               &aggregate,                     ///< [out] Total aggregate (valid in lane-0).
        T               &cta_prefix)                    ///< [in-out] Cta-wide prefix to scan (valid in lane-0).
    {
        if (WARP_SYNCHRONOUS)
        {
            // Short-circuit directly to warp scan
            WarpScan::InclusiveScan(
                smem_storage.warp_scan,
                input,
                output,
                scan_op,
                aggregate,
                cta_prefix);
        }
        else
        {
            // Place thread partial into shared memory raking grid
            T *placement_ptr = CtaRakingGrid::PlacementPtr(smem_storage.raking_grid);
            *placement_ptr = input;

            __syncthreads();

            // Reduce parallelism down to just raking threads
            if (threadIdx.x < RAKING_THREADS)
            {
                // Raking upsweep reduction in grid
                T *raking_ptr = CtaRakingGrid::RakingPtr(smem_storage.raking_grid);
                T raking_partial = ThreadReduce<RAKING_LENGTH>(raking_ptr, scan_op);

                // Warp synchronous scan
                WarpScan::ExclusiveScan(
                    smem_storage.warp_scan,
                    raking_partial,
                    raking_partial,
                    scan_op,
                    aggregate,
                    cta_prefix);

                // Exclusive raking downsweep scan
                ThreadScanExclusive<RAKING_LENGTH>(raking_ptr, raking_ptr, scan_op, raking_partial);

                if (!CtaRakingGrid::UNGUARDED)
                {
                    // CTA size isn't a multiple of warp size, so grab aggregate from the appropriate raking cell
                    aggregate = *CtaRakingGrid::PlacementPtr(smem_storage.raking_grid, 0, CTA_THREADS);
                    cta_prefix = scan_op(cta_prefix, aggregate);
                }
            }

            __syncthreads();

            // Grab thread prefix from shared memory
            output = *placement_ptr;
        }
    }


    /**
     * \brief Inclusive CTA-wide prefix scan also producing aggregate, consuming/producing cta_prefix.
     *
     * \smemreuse
     *
     * \tparam ITEMS_PER_THREAD     [inferred] The number of consecutive items contributed by each thread.
     * \tparam ScanOp               [inferred] Binary scan functor type (a model of <a href="http://www.sgi.com/tech/stl/BinaryFunction.html">Binary Function</a>).
     */
    template <
        int             ITEMS_PER_THREAD,
        typename         ScanOp>
    static __device__ __forceinline__ void InclusiveScan(
        SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
        T               (&input)[ITEMS_PER_THREAD],     ///< [in] Input
        T               (&output)[ITEMS_PER_THREAD],    ///< [out] Output (may be aliased to input)
        ScanOp          scan_op,                        ///< [in] Binary scan operator
        T               &aggregate,                     ///< [out] Total aggregate (valid in lane-0).
        T               &cta_prefix)                    ///< [in-out] Cta-wide prefix to scan (valid in lane-0).
    {
        // Reduce consecutive thread items in registers
        T thread_partial = ThreadReduce(input, scan_op);

        // Exclusive CTA-scan
        ExclusiveScan(smem_storage, thread_partial, thread_partial, scan_op, aggregate, cta_prefix);

        // Inclusive scan in registers with prefix
        ThreadScanInclusive(input, output, scan_op, thread_partial);
    }


    /**
     * \brief Inclusive CTA-wide prefix scan.
     *
     * \smemreuse
     *
     * \tparam ScanOp               [inferred] Binary scan functor type (a model of <a href="http://www.sgi.com/tech/stl/BinaryFunction.html">Binary Function</a>).
     */
    template <typename ScanOp>
    static __device__ __forceinline__ void InclusiveScan(
        SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
        T               input,                          ///< [in] Input
        T               &output,                        ///< [out] Output (may be aliased to input)
        ScanOp          scan_op)                        ///< [in] Binary scan operator
    {
        T aggregate;
        InclusiveScan(smem_storage, input, output, scan_op);
    }


    /**
     * \brief Inclusive CTA-wide prefix scan.
     *
     * \smemreuse
     *
     * \tparam ITEMS_PER_THREAD     [inferred] The number of consecutive items contributed by each thread.
     * \tparam ScanOp               [inferred] Binary scan functor type (a model of <a href="http://www.sgi.com/tech/stl/BinaryFunction.html">Binary Function</a>).
     */
    template <
        int             ITEMS_PER_THREAD,
        typename        ScanOp>
    static __device__ __forceinline__ void InclusiveScan(
        SmemStorage     &smem_storage,                  ///< [in] Shared reference to opaque SmemStorage layout
        T               (&input)[ITEMS_PER_THREAD],     ///< [in] Input
        T               (&output)[ITEMS_PER_THREAD],    ///< [out] Output (may be aliased to input)
        ScanOp          scan_op)                        ///< [in] Binary scan operator
    {
        // Reduce consecutive thread items in registers
        T thread_partial = ThreadReduce(input, scan_op);

        // Exclusive CTA-scan
        ExclusiveScan(smem_storage, thread_partial, thread_partial, scan_op);

        // Inclusive scan in registers with prefix
        ThreadScanInclusive(input, output, scan_op, thread_partial, (threadIdx.x != 0));
    }


};



} // namespace cub
CUB_NS_POSTFIX
