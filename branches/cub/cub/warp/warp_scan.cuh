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
 * The cub::WarpScan type provides variants of parallel prefix scan across threads within a CUDA warp.
 */

#pragma once

#include "../thread/thread_load.cuh"
#include "../thread/thread_store.cuh"
#include "../device_props.cuh"
#include "../type_utils.cuh"
#include "../operators.cuh"
#include "../ns_wrapper.cuh"

CUB_NS_PREFIX

/// CUB namespace
namespace cub {


/**
 * \brief The WarpScan type provides variants of parallel prefix scan across threads within a CUDA warp.
 *
 * \tparam T                        The scan input/output element type
 * \tparam WARPS                    The number of "logical" warps performing concurrent warp scans
 * \tparam LOGICAL_WARP_THREADS     [optional] The number of threads per "logical" warp (may be less than the number of hardware warp threads).  Default is the warp size associated with the CUDA Compute Capability targeted by the compiler (e.g., 32 warps for SM20).
 *
 * <b>Overview</b>
 *
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
 * These parallel prefix scan variants assume the input and
 * output lists to be logically partitioned among threads with warp lane-<em>i</em>
 * having the <em>i</em><sup>th</sup> input and output elements.
 *
 * <b>Features</b>
 * \par
 * - Support for "logical" warps smaller than the physical warp size (e.g., 8 threads).
 * - Support for non-commutative binary associative scan functors.
 * - Support for concurrent scans within multiple warps.
 * - Zero bank conflicts for most types.
 *
 * <b>Algorithm</b>
 * \par
 * These parallel prefix scan variants implement a warp-synchronous
 * Kogge-Stone algorithm having <em>O</em>(log<em>n</em>)
 * steps and <em>O</em>(<em>n</em>log<em>n</em>) work complexity,
 * where <em>n</em> = \p LOGICAL_WARP_THREADS (which defaults to the warp
 * size associated with the CUDA Compute Capability targeted by the compiler).
 *
 * \image html kogge_stone_scan.png "Data flow within a 16-thread Kogge-Stone scan construction.  Junctions represent binary operators."
 *
 * <b>Considerations</b>
 * \par
 * - The operations are most efficient (lowest instruction overhead) when:
 *      - The data type \p T is a built-in primitive or CUDA vector type (e.g.,
 *        \p short, \p int2, \p double, \p float2, etc.)  The implementation may use memory
 *        fences to prevent reference reordering of non-primitive types.
 *      - Performing exclusive scans. The implementation may use guarded shared memory
 *        accesses for inclusive scans (other than prefix sum) because no identity
 *        element is provided.
 *
 * <b>Examples</b>
 * \par
 * - <b>Example 1:</b> Simple exclusive prefix sum for one warp
 *      \code
 *      #include <cub.cuh>
 *
 *      __global__ void SomeKernel(...)
 *      {
 *          typedef cub::WarpScan<int, 1> WarpScan;                     // A parameterized WarpScan type for use with 1 warp on type int.
 *
 *          __shared__ typename WarpScan::SmemStorage smem_storage;     // Opaque shared memory for WarpScan
 *
 *          // Perform prefix sum of threadIds in first warp
 *          if (threadIdx.x < 32)
 *          {
 *              int output, input = threadIdx.x;
 *              WarpScan::ExclusiveSum(smem_storage, input, output);
 *
 *              printf("tid(%d) output(%d)\n\n", threadIdx.x, output);
 *          }
 *      \endcode
 *
 *      Printed output:
 *      \code
 *      tid(0) output(0)
 *      tid(1) output(0)
 *      tid(2) output(1)
 *      tid(3) output(3)
 *      tid(4) output(6)
 *      ...
 *      tid(31) output(465)
 *      \endcode
 * <br>
 *
 * \par
 * - <b>Example 2:</b> More sophisticated exclusive prefix sum for one warp
 *      \code
 *      #include <cub.cuh>
 *
 *      __global__ void SomeKernel(...)
 *      {
 *          typedef cub::WarpScan<int, 1> WarpScan;                     // A parameterized WarpScan type for use with 1 warp on type int.
 *
 *          __shared__ typename WarpScan::SmemStorage smem_storage;     // Opaque shared memory for WarpScan
 *
 *          // Perform prefix sum of 2s in first warp, all seeded with a prefix value of 10
 *          if (threadIdx.x < 32)
 *          {
 *              int output, aggregate, input = 2, warp_prefix = 10;
 *              WarpScan::ExclusiveSum(smem_storage, input, output, warp_prefix, aggregate);
 *
 *              printf("tid(%d) output(%d)\n\n", threadIdx.x, output);
 *              if (threadIdx.x == 0)
 *                  printf("computed aggregate(%d), updated warp_prefix(%d)\n", aggregate, warp_prefix);
 *          }
 *          \endcode
 *
 *      Printed output:
 *      \code
 *      tid(0) output(10)
 *      tid(1) output(12)
 *      tid(2) output(14)
 *      tid(3) output(16)
 *      tid(4) output(18)
 *      ...
 *      tid(31) output(72)
 *
 *      computed aggregate(74), udpated warp_prefix(84)
 *      \endcode
 */
template <
    typename    T,
    int         WARPS,
    int         LOGICAL_WARP_THREADS = DeviceProps::WARP_THREADS>
class WarpScan
{
    //---------------------------------------------------------------------
    // Constants and typedefs
    //---------------------------------------------------------------------

private:

    enum
    {
        /// The number of warp scan steps
        STEPS = Log2<LOGICAL_WARP_THREADS>::VALUE,

        /// The number of threads in half a warp
        HALF_WARP_THREADS = 1 << (STEPS - 1),

        /// The number of shared memory elements per warp
        WARP_SMEM_ELEMENTS =  LOGICAL_WARP_THREADS + HALF_WARP_THREADS,
    };

    /// Shared memory storage layout type
    struct SmemStorage
    {
        T warp_scan[WARPS][WARP_SMEM_ELEMENTS];
    };

public:

    /// Opaque shared memory storage type required by the WarpScan template instantiation.
    typedef SmemStorage SmemStorage;


    //---------------------------------------------------------------------
    // Template iteration structures.  (Regular iteration cannot always be
    // unrolled due to conditionals or ABI procedure calls within
    // functors).
    //---------------------------------------------------------------------

private:

    /// General template iteration
    template <int COUNT, int MAX, bool HAS_IDENTITY, bool SHARE_FINAL>
    struct Iterate
    {
        /// Inclusive scan step
        template <typename ScanOp>
        static __device__ __forceinline__ T InclusiveScan(
            SmemStorage     &smem_storage,      ///< Shared reference to opaque SmemStorage layout
            unsigned int    warp_id,            ///< Warp id
            unsigned int    lane_id,            ///< Lane id
            T               partial,            ///< Calling thread's input partial reduction
            ScanOp          scan_op)            ///< Binary associative scan functor
        {
            const int OFFSET = 1 << COUNT;

            // Share partial into buffer
            ThreadStore<PTX_STORE_VS>(&smem_storage.warp_scan[warp_id][HALF_WARP_THREADS + lane_id], partial);

            // Update partial if addend is in range
            if (HAS_IDENTITY || (lane_id >= OFFSET))
            {
                T addend = ThreadLoad<PTX_LOAD_VS>(&smem_storage.warp_scan[warp_id][HALF_WARP_THREADS + lane_id - OFFSET]);

                partial = scan_op(partial, addend);
            }

            return Iterate<COUNT + 1, MAX, HAS_IDENTITY, SHARE_FINAL>::InclusiveScan(
                smem_storage,
                warp_id,
                lane_id,
                partial,
                scan_op);
        }
    };


    /// Termination
    template <int MAX, bool HAS_IDENTITY, bool SHARE_FINAL>
    struct Iterate<MAX, MAX, HAS_IDENTITY, SHARE_FINAL>
    {
        /// Inclusive scan step
        template <typename ScanOp>
        static __device__ __forceinline__ T InclusiveScan(
            SmemStorage     &smem_storage,      ///< Shared reference to opaque SmemStorage layout
            unsigned int    warp_id,            ///< Warp id
            unsigned int    lane_id,            ///< Lane id
            T               partial,            ///< Calling thread's input partial reduction
            ScanOp          scan_op)            ///< Binary associative scan functor
        {
            if (SHARE_FINAL)
            {
                // Share partial into buffer
                ThreadStore<PTX_STORE_VS>(&smem_storage.warp_scan[warp_id][HALF_WARP_THREADS + lane_id], partial);
            }

            return partial;
        }
    };


public:

    //---------------------------------------------------------------------
    // Inclusive prefix sum interface
    //---------------------------------------------------------------------

    /**
     * \brief Computes an inclusive prefix sum in each logical warp.
     */
    static __device__ __forceinline__ void InclusiveSum(
        SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
        T               input,              ///< [in] Calling thread's input
        T               &output)            ///< [out] Calling thread's output.  May be aliased with \p input.
    {
        // Warp, lane-IDs
        unsigned int warp_id = (WARPS == 1) ? 0 : (threadIdx.x / LOGICAL_WARP_THREADS);
        unsigned int lane_id = (WARPS == 1) ? threadIdx.x : (threadIdx.x & (LOGICAL_WARP_THREADS - 1));

        // Initialize identity region
        ThreadStore<PTX_STORE_VS>(&smem_storage.warp_scan[warp_id][lane_id], T(0));

        // Compute inclusive warp scan (has identity, don't share final)
        output = Iterate<0, STEPS, true, false>::InclusiveScan(
            smem_storage,
            warp_id,
            lane_id,
            input,
            Sum<T>());
    }


    /**
     * \brief Computes an inclusive prefix sum in each logical warp.  Also computes the warp-wide \p aggregate of all thread inputs (only valid in each logical warp lane<sub>0</sub>).
     */
    static __device__ __forceinline__ void InclusiveSum(
        SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
        T               input,              ///< [in] Calling thread's input
        T               &output,            ///< [out] Calling thread's output.  May be aliased with \p input.
        T               &aggregate)         ///< [out] Total aggregate (valid in each logical warp lane<sub>0</sub>)
    {
        // Warp, lane-IDs
        unsigned int warp_id = (WARPS == 1) ? 0 : (threadIdx.x / LOGICAL_WARP_THREADS);
        unsigned int lane_id = (WARPS == 1) ? threadIdx.x : (threadIdx.x & (LOGICAL_WARP_THREADS - 1));

        // Initialize identity region
        ThreadStore<PTX_STORE_VS>(&smem_storage.warp_scan[warp_id][lane_id], T(0));

        // Compute inclusive warp scan (has identity, share final)
        output = Iterate<0, STEPS, true, true>::InclusiveScan(
            smem_storage,
            warp_id,
            lane_id,
            input,
            Sum<T>());

        // Retrieve aggregate in each logical warp lane<sub>0</sub>
        aggregate = smem_storage.warp_scan[warp_id][WARP_SMEM_ELEMENTS - 1];
    }


    /**
     * \brief Computes an inclusive prefix sum in each logical warp.  The \p warp_prefix value from each logical warp lane<sub>0</sub> is applied to all scan outputs.  Also computes the warp-wide \p aggregate of all thread inputs (only valid in each logical warp lane<sub>0</sub>).  The \p warp_prefix is further updated by the value of \p aggregate.
     */
    static __device__ __forceinline__ void InclusiveSum(
        SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
        T               input,              ///< [in] Calling thread's input
        T               &output,            ///< [out] Calling thread's output.  May be aliased with \p input.
        T               &aggregate,         ///< [out] Total aggregate (valid in each logical warp lane<sub>0</sub>)
        T               &warp_prefix)       ///< [in-out] Warp-wide prefix to seed with (valid in each logical warp lane<sub>0</sub>).
    {
        // Lane-IDs
        unsigned int lane_id = (WARPS == 1) ? threadIdx.x : (threadIdx.x & (LOGICAL_WARP_THREADS - 1));

        // Incorporate warp-prefix from each logical warp lane<sub>0</sub>
        if (lane_id == 0)
        {
            input = warp_prefix + input;
        }

        // Compute inclusive warp scan
        InclusiveSum(smem_storage, input, output, aggregate);

        // Update warp_prefix
        warp_prefix += aggregate;
    }


    //---------------------------------------------------------------------
    // Exclusive prefix sum interface
    //---------------------------------------------------------------------

    /**
     * \brief Computes an exclusive prefix sum in each logical warp.
     */
    static __device__ __forceinline__ void ExclusiveSum(
        SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
        T               input,              ///< [in] Calling thread's input
        T               &output)            ///< [out] Calling thread's output.  May be aliased with \p input.
    {
        // Compute exclusive warp scan from inclusive warp scan
        T inclusive;
        InclusiveSum(smem_storage, input, inclusive);
        output = inclusive - input;
    }


    /**
     * \brief Computes an exclusive prefix sum in each logical warp.  Also computes the warp-wide \p aggregate of all thread inputs (only valid in each logical warp lane<sub>0</sub>).
     */
    static __device__ __forceinline__ void ExclusiveSum(
        SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
        T               input,              ///< [in] Calling thread's input
        T               &output,            ///< [out] Calling thread's output.  May be aliased with \p input.
        T               &aggregate)         ///< [out] Total aggregate (valid in each logical warp lane<sub>0</sub>)
    {
        // Compute exclusive warp scan from inclusive warp scan
        T inclusive;
        InclusiveSum(smem_storage, input, inclusive, aggregate);
        output = inclusive - input;
    }


    /**
     * \brief Computes an exclusive prefix sum in each logical warp.  The \p warp_prefix value from each logical warp lane<sub>0</sub> is applied to all scan outputs.  Also computes the warp-wide \p aggregate of all thread inputs (only valid in each logical warp lane<sub>0</sub>).  The \p warp_prefix is further updated by the value of \p aggregate.
     */
    static __device__ __forceinline__ void ExclusiveSum(
        SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
        T               input,              ///< [in] Calling thread's input
        T               &output,            ///< [out] Calling thread's output.  May be aliased with \p input.
        T               &aggregate,         ///< [out] Total aggregate (valid in each logical warp lane<sub>0</sub>).
        T               &warp_prefix)       ///< [in-out] Warp-wide prefix to seed with (valid in each logical warp lane<sub>0</sub>).
    {
        // Warp, lane-IDs
        unsigned int lane_id = (WARPS == 1) ? threadIdx.x : (threadIdx.x & (LOGICAL_WARP_THREADS - 1));

        // Incorporate warp-prefix from each logical warp lane<sub>0</sub>
        T partial = input;
        if (lane_id == 0)
        {
            partial = warp_prefix + input;
        }

        // Compute exclusive warp scan from inclusive warp scan
        T inclusive;
        InclusiveSum(smem_storage, partial, inclusive, aggregate);
        output = inclusive - input;

        // Update warp_prefix
        warp_prefix += aggregate;
    }


    //---------------------------------------------------------------------
    // Inclusive prefix scan interface
    //---------------------------------------------------------------------

    /**
     * \brief Computes an inclusive prefix sum using the specified binary scan functor in each logical warp.
     *
     * \tparam ScanOp     [inferred] Binary scan functor type (a model of <a href="http://www.sgi.com/tech/stl/BinaryFunction.html">Binary Function</a>).
     */
    template <typename ScanOp>
    static __device__ __forceinline__ void InclusiveScan(
        SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
        T               input,              ///< [in] Calling thread's input
        T               &output,            ///< [out] Calling thread's output.  May be aliased with \p input.
        ScanOp          scan_op)            ///< [in] Binary associative scan functor.
    {
        // Warp, lane-IDs
        unsigned int warp_id = (WARPS == 1) ? 0 : (threadIdx.x / LOGICAL_WARP_THREADS);
        unsigned int lane_id = (WARPS == 1) ? threadIdx.x : (threadIdx.x & (LOGICAL_WARP_THREADS - 1));

        // Compute inclusive warp scan (no identity, don't share final)
        output = Iterate<0, STEPS, false, false>::InclusiveScan(
            smem_storage,
            warp_id,
            lane_id,
            input,
            scan_op);
    }


    /**
     * \brief Computes an inclusive prefix sum using the specified binary scan functor in each logical warp.  Also computes the warp-wide \p aggregate of all thread inputs (only valid in each logical warp lane<sub>0</sub>).
     *
     * \tparam ScanOp     [inferred] Binary scan functor type (a model of <a href="http://www.sgi.com/tech/stl/BinaryFunction.html">Binary Function</a>).
     */
    template <typename ScanOp>
    static __device__ __forceinline__ void InclusiveScan(
        SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
        T               input,              ///< [in] Calling thread's input
        T               &output,            ///< [out] Calling thread's output.  May be aliased with \p input.
        ScanOp          scan_op,            ///< [in] Binary associative scan functor.
        T               &aggregate)         ///< [out] Total aggregate (valid in each logical warp lane<sub>0</sub>)
    {
        // Warp, lane-IDs
        unsigned int warp_id = (WARPS == 1) ? 0 : (threadIdx.x / LOGICAL_WARP_THREADS);
        unsigned int lane_id = (WARPS == 1) ? threadIdx.x : (threadIdx.x & (LOGICAL_WARP_THREADS - 1));

        // Compute inclusive warp scan (no identity, share final)
        output = Iterate<0, STEPS, false, true>::InclusiveScan(
            smem_storage,
            warp_id,
            lane_id,
            input,
            scan_op);

        // Retrieve aggregate
        aggregate = smem_storage.warp_scan[warp_id][WARP_SMEM_ELEMENTS - 1];
    }


    /**
     * \brief Computes an inclusive prefix sum using the specified binary scan functor in each logical warp.  The \p warp_prefix value from each logical warp lane<sub>0</sub> is applied to all scan outputs.  Also computes the warp-wide \p aggregate of all thread inputs (only valid in each logical warp lane<sub>0</sub>).  The \p warp_prefix is further updated by the value of \p aggregate.
     *
     * \tparam ScanOp     [inferred] Binary scan functor type (a model of <a href="http://www.sgi.com/tech/stl/BinaryFunction.html">Binary Function</a>).
     */
    template <typename ScanOp>
    static __device__ __forceinline__ void InclusiveScan(
        SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
        T               input,              ///< [in] Calling thread's input
        T               &output,            ///< [out] Calling thread's output.  May be aliased with \p input.
        ScanOp          scan_op,            ///< [in] Binary associative scan functor.
        T               &aggregate,         ///< [out] Total aggregate (valid in each logical warp lane<sub>0</sub>).
        T               &warp_prefix)       ///< [in-out] Warp-wide prefix to seed with (valid in each logical warp lane<sub>0</sub>).
    {
        // Warp, lane-IDs
        unsigned int lane_id = (WARPS == 1) ? threadIdx.x : (threadIdx.x & (LOGICAL_WARP_THREADS - 1));

        // Incorporate warp-prefix from each logical warp lane<sub>0</sub>
        if (lane_id == 0)
        {
            input = scan_op(warp_prefix, input);
        }

        // Compute inclusive warp scan
        InclusiveScan(smem_storage, input, output, scan_op, aggregate);

        // Update warp_prefix
        warp_prefix = scan_op(warp_prefix, aggregate);
    }


    //---------------------------------------------------------------------
    // Exclusive prefix scan interface
    //---------------------------------------------------------------------

    /**
     * \brief Computes an exclusive prefix scan using the specified binary scan functor in each logical warp.
     *
     * \tparam ScanOp     [inferred] Binary scan functor type (a model of <a href="http://www.sgi.com/tech/stl/BinaryFunction.html">Binary Function</a>).
     */
    template <typename ScanOp>
    static __device__ __forceinline__ void ExclusiveScan(
        SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
        T               input,              ///< [in] Calling thread's input
        T               &output,            ///< [out] Calling thread's output.  May be aliased with \p input.
        T               identity,           ///< [in] Identity value.
        ScanOp          scan_op)            ///< [in] Binary associative scan functor.
    {
        // Warp, lane-IDs
        unsigned int warp_id = (WARPS == 1) ? 0 : (threadIdx.x / LOGICAL_WARP_THREADS);
        unsigned int lane_id = (WARPS == 1) ? threadIdx.x : (threadIdx.x & (LOGICAL_WARP_THREADS - 1));

        // Initialize identity region
        ThreadStore<PTX_STORE_VS>(&smem_storage.warp_scan[warp_id][lane_id], identity);

        // Compute inclusive warp scan (identity, share final)
        T inclusive = Iterate<0, STEPS, true, true>::InclusiveScan(
            smem_storage,
            warp_id,
            lane_id,
            input,
            scan_op);

        // Retrieve exclusive scan
        output = ThreadLoad<PTX_LOAD_VS>(&smem_storage.warp_scan[warp_id][HALF_WARP_THREADS + lane_id - 1]);
    }


    /**
     * \brief Computes an exclusive prefix scan using the specified binary scan functor in each logical warp.  Because no identity value is supplied, the \p output computed for each logical warp lane<sub>0</sub> is invalid.
     *
     * \tparam ScanOp     [inferred] Binary scan functor type (a model of <a href="http://www.sgi.com/tech/stl/BinaryFunction.html">Binary Function</a>).
     */
    template <typename ScanOp>
    static __device__ __forceinline__ void ExclusiveScan(
        SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
        T               input,              ///< [in] Calling thread's input
        T               &output,            ///< [out] Calling thread's output.  May be aliased with \p input.
        ScanOp          scan_op)            ///< [in] Binary associative scan functor.
    {
        // Warp, lane-IDs
        unsigned int warp_id = (WARPS == 1) ? 0 : (threadIdx.x / LOGICAL_WARP_THREADS);
        unsigned int lane_id = (WARPS == 1) ? threadIdx.x : (threadIdx.x & (LOGICAL_WARP_THREADS - 1));

        // Compute inclusive warp scan (identity, share final)
        T inclusive = Iterate<0, STEPS, false, true>::InclusiveScan(
            smem_storage,
            warp_id,
            lane_id,
            input,
            scan_op);

        // Retrieve exclusive scan
        output = ThreadLoad<PTX_LOAD_VS>(&smem_storage.warp_scan[warp_id][HALF_WARP_THREADS + lane_id - 1]);
    }


    /**
     * \brief Computes an exclusive prefix scan using the specified binary scan functor in each logical warp.  Also computes the warp-wide \p aggregate of all thread inputs (only valid in each logical warp lane<sub>0</sub>).
     *
     * \tparam ScanOp     [inferred] Binary scan functor type (a model of <a href="http://www.sgi.com/tech/stl/BinaryFunction.html">Binary Function</a>).
     */
    template <typename ScanOp>
    static __device__ __forceinline__ void ExclusiveScan(
        SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
        T               input,              ///< [in] Calling thread's input
        T               &output,            ///< [out] Calling thread's output.  May be aliased with \p input.
        T               identity,           ///< [in] Identity value.
        ScanOp          scan_op,            ///< [in] Binary associative scan functor.
        T               &aggregate)         ///< [out] Total aggregate (valid in each logical warp lane<sub>0</sub>)
    {
        // Warp id
        unsigned int warp_id = (WARPS == 1) ? 0 : (threadIdx.x / LOGICAL_WARP_THREADS);

        // Exclusive warp scan
        ExclusiveScan(smem_storage, input, output, identity, scan_op);

        // Retrieve aggregate
        aggregate = smem_storage.warp_scan[warp_id][WARP_SMEM_ELEMENTS - 1];
    }


    /**
     * \brief Computes an exclusive prefix scan using the specified binary scan functor in each logical warp.  Because no identity value is supplied, the \p output computed for each logical warp lane<sub>0</sub> is invalid.  Also computes the warp-wide \p aggregate of all thread inputs (only valid in each logical warp lane<sub>0</sub>).
     *
     * \tparam ScanOp     [inferred] Binary scan functor type (a model of <a href="http://www.sgi.com/tech/stl/BinaryFunction.html">Binary Function</a>).
     */
    template <typename ScanOp>
    static __device__ __forceinline__ void ExclusiveScan(
        SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
        T               input,              ///< [in] Calling thread's input
        T               &output,            ///< [out] Calling thread's output.  May be aliased with \p input.
        ScanOp          scan_op,            ///< [in] Binary associative scan functor.
        T               &aggregate)         ///< [out] Total aggregate (valid in each logical warp lane<sub>0</sub>)
    {
        // Warp id
        unsigned int warp_id = (WARPS == 1) ? 0 : (threadIdx.x / LOGICAL_WARP_THREADS);

        // Exclusive warp scan
        ExclusiveScan(smem_storage, input, output, scan_op);

        // Retrieve aggregate
        aggregate = smem_storage.warp_scan[warp_id][WARP_SMEM_ELEMENTS - 1];
    }


    /**
     * \brief Computes an exclusive prefix scan using the specified binary scan functor in each logical warp.  The \p warp_prefix value from each logical warp lane<sub>0</sub> is applied to all scan outputs.  Also computes the warp-wide \p aggregate of all thread inputs (only valid in each logical warp lane<sub>0</sub>).  The \p warp_prefix is further updated by the value of \p aggregate.
     *
     * \tparam ScanOp     [inferred] Binary scan functor type (a model of <a href="http://www.sgi.com/tech/stl/BinaryFunction.html">Binary Function</a>).
     */
    template <typename ScanOp>
    static __device__ __forceinline__ void ExclusiveScan(
        SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
        T               input,              ///< [in] Calling thread's input
        T               &output,            ///< [out] Calling thread's output.  May be aliased with \p input.
        T               identity,           ///< [in] Identity value.
        ScanOp          scan_op,            ///< [in] Binary associative scan functor.
        T               &aggregate,         ///< [out] Total aggregate (valid in each logical warp lane<sub>0</sub>).
        T               &warp_prefix)       ///< [in-out] Warp-wide prefix to seed with (valid in each logical warp lane<sub>0</sub>).
    {
        // Warp, lane-IDs
        unsigned int lane_id = (WARPS == 1) ? threadIdx.x : (threadIdx.x & (LOGICAL_WARP_THREADS - 1));

        // Incorporate warp-prefix from each logical warp lane<sub>0</sub>
        if (lane_id == 0)
        {
            input = scan_op(warp_prefix, input);
        }

        // Exclusive warp scan
        ExclusiveScan(smem_storage, input, output, identity, scan_op, aggregate);

        // lane<sub>0</sub> gets warp_prefix (instead of identity)
        if (lane_id == 0)
        {
            output = warp_prefix;
            warp_prefix = scan_op(warp_prefix, aggregate);
        }
    }


    /**
     * \brief Computes an exclusive prefix scan using the specified binary scan functor in each logical warp.  Because no identity value is supplied, the \p output computed for each logical warp lane<sub>0</sub> is invalid.  The \p warp_prefix value from each logical warp lane<sub>0</sub> is applied to all scan outputs.  Also computes the warp-wide \p aggregate of all thread inputs (only valid in each logical warp lane<sub>0</sub>).  The \p warp_prefix is further updated by the value of \p aggregate.
     *
     * \tparam ScanOp     [inferred] Binary scan functor type (a model of <a href="http://www.sgi.com/tech/stl/BinaryFunction.html">Binary Function</a>).
     */
    template <typename ScanOp>
    static __device__ __forceinline__ void ExclusiveScan(
        SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
        T               input,              ///< [in] Calling thread's input
        T               &output,            ///< [out] Calling thread's output.  May be aliased with \p input.
        ScanOp          scan_op,            ///< [in] Binary associative scan functor.
        T               &aggregate,         ///< [out] Total aggregate (valid in each logical warp lane<sub>0</sub>).
        T               &warp_prefix)       ///< [in-out] Warp-wide prefix to seed with (valid in each logical warp lane<sub>0</sub>).
    {
        // Warp, lane-IDs
        unsigned int lane_id = (WARPS == 1) ? threadIdx.x : (threadIdx.x & (LOGICAL_WARP_THREADS - 1));

        // Incorporate warp-prefix from each logical warp lane<sub>0</sub>
        if (lane_id == 0)
        {
            input = scan_op(warp_prefix, input);
        }

        // Exclusive warp scan
        ExclusiveScan(smem_storage, input, output, scan_op, aggregate);

        // lane<sub>0</sub> gets warp_prefix (instead of identity)
        if (lane_id == 0)
        {
            output = warp_prefix;
            warp_prefix = scan_op(warp_prefix, aggregate);
        }
    }
};


} // namespace cub
CUB_NS_POSTFIX
