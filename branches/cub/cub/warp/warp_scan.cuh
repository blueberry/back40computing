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
 * \addtogroup SimtCoop
 * @{
 */

/**
 * \brief The WarpScan type provides variants of parallel prefix scan across threads within a CUDA warp.  ![](warp_scan_logo.png)
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
 * These parallel prefix scan variants assume the input and output lists to
 * be logically partitioned among threads with warp thread-lane-<em>i</em>
 * having the <em>i</em><sup>th</sup> input and output elements.  To minimize
 * synchronization overhead for operations involving the cumulative
 * \p aggregate and \p warp_prefix, these values are only valid in
 * thread-lane<sub>0</sub>.
 *
 * \tparam T                        The scan input/output element type
 * \tparam WARPS                    The number of "logical" warps performing concurrent warp scans
 * \tparam LOGICAL_WARP_THREADS     <b>[optional]</b> The number of threads per "logical" warp (may be less than the number of hardware warp threads).  Default is the warp size associated with the CUDA Compute Capability targeted by the compiler (e.g., 32 warps for SM20).
 *
 * <b>Important Features and Considerations</b>
 * \par
 * - Support for "logical" warps smaller than the physical warp size (e.g., 8 threads).
 * - Support for non-commutative binary associative scan functors.
 * - Support for concurrent scans within multiple warps.
 * - Zero bank conflicts for most types.
 * - After any operation, a subsequent CTA barrier (<tt>__syncthreads</tt>) is
 *   required if the supplied WarpScan::SmemStorage is to be reused/repurposed by the CTA.
 * - The operations are most efficient (lowest instruction overhead) when:
 *      - The data type \p T is a built-in primitive or CUDA vector type (e.g.,
 *        \p short, \p int2, \p double, \p float2, etc.)  Otherwise the implementation
 *        may use memory fences to prevent reference reordering of
 *        non-primitive types.
 *      - Performing exclusive scans. The implementation may use guarded
 *        shared memory accesses for inclusive scans (other than prefix sum)
 *        because no identity element is provided.
 *
 * <b>Algorithm</b>
 * \par
 * These parallel prefix scan variants implement a warp-synchronous
 * Kogge-Stone algorithm having <em>O</em>(log<em>n</em>)
 * steps and <em>O</em>(<em>n</em>log<em>n</em>) work complexity,
 * where <em>n</em> = \p LOGICAL_WARP_THREADS (which defaults to the warp
 * size associated with the CUDA Compute Capability targeted by the compiler).
 *
 * \image html kogge_stone_scan.png
 * <center><b>"Data flow within a 16-thread Kogge-Stone scan construction.  Junctions represent binary operators."</b></center>
 *
 * <b>Examples</b>
 *
 * \par
 * - <b>Example 1:</b> Simple exclusive prefix sum for one warp
 *      \code
 *      #include <cub.cuh>
 *
 *      __global__ void SomeKernel(...)
 *      {
 *          // A parameterized WarpScan type for use with one warp on type int.
 *          typedef cub::WarpScan<int, 1> WarpScan;
 *
 *          // Opaque shared memory for WarpScan
 *          __shared__ typename WarpScan::SmemStorage smem_storage;
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
 *
 * \par
 * - <b>Example 2:</b> More sophisticated exclusive prefix sum for one warp
 *      \code
 *      #include <cub.cuh>
 *
 *      __global__ void SomeKernel(...)
 *      {
 *          // A parameterized WarpScan type for use with one warp on type int.
 *          typedef cub::WarpScan<int, 1> WarpScan;
 *
 *          // Opaque shared memory for WarpScan
 *          __shared__ typename WarpScan::SmemStorage smem_storage;
 *
 *          // Perform prefix sum of 2s, all seeded with a warp prefix value of 10
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

    /// WarpScan algorithmic variants
    enum WarpScanPolicy
    {
        SHFL_SCAN,          // Warp-synchronous SHFL-based scan
        SMEM_SCAN,          // Warp-synchronous smem-based scan
    };

    /// Constants
    enum
    {
        /// SHFL-scan if SM3 on 4-byte or smaller primitives
        POLICY = ((PTX_ARCH >= 300) && Traits<T>::PRIMITIVE && (sizeof(T) <= 4)) ?
            SHFL_SCAN :
            SMEM_SCAN,

        /// The number of warp scan steps
        STEPS = Log2<LOGICAL_WARP_THREADS>::VALUE,
    };


    /**
     * Specialized WarpScan implementations
     */
    template <int POLICY>
    struct WarpScanInternal;


    /**
     * Warpscan specialized for SHFL_SCAN variant
     */
    struct WarpScanInternal<SHFL_SCAN>
    {
        /// Constants
        enum
        {
            SHFL_MASK = LOGICAL_WARP_THREADS << 5
        };


        /// Shared memory storage layout type
        typedef NullType SmemStorage;


        /// Specialized inclusive sum for unsigned int
        static __device__ __forceinline__ unsigned int InclusiveSum(unsigned int partial)
        {

            // Iterate scan steps
            #pragma unroll
            for (int STEP = 0; STEP < STEPS; STEP++)
            {
                asm(
                    "{.reg .u32 r0;"
                    ".reg .pred p;"
                    "shfl.up.b32 r0|p, %1, %2, %3;"
                    "@p add.u32 r0, r0, %4;"
                    "mov.u32 %0, r0;}"
                    : "=r"(partial) : "r"(partial), "r"(1 << STEP), "r"(SHFL_MASK), "r"(partial));
            }

            return partial;
        }


        /// Specialized inclusive sum for float
        static __device__ __forceinline__ float InclusiveSum(float partial)
        {
            // Iterate scan steps
            #pragma unroll
            for (int STEP = 0; STEP < STEPS; STEP++)
            {
                asm(
                    "{.reg .u32 r0;"
                    ".reg .pred p;"
                    "shfl.up.b32 r0|p, %1, %2, %3;"
                    "@p add.f32 r0, r0, %4;"
                    "mov.f32 %0, r0;}"
                    : "=f"(partial) : "f"(partial), "r"(1 << STEP), "r"(SHFL_MASK), "f"(partial));
            }

            return partial;
        }


        /// Inclusive sum for other integer primitives
        static __device__ __forceinline__ T InclusiveSum(T partial)
        {
            // Cast as unsigned int
            unsigned int upartial = reinterpret_cast<unsigned int&>(partial);
            return reinterpret_cast<T&>(InclusiveSum(upartial));
        }


        /// Inclusive scan
        template <typename ScanOp>
        static __device__ __forceinline__ T InclusiveScan(
            T               partial,
            ScanOp          scan_op)
        {
            T               temp;
            unsigned int    *upartial = reinterpret_cast<unsigned int*>(&partial);
            unsigned int    *utemp = reinterpret_cast<unsigned int*>(&temp);

            // Iterate scan steps
            #pragma unroll
            for (int STEP = 0; STEP < STEPS; STEP++)
            {
                asm(
                    "{.reg .pred p;"
                    "shfl.up.b32 %0|p, %1, %2, %3;"
                    : "=r"(*utemp) : "r"(*upartial), "r"(1 << STEP), "r"(SHFL_MASK));

                temp = scan_op(temp, partial);

                asm(
                    "selp.b32 %0, %1, %2, p;}"
                    : "=r"(*upartial) : "r"(*utemp), "r"(*upartial));
            }

            return partial;
        }


        /// Exclusive scan
        template <typename ScanOp>
        static __device__ __forceinline__ T ExclusiveScan(
            T               partial,
            T               identity,
            ScanOp          scan_op)
        {
            T               exclusive;
            T               inclusive = InclusiveScan(partial, scan_op);
            unsigned int    *uinclusive = reinterpret_cast<unsigned int*>(&inclusive);
            unsigned int    *uidentity = reinterpret_cast<unsigned int*>(&identity);
            unsigned int    *uexclusive = reinterpret_cast<unsigned int*>(&exclusive);

            asm(
                "{.reg .u32 r0;"
                ".reg .pred p;"
                "shfl.up.b32 r0|p, %1, 1, %2;"
                "selp.b32 %0, r0, %3, p;}"
                : "=r"(*uexclusive) : "r"(*uinclusive), "r"(SHFL_MASK), uidentity);

            return exclusive;
        }


        /// Exclusive scan without identity
        template <typename ScanOp>
        static __device__ __forceinline__ T ExclusiveScan(
            T               partial,
            ScanOp          scan_op)
        {
            T               exclusive;
            T               inclusive = InclusiveScan(partial, scan_op);
            unsigned int    *uinclusive = reinterpret_cast<unsigned int*>(&inclusive);
            unsigned int    *uexclusive = reinterpret_cast<unsigned int*>(&exclusive);

            asm(
                "{.reg .pred p;"
                "shfl.up.b32 %0, %1, 1, %2;"
                : "=r"(*uexclusive) : "r"(*uinclusive), "r"(SHFL_MASK));

            return exclusive;
        }
    };


    /**
     * Warpscan specialized for SMEM_SCAN
     */
    struct WarpScanInternal<SMEM_SCAN>
    {
        /// Constants
        enum
        {
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


        /// Basic inclusive scan
        template <
            bool HAS_IDENTITY,
            bool SHARE_FINAL,
            typename ScanOp>
        static __device__ __forceinline__ T BasicScan(
            SmemStorage     &smem_storage,      ///< Shared reference to opaque SmemStorage layout
            unsigned int    warp_id,            ///< Warp id
            unsigned int    lane_id,            ///< thread-lane id
            T               partial,            ///< Calling thread's input partial reduction
            ScanOp          scan_op)            ///< Binary associative scan functor
        {
            // Iterate scan steps
            #pragma unroll
            for (int STEP = 0; STEP < STEPS; STEP++)
            {
                const int OFFSET = 1 << STEP;

                // Share partial into buffer
                ThreadStore<PTX_STORE_VS>(&smem_storage.warp_scan[warp_id][HALF_WARP_THREADS + lane_id], partial);

                // Update partial if addend is in range
                if (HAS_IDENTITY || (lane_id >= OFFSET))
                {
                    T addend = ThreadLoad<PTX_LOAD_VS>(&smem_storage.warp_scan[warp_id][HALF_WARP_THREADS + lane_id - OFFSET]);
                    partial = scan_op(partial, addend);
                }
            }

            if (SHARE_FINAL)
            {
                // Share partial into buffer
                ThreadStore<PTX_STORE_VS>(&smem_storage.warp_scan[warp_id][HALF_WARP_THREADS + lane_id], partial);
            }

            return partial;
        }


        /// Inclusive prefix sum
        static __device__ __forceinline__ void InclusiveSum(
            SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
            T               input,              ///< [in] Calling thread's input
            T               &output)            ///< [out] Calling thread's output.  May be aliased with \p input.
        {
            // Warp, thread-lane-IDs
            unsigned int warp_id = (WARPS == 1) ? 0 : (threadIdx.x / LOGICAL_WARP_THREADS);
            unsigned int lane_id = (WARPS == 1) ? threadIdx.x : (threadIdx.x & (LOGICAL_WARP_THREADS - 1));

            // Initialize identity region
            ThreadStore<PTX_STORE_VS>(&smem_storage.warp_scan[warp_id][lane_id], T(0));

            // Compute inclusive warp scan (has identity, don't share final)
            output = BasicScan<true, false>(
                smem_storage,
                warp_id,
                lane_id,
                input,
                Sum<T>());
        }


        /// Inclusive prefix sum with aggregate
        static __device__ __forceinline__ void InclusiveSum(
            SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
            T               input,              ///< [in] Calling thread's input
            T               &output,            ///< [out] Calling thread's output.  May be aliased with \p input.
            T               &aggregate)         ///< [out] Total aggregate (valid in thread-lane<sub>0</sub>)
        {
            // Warp, thread-lane-IDs
            unsigned int warp_id = (WARPS == 1) ? 0 : (threadIdx.x / LOGICAL_WARP_THREADS);
            unsigned int lane_id = (WARPS == 1) ? threadIdx.x : (threadIdx.x & (LOGICAL_WARP_THREADS - 1));

            // Initialize identity region
            ThreadStore<PTX_STORE_VS>(&smem_storage.warp_scan[warp_id][lane_id], T(0));

            // Compute inclusive warp scan (has identity, share final)
            output = BasicScan<true, true>(
                smem_storage,
                warp_id,
                lane_id,
                input,
                Sum<T>());

            // Retrieve aggregate in thread-lane<sub>0</sub>
            aggregate = smem_storage.warp_scan[warp_id][WARP_SMEM_ELEMENTS - 1];
        }


        /// Inclusive prefix sum with aggregate and warp prefix
        static __device__ __forceinline__ void InclusiveSum(
            SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
            T               input,              ///< [in] Calling thread's input
            T               &output,            ///< [out] Calling thread's output.  May be aliased with \p input.
            T               &aggregate,         ///< [out] Total aggregate (valid in thread-lane<sub>0</sub>)
            T               &warp_prefix)       ///< [in-out] Warp-wide prefix to seed with (valid in thread-lane<sub>0</sub>).
        {
            // thread-lane-IDs
            unsigned int lane_id = (WARPS == 1) ? threadIdx.x : (threadIdx.x & (LOGICAL_WARP_THREADS - 1));

            // Incorporate warp-prefix from thread-lane<sub>0</sub>
            if (lane_id == 0)
            {
                input = warp_prefix + input;
            }

            // Compute inclusive warp scan
            InclusiveSum(smem_storage, input, output, aggregate);

            // Update warp_prefix
            warp_prefix += aggregate;
        }

        //@}
        /******************************************************************//**
         * \name Exclusive prefix sums
         *********************************************************************/
        //@{

        /**
         * \brief Computes an exclusive prefix sum in each logical warp.
         *
         * \smemreuse
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
         * \brief Computes an exclusive prefix sum in each logical warp.  Also computes the warp-wide \p aggregate of all inputs for thread-lane<sub>0</sub>.
         *
         * The \p aggregate is undefined in threads other than thread-lane<sub>0</sub>.
         *
         * \smemreuse
         */
        static __device__ __forceinline__ void ExclusiveSum(
            SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
            T               input,              ///< [in] Calling thread's input
            T               &output,            ///< [out] Calling thread's output.  May be aliased with \p input.
            T               &aggregate)         ///< [out] Total aggregate (valid in thread-lane<sub>0</sub>)
        {
            // Compute exclusive warp scan from inclusive warp scan
            T inclusive;
            InclusiveSum(smem_storage, input, inclusive, aggregate);
            output = inclusive - input;
        }


        /**
         * \brief Computes an exclusive prefix sum in each logical warp.  The \p warp_prefix value from thread-lane<sub>0</sub> is applied to all scan outputs.  Also computes the warp-wide \p aggregate of all inputs for thread-lane<sub>0</sub>.  The \p warp_prefix is further updated by the value of \p aggregate.
         *
         * The \p aggregate and \p warp_prefix are undefined in threads other than thread-lane<sub>0</sub>.
         *
         * \smemreuse
         */
        static __device__ __forceinline__ void ExclusiveSum(
            SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
            T               input,              ///< [in] Calling thread's input
            T               &output,            ///< [out] Calling thread's output.  May be aliased with \p input.
            T               &aggregate,         ///< [out] Total aggregate (valid in thread-lane<sub>0</sub>).
            T               &warp_prefix)       ///< [in-out] Warp-wide prefix to seed with (valid in thread-lane<sub>0</sub>).
        {
            // Warp, thread-lane-IDs
            unsigned int lane_id = (WARPS == 1) ? threadIdx.x : (threadIdx.x & (LOGICAL_WARP_THREADS - 1));

            // Incorporate warp-prefix from thread-lane<sub>0</sub>
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

        //@}
        /******************************************************************//**
         * \name Inclusive prefix scans
         *********************************************************************/
        //@{

        /**
         * \brief Computes an inclusive prefix sum using the specified binary scan functor in each logical warp.
         *
         * \smemreuse
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
            // Warp, thread-lane-IDs
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
         * \brief Computes an inclusive prefix sum using the specified binary scan functor in each logical warp.  Also computes the warp-wide \p aggregate of all inputs for thread-lane<sub>0</sub>.
         *
         * The \p aggregate is undefined in threads other than thread-lane<sub>0</sub>.
         *
         * \smemreuse
         *
         * \tparam ScanOp     [inferred] Binary scan functor type (a model of <a href="http://www.sgi.com/tech/stl/BinaryFunction.html">Binary Function</a>).
         */
        template <typename ScanOp>
        static __device__ __forceinline__ void InclusiveScan(
            SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
            T               input,              ///< [in] Calling thread's input
            T               &output,            ///< [out] Calling thread's output.  May be aliased with \p input.
            ScanOp          scan_op,            ///< [in] Binary associative scan functor.
            T               &aggregate)         ///< [out] Total aggregate (valid in thread-lane<sub>0</sub>)
        {
            // Warp, thread-lane-IDs
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
         * \brief Computes an inclusive prefix sum using the specified binary scan functor in each logical warp.  The \p warp_prefix value from thread-lane<sub>0</sub> is applied to all scan outputs.  Also computes the warp-wide \p aggregate of all inputs for thread-lane<sub>0</sub>.  The \p warp_prefix is further updated by the value of \p aggregate.
         *
         * The \p aggregate and \p warp_prefix are undefined in threads other than thread-lane<sub>0</sub>.
         *
         * \smemreuse
         *
         * \tparam ScanOp     [inferred] Binary scan functor type (a model of <a href="http://www.sgi.com/tech/stl/BinaryFunction.html">Binary Function</a>).
         */
        template <typename ScanOp>
        static __device__ __forceinline__ void InclusiveScan(
            SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
            T               input,              ///< [in] Calling thread's input
            T               &output,            ///< [out] Calling thread's output.  May be aliased with \p input.
            ScanOp          scan_op,            ///< [in] Binary associative scan functor.
            T               &aggregate,         ///< [out] Total aggregate (valid in thread-lane<sub>0</sub>).
            T               &warp_prefix)       ///< [in-out] Warp-wide prefix to seed with (valid in thread-lane<sub>0</sub>).
        {
            // Warp, thread-lane-IDs
            unsigned int lane_id = (WARPS == 1) ? threadIdx.x : (threadIdx.x & (LOGICAL_WARP_THREADS - 1));

            // Incorporate warp-prefix from thread-lane<sub>0</sub>
            if (lane_id == 0)
            {
                input = scan_op(warp_prefix, input);
            }

            // Compute inclusive warp scan
            InclusiveScan(smem_storage, input, output, scan_op, aggregate);

            // Update warp_prefix
            warp_prefix = scan_op(warp_prefix, aggregate);
        }


        //@}
        /******************************************************************//**
         * \name Exclusive prefix scans
         *********************************************************************/
        //@{

        /**
         * \brief Computes an exclusive prefix scan using the specified binary scan functor in each logical warp.
         *
         * \smemreuse
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
            // Warp, thread-lane-IDs
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
         * \brief Computes an exclusive prefix scan using the specified binary scan functor in each logical warp.  Also computes the warp-wide \p aggregate of all inputs for thread-lane<sub>0</sub>.
         *
         * The \p aggregate is undefined in threads other than thread-lane<sub>0</sub>.
         *
         * \smemreuse
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
            T               &aggregate)         ///< [out] Total aggregate (valid in thread-lane<sub>0</sub>)
        {
            // Warp id
            unsigned int warp_id = (WARPS == 1) ? 0 : (threadIdx.x / LOGICAL_WARP_THREADS);

            // Exclusive warp scan
            ExclusiveScan(smem_storage, input, output, identity, scan_op);

            // Retrieve aggregate
            aggregate = smem_storage.warp_scan[warp_id][WARP_SMEM_ELEMENTS - 1];
        }


        /**
         * \brief Computes an exclusive prefix scan using the specified binary scan functor in each logical warp.  The \p warp_prefix value from thread-lane<sub>0</sub> is applied to all scan outputs.  Also computes the warp-wide \p aggregate of all inputs for thread-lane<sub>0</sub>.  The \p warp_prefix is further updated by the value of \p aggregate.
         *
         * The \p aggregate and \p warp_prefix are undefined in threads other than thread-lane<sub>0</sub>.
         *
         * \smemreuse
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
            T               &aggregate,         ///< [out] Total aggregate (valid in thread-lane<sub>0</sub>).
            T               &warp_prefix)       ///< [in-out] Warp-wide prefix to seed with (valid in thread-lane<sub>0</sub>).
        {
            // Warp, thread-lane-IDs
            unsigned int lane_id = (WARPS == 1) ? threadIdx.x : (threadIdx.x & (LOGICAL_WARP_THREADS - 1));

            // Incorporate warp-prefix from thread-lane<sub>0</sub>
            if (lane_id == 0)
            {
                input = scan_op(warp_prefix, input);
            }

            // Exclusive warp scan
            ExclusiveScan(smem_storage, input, output, identity, scan_op, aggregate);

            // thread-lane<sub>0</sub> gets warp_prefix (instead of identity)
            if (lane_id == 0)
            {
                output = warp_prefix;
                warp_prefix = scan_op(warp_prefix, aggregate);
            }
        }


        //@}
        /******************************************************************//**
         * \name Exclusive prefix scans (without supplied identity)
         *********************************************************************/
        //@{


        /**
         * \brief Computes an exclusive prefix scan using the specified binary scan functor in each logical warp.  Because no identity value is supplied, the \p output computed for thread-lane<sub>0</sub> is invalid.
         *
         * \smemreuse
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
            // Warp, thread-lane-IDs
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
         * \brief Computes an exclusive prefix scan using the specified binary scan functor in each logical warp.  Because no identity value is supplied, the \p output computed for thread-lane<sub>0</sub> is invalid.  Also computes the warp-wide \p aggregate of all inputs for thread-lane<sub>0</sub>.
         *
         * The \p aggregate is undefined in threads other than thread-lane<sub>0</sub>.
         *
         * \smemreuse
         *
         * \tparam ScanOp     [inferred] Binary scan functor type (a model of <a href="http://www.sgi.com/tech/stl/BinaryFunction.html">Binary Function</a>).
         */
        template <typename ScanOp>
        static __device__ __forceinline__ void ExclusiveScan(
            SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
            T               input,              ///< [in] Calling thread's input
            T               &output,            ///< [out] Calling thread's output.  May be aliased with \p input.
            ScanOp          scan_op,            ///< [in] Binary associative scan functor.
            T               &aggregate)         ///< [out] Total aggregate (valid in thread-lane<sub>0</sub>)
        {
            // Warp id
            unsigned int warp_id = (WARPS == 1) ? 0 : (threadIdx.x / LOGICAL_WARP_THREADS);

            // Exclusive warp scan
            ExclusiveScan(smem_storage, input, output, scan_op);

            // Retrieve aggregate
            aggregate = smem_storage.warp_scan[warp_id][WARP_SMEM_ELEMENTS - 1];
        }


        /**
         * \brief Computes an exclusive prefix scan using the specified binary scan functor in each logical warp.  Because no identity value is supplied, the \p output computed for thread-thread-lane<sub>0</sub> is invalid.  The \p warp_prefix value from thread-thread-lane<sub>0</sub> is applied to all scan outputs.  Also computes the warp-wide \p aggregate of all inputs for thread-thread-lane<sub>0</sub>.  The \p warp_prefix is further updated by the value of \p aggregate.
         *
         * The \p aggregate and \p warp_prefix are undefined in threads other than thread-lane<sub>0</sub>.
         *
         * \smemreuse
         *
         * \tparam ScanOp     [inferred] Binary scan functor type (a model of <a href="http://www.sgi.com/tech/stl/BinaryFunction.html">Binary Function</a>).
         */
        template <typename ScanOp>
        static __device__ __forceinline__ void ExclusiveScan(
            SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
            T               input,              ///< [in] Calling thread's input
            T               &output,            ///< [out] Calling thread's output.  May be aliased with \p input.
            ScanOp          scan_op,            ///< [in] Binary associative scan functor.
            T               &aggregate,         ///< [out] Total aggregate (valid in thread-lane<sub>0</sub>).
            T               &warp_prefix)       ///< [in-out] Warp-wide prefix to seed with (valid in thread-lane<sub>0</sub>).
        {
            // Warp, thread-lane-IDs
            unsigned int lane_id = (WARPS == 1) ? threadIdx.x : (threadIdx.x & (LOGICAL_WARP_THREADS - 1));

            // Incorporate warp-prefix from thread-lane<sub>0</sub>
            if (lane_id == 0)
            {
                input = scan_op(warp_prefix, input);
            }

            // Exclusive warp scan
            ExclusiveScan(smem_storage, input, output, scan_op, aggregate);

            // thread-lane<sub>0</sub> gets warp_prefix (instead of identity)
            if (lane_id == 0)
            {
                output = warp_prefix;
                warp_prefix = scan_op(warp_prefix, aggregate);
            }
        }





    };


    };














public:

    /// The operations exposed by WarpScan require shared memory of this
    /// type.  This opaque storage can be allocated directly using the
    /// <tt>__shared__</tt> keyword.  Alternatively, it can be aliased to
    /// externally allocated shared memory or <tt>union</tt>'d with other types
    /// to facilitate shared memory reuse.
    typedef SmemStorage SmemStorage;


private:

    //---------------------------------------------------------------------
    // Template iteration structures.  (Regular iteration cannot always be
    // unrolled due to conditionals or ABI procedure calls within
    // functors).
    //---------------------------------------------------------------------

    /// General template iteration
    template <int COUNT, int MAX, bool HAS_IDENTITY, bool SHARE_FINAL>
    struct Iterate
    {
        /// Inclusive scan step
        template <typename ScanOp>
        static __device__ __forceinline__ T InclusiveScan(
            SmemStorage     &smem_storage,      ///< Shared reference to opaque SmemStorage layout
            unsigned int    warp_id,            ///< Warp id
            unsigned int    lane_id,            ///< thread-lane id
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
            unsigned int    lane_id,            ///< thread-lane id
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

    /******************************************************************//**
     * \name Inclusive prefix sums
     *********************************************************************/
    //@{

    /**
     * \brief Computes an inclusive prefix sum in each logical warp.
     *
     * \smemreuse
     */
    static __device__ __forceinline__ void InclusiveSum(
        SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
        T               input,              ///< [in] Calling thread's input
        T               &output)            ///< [out] Calling thread's output.  May be aliased with \p input.
    {
        // Warp, thread-lane-IDs
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
     * \brief Computes an inclusive prefix sum in each logical warp.  Also computes the warp-wide \p aggregate of all inputs for thread-lane<sub>0</sub>.
     *
     * The \p aggregate is undefined in threads other than thread-lane<sub>0</sub>.
     *
     * \smemreuse
     */
    static __device__ __forceinline__ void InclusiveSum(
        SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
        T               input,              ///< [in] Calling thread's input
        T               &output,            ///< [out] Calling thread's output.  May be aliased with \p input.
        T               &aggregate)         ///< [out] Total aggregate (valid in thread-lane<sub>0</sub>)
    {
        // Warp, thread-lane-IDs
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

        // Retrieve aggregate in thread-lane<sub>0</sub>
        aggregate = smem_storage.warp_scan[warp_id][WARP_SMEM_ELEMENTS - 1];
    }


    /**
     * \brief Computes an inclusive prefix sum in each logical warp.  The \p warp_prefix value from thread-lane<sub>0</sub> is applied to all scan outputs.  Also computes the warp-wide \p aggregate of all inputs for thread-lane<sub>0</sub>.  The \p warp_prefix is further updated by the value of \p aggregate.
     *
     * The \p aggregate and \p warp_prefix are undefined in threads other than thread-lane<sub>0</sub>.
     *
     * \smemreuse
     */
    static __device__ __forceinline__ void InclusiveSum(
        SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
        T               input,              ///< [in] Calling thread's input
        T               &output,            ///< [out] Calling thread's output.  May be aliased with \p input.
        T               &aggregate,         ///< [out] Total aggregate (valid in thread-lane<sub>0</sub>)
        T               &warp_prefix)       ///< [in-out] Warp-wide prefix to seed with (valid in thread-lane<sub>0</sub>).
    {
        // thread-lane-IDs
        unsigned int lane_id = (WARPS == 1) ? threadIdx.x : (threadIdx.x & (LOGICAL_WARP_THREADS - 1));

        // Incorporate warp-prefix from thread-lane<sub>0</sub>
        if (lane_id == 0)
        {
            input = warp_prefix + input;
        }

        // Compute inclusive warp scan
        InclusiveSum(smem_storage, input, output, aggregate);

        // Update warp_prefix
        warp_prefix += aggregate;
    }

    //@}
    /******************************************************************//**
     * \name Exclusive prefix sums
     *********************************************************************/
    //@{

    /**
     * \brief Computes an exclusive prefix sum in each logical warp.
     *
     * \smemreuse
     */
    static __device__ __forceinline__ void ExclusiveSum(
        SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
        T               input,              ///< [in] Calling thread's input
        T               &output)            ///< [out] Calling thread's output.  May be aliased with \p input.
    {
        // Compute exclusive warp scan from inclusive warp scan
        T inclusive;
        WarpScanInternal<POLICY>::InclusiveSum(smem_storage, input, inclusive);
        output = inclusive - input;
    }


    /**
     * \brief Computes an exclusive prefix sum in each logical warp.  Also computes the warp-wide \p aggregate of all inputs for thread-lane<sub>0</sub>.
     *
     * The \p aggregate is undefined in threads other than thread-lane<sub>0</sub>.
     *
     * \smemreuse
     */
    static __device__ __forceinline__ void ExclusiveSum(
        SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
        T               input,              ///< [in] Calling thread's input
        T               &output,            ///< [out] Calling thread's output.  May be aliased with \p input.
        T               &aggregate)         ///< [out] Total aggregate (valid in thread-lane<sub>0</sub>)
    {
        // Compute exclusive warp scan from inclusive warp scan
        T inclusive;
        WarpScanInternal<POLICY>::InclusiveSum(smem_storage, input, inclusive, aggregate);
        output = inclusive - input;
    }


    /**
     * \brief Computes an exclusive prefix sum in each logical warp.  The \p warp_prefix value from thread-lane<sub>0</sub> is applied to all scan outputs.  Also computes the warp-wide \p aggregate of all inputs for thread-lane<sub>0</sub>.  The \p warp_prefix is further updated by the value of \p aggregate.
     *
     * The \p aggregate and \p warp_prefix are undefined in threads other than thread-lane<sub>0</sub>.
     *
     * \smemreuse
     */
    static __device__ __forceinline__ void ExclusiveSum(
        SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
        T               input,              ///< [in] Calling thread's input
        T               &output,            ///< [out] Calling thread's output.  May be aliased with \p input.
        T               &aggregate,         ///< [out] Total aggregate (valid in thread-lane<sub>0</sub>).
        T               &warp_prefix)       ///< [in-out] Warp-wide prefix to seed with (valid in thread-lane<sub>0</sub>).
    {
        // Warp, thread-lane-IDs
        unsigned int lane_id = (WARPS == 1) ? threadIdx.x : (threadIdx.x & (LOGICAL_WARP_THREADS - 1));

        // Incorporate warp-prefix from thread-lane<sub>0</sub>
        T partial = input;
        if (lane_id == 0)
        {
            partial = warp_prefix + input;
        }

        // Compute exclusive warp scan from inclusive warp scan
        T inclusive;
        WarpScanInternal<POLICY>::InclusiveSum(smem_storage, partial, inclusive, aggregate);
        output = inclusive - input;

        // Update warp_prefix
        warp_prefix += aggregate;
    }

    //@}
    /******************************************************************//**
     * \name Inclusive prefix scans
     *********************************************************************/
    //@{

    /**
     * \brief Computes an inclusive prefix sum using the specified binary scan functor in each logical warp.
     *
     * \smemreuse
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
        WarpScanInternal<POLICY>::InclusiveScan(smem_storage, input, output, scan_op);
    }


    /**
     * \brief Computes an inclusive prefix sum using the specified binary scan functor in each logical warp.  Also computes the warp-wide \p aggregate of all inputs for thread-lane<sub>0</sub>.
     *
     * The \p aggregate is undefined in threads other than thread-lane<sub>0</sub>.
     *
     * \smemreuse
     *
     * \tparam ScanOp     [inferred] Binary scan functor type (a model of <a href="http://www.sgi.com/tech/stl/BinaryFunction.html">Binary Function</a>).
     */
    template <typename ScanOp>
    static __device__ __forceinline__ void InclusiveScan(
        SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
        T               input,              ///< [in] Calling thread's input
        T               &output,            ///< [out] Calling thread's output.  May be aliased with \p input.
        ScanOp          scan_op,            ///< [in] Binary associative scan functor.
        T               &aggregate)         ///< [out] Total aggregate (valid in thread-lane<sub>0</sub>)
    {
        WarpScanInternal<POLICY>::InclusiveScan(smem_storage, input, output, scan_op, aggregate);
    }


    /**
     * \brief Computes an inclusive prefix sum using the specified binary scan functor in each logical warp.  The \p warp_prefix value from thread-lane<sub>0</sub> is applied to all scan outputs.  Also computes the warp-wide \p aggregate of all inputs for thread-lane<sub>0</sub>.  The \p warp_prefix is further updated by the value of \p aggregate.
     *
     * The \p aggregate and \p warp_prefix are undefined in threads other than thread-lane<sub>0</sub>.
     *
     * \smemreuse
     *
     * \tparam ScanOp     [inferred] Binary scan functor type (a model of <a href="http://www.sgi.com/tech/stl/BinaryFunction.html">Binary Function</a>).
     */
    template <typename ScanOp>
    static __device__ __forceinline__ void InclusiveScan(
        SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
        T               input,              ///< [in] Calling thread's input
        T               &output,            ///< [out] Calling thread's output.  May be aliased with \p input.
        ScanOp          scan_op,            ///< [in] Binary associative scan functor.
        T               &aggregate,         ///< [out] Total aggregate (valid in thread-lane<sub>0</sub>).
        T               &warp_prefix)       ///< [in-out] Warp-wide prefix to seed with (valid in thread-lane<sub>0</sub>).
    {
        // Warp, thread-lane-IDs
        unsigned int lane_id = (WARPS == 1) ? threadIdx.x : (threadIdx.x & (LOGICAL_WARP_THREADS - 1));

        // Incorporate warp-prefix from thread-lane<sub>0</sub>
        if (lane_id == 0)
        {
            input = scan_op(warp_prefix, input);
        }

        // Compute inclusive warp scan
        WarpScanInternal<POLICY>::InclusiveScan(smem_storage, input, output, scan_op, aggregate);

        // Update warp_prefix
        warp_prefix = scan_op(warp_prefix, aggregate);
    }


    //@}
    /******************************************************************//**
     * \name Exclusive prefix scans
     *********************************************************************/
    //@{

    /**
     * \brief Computes an exclusive prefix scan using the specified binary scan functor in each logical warp.
     *
     * \smemreuse
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
        WarpScanInternal<POLICY>::ExclusiveScan(smem_storage, input, output, identity, scan_op);
    }


    /**
     * \brief Computes an exclusive prefix scan using the specified binary scan functor in each logical warp.  Also computes the warp-wide \p aggregate of all inputs for thread-lane<sub>0</sub>.
     *
     * The \p aggregate is undefined in threads other than thread-lane<sub>0</sub>.
     *
     * \smemreuse
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
        T               &aggregate)         ///< [out] Total aggregate (valid in thread-lane<sub>0</sub>)
    {
        WarpScanInternal<POLICY>::ExclusiveScan(smem_storage, input, output, identity, scan_op, aggregate);
    }


    /**
     * \brief Computes an exclusive prefix scan using the specified binary scan functor in each logical warp.  The \p warp_prefix value from thread-lane<sub>0</sub> is applied to all scan outputs.  Also computes the warp-wide \p aggregate of all inputs for thread-lane<sub>0</sub>.  The \p warp_prefix is further updated by the value of \p aggregate.
     *
     * The \p aggregate and \p warp_prefix are undefined in threads other than thread-lane<sub>0</sub>.
     *
     * \smemreuse
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
        T               &aggregate,         ///< [out] Total aggregate (valid in thread-lane<sub>0</sub>).
        T               &warp_prefix)       ///< [in-out] Warp-wide prefix to seed with (valid in thread-lane<sub>0</sub>).
    {
        // Warp, thread-lane-IDs
        unsigned int lane_id = (WARPS == 1) ? threadIdx.x : (threadIdx.x & (LOGICAL_WARP_THREADS - 1));

        // Incorporate warp-prefix from thread-lane<sub>0</sub>
        if (lane_id == 0)
        {
            input = scan_op(warp_prefix, input);
        }

        // Exclusive warp scan
        WarpScanInternal<POLICY>::ExclusiveScan(smem_storage, input, output, identity, scan_op, aggregate);

        // thread-lane<sub>0</sub> gets warp_prefix (instead of identity)
        if (lane_id == 0)
        {
            output = warp_prefix;
            warp_prefix = scan_op(warp_prefix, aggregate);
        }
    }


    //@}
    /******************************************************************//**
     * \name Exclusive prefix scans (without supplied identity)
     *********************************************************************/
    //@{


    /**
     * \brief Computes an exclusive prefix scan using the specified binary scan functor in each logical warp.  Because no identity value is supplied, the \p output computed for thread-lane<sub>0</sub> is invalid.
     *
     * \smemreuse
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
        WarpScanInternal<POLICY>::ExclusiveScan(smem_storage, input, output, scan_op);
    }


    /**
     * \brief Computes an exclusive prefix scan using the specified binary scan functor in each logical warp.  Because no identity value is supplied, the \p output computed for thread-lane<sub>0</sub> is invalid.  Also computes the warp-wide \p aggregate of all inputs for thread-lane<sub>0</sub>.
     *
     * The \p aggregate is undefined in threads other than thread-lane<sub>0</sub>.
     *
     * \smemreuse
     *
     * \tparam ScanOp     [inferred] Binary scan functor type (a model of <a href="http://www.sgi.com/tech/stl/BinaryFunction.html">Binary Function</a>).
     */
    template <typename ScanOp>
    static __device__ __forceinline__ void ExclusiveScan(
        SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
        T               input,              ///< [in] Calling thread's input
        T               &output,            ///< [out] Calling thread's output.  May be aliased with \p input.
        ScanOp          scan_op,            ///< [in] Binary associative scan functor.
        T               &aggregate)         ///< [out] Total aggregate (valid in thread-lane<sub>0</sub>)
    {
        WarpScanInternal<POLICY>::ExclusiveScan(smem_storage, input, output, scan_op, aggregate);
    }


    /**
     * \brief Computes an exclusive prefix scan using the specified binary scan functor in each logical warp.  Because no identity value is supplied, the \p output computed for thread-thread-lane<sub>0</sub> is invalid.  The \p warp_prefix value from thread-thread-lane<sub>0</sub> is applied to all scan outputs.  Also computes the warp-wide \p aggregate of all inputs for thread-thread-lane<sub>0</sub>.  The \p warp_prefix is further updated by the value of \p aggregate.
     *
     * The \p aggregate and \p warp_prefix are undefined in threads other than thread-lane<sub>0</sub>.
     *
     * \smemreuse
     *
     * \tparam ScanOp     [inferred] Binary scan functor type (a model of <a href="http://www.sgi.com/tech/stl/BinaryFunction.html">Binary Function</a>).
     */
    template <typename ScanOp>
    static __device__ __forceinline__ void ExclusiveScan(
        SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
        T               input,              ///< [in] Calling thread's input
        T               &output,            ///< [out] Calling thread's output.  May be aliased with \p input.
        ScanOp          scan_op,            ///< [in] Binary associative scan functor.
        T               &aggregate,         ///< [out] Total aggregate (valid in thread-lane<sub>0</sub>).
        T               &warp_prefix)       ///< [in-out] Warp-wide prefix to seed with (valid in thread-lane<sub>0</sub>).
    {
        // Warp, thread-lane-IDs
        unsigned int lane_id = (WARPS == 1) ? threadIdx.x : (threadIdx.x & (LOGICAL_WARP_THREADS - 1));

        // Incorporate warp-prefix from thread-lane<sub>0</sub>
        if (lane_id == 0)
        {
            input = scan_op(warp_prefix, input);
        }

        // Exclusive warp scan
        WarpScanInternal<POLICY>::ExclusiveScan(smem_storage, input, output, scan_op, aggregate);

        // thread-lane<sub>0</sub> gets warp_prefix (instead of identity)
        if (lane_id == 0)
        {
            output = warp_prefix;
            warp_prefix = scan_op(warp_prefix, aggregate);
        }
    }

    //@}
};

/** @} */       // end of SimtCoop group

} // namespace cub
CUB_NS_POSTFIX
