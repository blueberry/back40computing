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
 * CUB umbrella include file
 */

#pragma once

// Grid
#include "grid/grid_barrier.cuh"
#include "grid/grid_queue.cuh"

// CTA
#include "cta/cta_load.cuh"
#include "cta/cta_even_share.cuh"
#include "cta/cta_radix_rank.cuh"
#include "cta/cta_radix_sort.cuh"
#include "cta/cta_reduce.cuh"
#include "cta/cta_scan.cuh"
#include "cta/cta_store.cuh"
#include "cta/cta_discontinuity.cuh"

// Warp
#include "warp/warp_scan.cuh"

// Thread
#include "thread/thread_load.cuh"
#include "thread/thread_reduce.cuh"
#include "thread/thread_scan.cuh"
#include "thread/thread_store.cuh"

// Host
#include "host/cuda_props.cuh"
#include "host/kernel_props.cuh"
#include "host/multi_buffer.cuh"
#include "host/spinlock.cuh"

// General
#include "debug.cuh"
#include "macro_utils.cuh"
#include "device_props.cuh"
#include "operators.cuh"
#include "ptx_intrinsics.cuh"
#include "type_utils.cuh"
#include "vector_type.cuh"
#include "allocator.cuh"


/**
 * \mainpage
 *
 * \tableofcontents
 *
 * \section sec1 (1) What is CUB?
 *
 * \par
 * CUB is a library of high performance SIMT primitives for CUDA kernel
 * programming. CUB enhances productivity and portability
 * by providing commonplace CTA-wide, warp-wide, and thread-level operations that
 * are flexible and tunable to fit your kernel needs.
 *
 * \par
 * Browse our collections of:
 * - [<b>SIMT cooperative primitives</b>](annotated.html)
 *   - CtaRadixSort, CtaReduce, WarpScan, etc.
 * - [<b>SIMT utilities</b>](group___simt_utils.html)
 *   - CTA loads/stores in blocked/striped arrangements (vectorized, coalesced, etc.)
 *   - Sequential ThreadScan, ThreadReduce, etc.
 *   - Cache-modified ThreadLoad/ThreadStore
 * - [<b>Host utilities</b>](group___host_util.html)
 *   - Caching allocators, error handling, etc.
 *
 * \section sec2 (2) A simple example
 *
 * \par
 * The following kernel snippet illustrates how easy it is to
 * compose CUB's scan and data-movement primitives into a
 * single kernel for computing prefix sum:
 *
 * \par
 * \code
 * #include <cub.cuh>
 *
 * // An exclusive prefix sum kernel (assuming only a single CTA)
 * template <
 *      int         CTA_THREADS,                        // Threads per CTA
 *      int         ITEMS_PER_THREAD,                   // Items per thread
 *      typename    T>                                  // Data type
 * __global__ void PrefixSumKernel(T *d_in, T *d_out)
 * {
 *      using namespace cub;
 *
 *      // Parameterize a CtaScan type for use in the current problem context
 *      typedef CtaScan<T, CTA_THREADS> CtaScan;
 *
 *      // The shared memory for CtaScan
 *      __shared__ typename CtaScan::SmemStorage smem_storage;
 *
 *      // A segment of data items per thread
 *      T data[ITEMS_PER_THREAD];
 *
 *      // Load a tile of data using vector-load instructions
 *      CtaLoadVectorized(data, d_in, 0);
 *
 *      // Perform an exclusive prefix sum across the tile of data
 *      CtaScan::ExclusiveSum(smem_storage, data, data);
 *
 *      // Store a tile of data using vector-load instructions
 *      CtaStoreVectorized(data, d_out, 0);
 * }
 * \endcode
 *
 * \par
 * The cub::CtaScan primitive implements an efficient prefix sum across CTA
 * threads that is specialized to the underlying architecture.
 * It is parameterized by the number of CTA threads and the aggregate
 * data type \p T.  Once instantiated, it exposes the opaque
 * cub::CtaScan::SmemStorage type which allows us to allocate the shared memory
 * needed by the primitive.
 *
 * \par
 * Furthermore, the kernel uses CUB's primitives for vectorizing global
 * loads and stores.  For example, <tt>ld.global.v4.s32</tt> will be generated when
 * \p T = \p int and \p ITEMS_PER_THREAD is a multiple of 4.
 *
 * \section sec3 (3) Why do you need CUB?
 *
 * \par
 * With the exception of CUB, there are few (if any) software libraries of
 * reusable CTA-level primitives.  This is unfortunate, especially for
 * complex algorithms with intricate dependences between threads.  For cooperative
 * problems, the SIMT kernel is often the most complex and performance-sensitive
 * layer in the CUDA software stack.  Best practices would have us
 * leverage libraries and abstraction layers to help  mitigate the complexity,
 * risks, and maintenance costs of this software.
 *
 * \par
 * As a SIMT library and software abstraction layer, CUB gives you:
 * -# <b>The ease of sequential programming.</b>  Parallel primitives within
 * kernels can be simply sequenced together (similar to Thrust programming on
 * the host).
 * -# <b>The benefits of transparent performance-portability.</b> Kernels can
 * be simply recompiled against new CUB releases (instead of hand-rewritten)
 * to leverage new algorithmic developments, hardware instructions, etc.
 *
 * \section sec4 (4) How does CUB work?
 *
 * \par
 * CUB leverages the following programming idioms:
 * -# [<b>C++ templates</b>](index.html#sec3sec1)
 * -# [<b>Reflective type structure</b>](index.html#sec3sec2)
 * -# [<b>Flexible data arrangement among threads</b>](index.html#sec3sec3)
 *
 * \subsection sec3sec1 4.1 &nbsp;&nbsp; C++ templates
 *
 * \par
 * As a SIMT library, CUB must be flexible enough to accommodate a wide spectrum
 * of <em>problem contexts</em>,
 * i.e., specific:
 *    - Data types
 *    - Widths of parallelism (CTA threads)
 *    - Grain sizes (data items per thread)
 *    - Underlying architectures (special instructions, warp width, rules for bank conflicts, etc.)
 *    - Tuning requirements (e.g., latency vs. throughput)
 *
 * \par
 * To provide this flexibility, CUB is implemented as a C++ template library.
 * C++ templates are a way to write generic algorithms and data structures.
 * There is no need to build CUB separately.  You simply #<tt>include</tt> the
 * <tt>cub.cuh</tt> header file into your <tt>.cu</tt> or <tt>.cpp</tt> sources
 * and compile with CUDA's <tt>nvcc</tt> compiler.
 *
 * \subsection sec3sec2 4.2 &nbsp;&nbsp; Reflective type structure
 *
 * \par
 * Cooperation requires shared memory for communicating between threads.
 * However, the specific size and layout of the memory needed by a given
 * primitive will be specific to the details of its problem context (e.g., how
 * many threads are calling into it, how many items per thread, etc.).  Furthermore,
 * this shared memory must be allocated externally to the component if it is to be
 * reused elsewhere by the CTA.
 *
 * \par
 * \code
 * // Parameterize a CtaRadixSort type for use with 128 threads
 * // and 4 items per thread
 * typedef cub::CtaRadixSort<unsigned int, 128, 4> CtaRadixSort;
 *
 * // Declare shared memory for CtaRadixSort
 * __shared__ typename CtaRadixSort::SmemStorage smem_storage;
 *
 * // A segment of consecutive input items per thread
 * int keys[4];
 *
 * // Obtain keys in blocked order
 * ...
 *
 * // Sort keys in ascending order
 * CtaRadixSort::SortBlocked(smem_storage, keys);
 *
 * \endcode
 *
 * \par
 * To address this issue, we encapsulate cooperative procedures within
 * <em>reflective type structure</em> (C++ classes).  As illustrated in the
 * cub::CtaRadixSort example above, these primitives are C++ classes with
 * interfaces that expose both (1) procedural methods as well as (2) the opaque
 * shared memory types needed for their operation.
 *
* \subsection sec3sec3 4.3 &nbsp;&nbsp; Flexible data arrangement among threads
 *
 * \par
 * We often design kernels such that each CTA is assigned a "tile" of data
 * items for processing.  When the tile size is equal to the CTA size, the
 * mapping of data onto threads is straightforward (1:1 items
 * to CTA threads). However, it is often desirable to design a tile size
 * that is a constant factor larger than the CTA size.  In this case, each thread
 * is assigned multiple data items.  The benefits of doing so include:
 * - <b>Algorithmic efficiency</b>.  Sequential work over multiple items in
 *   thread-private registers is cheaper than synchronized, cooperative
 *   work through shared memory spaces
 * - <b>Data occupancy</b>.  The number of items that can be resident on-chip in
 *   thread-private register storage is often greater than the number of
 *   schedulable threads
 * - <b>Instruction-level parallelism</b>.  Multiple items per thread also
 *   facilitates greater ILP for improved throughput and utilization
 *
 * \par
 * When tile size is greater than CTA size, we are faced with the question
 * how tile items should partitioned among CTA threads.  For linearly-ordered
 * tiles, CUB supports the following two arrangement patterns:
 * - <b><em>Blocked</em> arrangement</b>.  The aggregate tile of items is partitioned
 *   evenly across threads in "blocked" fashion with thread<sub><em>i</em></sub>
 *   owning the <em>i</em><sup>th</sup> segment of consecutive elements.
 * - <b><em>Striped</em> arrangement</b>.  The aggregate tile of items is partitioned across
 *   threads in "striped" fashion, i.e., the \p ITEMS_PER_THREAD items owned by
 *   each thread have logical stride \p CTA_THREADS between them.
 * <br><br>
 * \image html thread_data_1.png
 * <center><b>Blocked vs. striped arrangements with \p CTA_THREADS = 4 and \p ITEMS_PER_THREAD = 2</b></center>
 * <br>
 *
 * \par
 * The cub::CtaExchange primitive provides operations for converting between blocked
 * and striped arrangements. Blocked arrangements are often desirable for
 * algorithmic benefits (where long sequences of items can be processed sequentially
 * within each thread).  Striped arrangements are often desirable for data movement
 * through global memory (where read/write coalescing is a important performance
 * consideration).
 * <br><br>
 * \image html thread_data_2.png
 * <center><b>Exchanging items between blocked vs. striped arrangements (illustrated as a 2D transpose with \p CTA_THREADS = 8 and \p ITEMS_PER_THREAD = 4)</b></center>
 * <br>
 *
 * \par
 * We can also illustrate these arrangements as two dimensional representations (items-per-thread
 * vs. threads-per-CTA).  In this form, we see that conversions between the two
 * arrangements are isomorphic to 2D matrix transpose operations.
 *
 */


/**
 * \defgroup Simt SIMT Primitives
 */

/**
 * \defgroup SimtCoop Cooperative SIMT Operations
 * \ingroup Simt
 */

/**
 * \defgroup SimtUtils SIMT Utilities
 * \ingroup Simt
 */

/**
 * \defgroup HostUtil Host Utilities
 */
