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

#include "cta/cta_global_barrier.cuh"
#include "cta/cta_load.cuh"
#include "cta/cta_progress.cuh"
#include "cta/cta_radix_rank.cuh"
#include "cta/cta_radix_sort.cuh"
#include "cta/cta_reduce.cuh"
#include "cta/cta_scan.cuh"
#include "cta/cta_store.cuh"

#include "host/allocator.cuh"
#include "host/cuda_props.cuh"
#include "host/debug.cuh"
#include "host/kernel_props.cuh"
#include "host/multi_buffer.cuh"
#include "host/spinlock.cuh"

#include "thread/thread_load.cuh"
#include "thread/thread_reduce.cuh"
#include "thread/thread_scan.cuh"
#include "thread/thread_store.cuh"

#include "warp/warp_scan.cuh"

#include "macro_utils.cuh"
#include "device_props.cuh"
#include "operators.cuh"
#include "ptx_intrinsics.cuh"
#include "type_utils.cuh"
#include "vector_type.cuh"

/**
 * \mainpage
 *
 * \tableofcontents
 *
 * \section sec0 What is CUB?
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
 * - [<b>SIMT utilities</b>](group___simt_utils.html)
 * - [<b>Host utilities</b>](group___host_util.html)
 *
 * \section sec1 A simple example
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
 *      // Declare a parameterized CtaScan type for the given kernel configuration
 *      typedef CtaScan<T, CTA_THREADS> CtaScan;
 *
 *      // The shared memory for CtaScan
 *      __shared__ typename CtaScan::SmemStorage smem_storage;
 *
 *      // A segment of data items per thread
 *      T data[ITEMS_PER_THREAD];
 *
 *      // Load a tile of data using vector-load instructions if possible
 *      CtaLoadVectorized(data, d_in, 0);
 *
 *      // Perform an exclusive prefix sum across the tile of data
 *      CtaScan::ExclusiveSum(smem_storage, data, data);
 *
 *      // Store a tile of data using vector-load instructions if possible
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
 * \section sec2 Why do you need CUB?
 *
 * \par
 * Whereas data-parallelism is easy to implement, cooperative-parallelism
 * is hard.  For algorithms requiring local cooperation between threads, the
 * SIMT kernel is the most complex (and performance-sensitive) layer
 * in the CUDA software stack.  Developers must carefully manage the state
 * and interaction of many, many threads.  Best practices would have us
 * leverage libraries and abstraction layers to help mitigate the complexity,
 * risks, and maintenance costs of this software.  However, with the exception
 * of CUB, there are few (if any) software libraries of reusable CTA-level
 * primitives.
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
 * \section sec3 How does CUB work?
 * \subsection sec3sec0 Library overview
 *
 * \par
 * CUDA's SIMT programming model complicates the prospect of software
 * reuse within kernel code.  As a SIMT library, CUB must accommodate
 * arbitrary kernel <em>configuration contexts</em>, i.e., specific:
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
 * appropriate header files into your <tt>.cu</tt> or <tt>.cpp</tt> sources
 * and compile with CUDA's <tt>nvcc</tt> compiler.
 *
 * \subsection sec3sec1 Reflective type structure
 *
 * \par
 * Cooperative SIMT components cannot be constructed and composed like we
 * would traditional procedures and function calls.  They require shared
 * memory whose layout for a given primitive will be specific to the details
 * of the calling kernel's <em>configuration context</em> (see above).
 * Furthermore, this shared memory must be allocated externally to the
 * component if it is to be reused elsewhere by the CTA.
 *
 * \par
 * To address this issue, we encapsulate cooperative procedures within
 * reflective type structure (C++ classes).  Thus our CUB primitives
 * are C++ classes with interfaces that expose both procedural methods
 * as well as the opaque shared memory types needed for their operation.
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
