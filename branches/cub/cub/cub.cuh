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
 * CUB is a library of reusable SIMT primitives for CUDA kernel programming. It
 * provides commonplace CTA-wide, warp-wide, and thread-level operations that
 * are flexible and tunable to fit your needs.  CUB accommodates your specific:
 * - Data types
 * - Width of parallelism (CTA threads)
 * - Grain size (data items per thread)
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
 * compose CUB primitives for computing a parallel prefix sum
 * across CTA threads:
 *
 * \par
 * \code
 * #include <cub.cuh>
 *
 * // An exclusive prefix sum kernel (assuming only a single CTA)
 * template <
 *      int         CTA_THREADS,                        // Threads per CTA
 *      int         KEYS_PER_THREAD,                    // Items per thread
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
 *      T data[KEYS_PER_THREAD];
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
 * -# <b>The ease of sequential programming.</b>  Parallel primitives within kernels
 * can be simply sequenced together (similar to Thrust programming on the host).
 * -# <b>The benefits of transparent performance-portability.</b> Kernels can be
 * simply recompiled against new CUB releases (instead of hand-rewritten)
 * to leverage new algorithmic developments, hardware instructions, etc.
 *
 * \section sec3 What are the challenges of SIMT code reuse?
 *
 * \par
 * CUDA's data-parallel programming model complicates the prospect of software
 * reuse for thread-cooperative operations (e.g., CTA-reduce, CTA-sort, etc.).
 * The construction and usage of such SIMT components are not as straightforward
 * as traditional procedure definitions and function calls.  For example, the
 * shared memory layout and the number of steps for a CTA-wide reduction are
 * very specific to:
 * - The number of CTA threads
 * - The data type being reduced
 * - The number of items contributed by each thread
 * - The underlying architecture's warp width
 * - The underlying architecture's rules for bank conflicts
 *
 * \par
 * These configuration details will vary considerably in the context of
 * different application kernels, yet a reusable SIMT component must
 * accommodate the entire configuration domain.  Furthermore, the
 * interface of any primitive that has been specialized to a given
 * configuration needs to expose the corresponding shared memory requirement
 * to the calling code (where it can be allocated and possibly reused
 * elsewhere by the CTA).
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
