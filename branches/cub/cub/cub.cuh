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
 * \section sec0 What is CUB?
 *
 * CUB is a library of reusable primitives for CUDA kernel programming. It's
 * a collection of CTA-wide, warp-wide, and thread primitives that are flexible
 * to fit your needs, i.e., your specific:
 * - data types
 * - parallelism (CTA threads)
 * - grain size (data items per thread)
 *
 * CUB also includes many useful host-primitives as well.
 *
 * \section sec1 Why do you need CUB?
 *
 * Abstraction layers, program modularity, and code reuse are Good Ideas.  They are
 * critical aspects of sustainable software engineering.  Complexity, risk, and
 * maintenance costs can all be mitigated by encapsulating sophisticated code
 * behind simple interfaces.
 *
 * However, the prospect of software reuse within CUDA is complicated.  The
 * programming model has two distinct software environments: (1) the sequential host
 * program and (2) the SIMT device kernels.  On one hand, libraries of GPU
 * primitives for host programs are abundant (Thrust, CUBLAS, etc.) and can
 * obviate the need to reason about the complexity of parallelism at all.
 * Yet there is negligible software reuse within GPU kernels themselves.
 * Kernel development is the more complicated arena of CUDA programming,
 * and virtually every kernel is written completely from scratch.
 *
 * This is an unfortunate state of affairs.  It can be extremely challenging to
 * implement commonplace CTA-wide operations (reduction, prefix sum, sort,
 * merge, etc.) that are work-efficient and avoid architecture-specific hazards
 * such as bank conflicts.  Furthermore, GPU computing requires efficient
 * utilization of the underlying hardware, a moving target that evolves
 * significantly every eighteen months.  It is unreasonable to continually
 * rewrite existing kernels for new microarchitectures that provision new
 * resources/instructions.  CUDA programmers are desperate for CTA-level
 * abstraction layers that provide:
 * -# The straightforward sequencing of parallel primitives within kernels
 * (similar to Thrust programming on the host)
 * -# Transparent benefits for simply recompiling kernels against fresh
 * libraries of high performance CTA primitives.
 *
 * This dearth of software reuse is a consequence of the daunting flexibility
 * and complexity needed to construct abstract SIMT components.  For example,
 * the code generation and shared memory layout for a CTA-wide prefix sum is
 * dependent on (1) the number of CTA threads, (2) the data type being
 * summed, (3) the number of items per thread, (4) warp width, and (5)
 * architecture rules for bank conflicts.  These details will vary
 * considerably in the context of different application kernels.  Furthermore,
 * the interface for CTA-wide prefix sum needs to export the corresponding
 * shared memory requirement to the caller (where it can be allocated and
 * possibly reused elsewhere by the CTA).
 *
 */
