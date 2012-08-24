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

/******************************************************************************
 * CUB umbrella include file
 ******************************************************************************/

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

