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
 *
 ******************************************************************************/

#pragma once

#include "../../cub/cub.cuh"
#include "../ns_wrapper.cuh"

#include "cta_upsweep_pass.cuh"

BACK40_NS_PREFIX
namespace back40 {
namespace radix_sort {


/**
 * Kernel entry point
 */
template <
    typename    CtaUpsweepPassPolicy,
    int         MIN_CTA_OCCUPANCY,
    typename    KeyType,
    typename    SizeT>
__launch_bounds__ (
    CtaUpsweepPassPolicy::CTA_THREADS,
    MIN_CTA_OCCUPANCY)
__global__
void UpsweepKernel(
    KeyType                     *d_keys_in,
    SizeT                       *d_spine,
    cub::CtaEvenShare<SizeT>    cta_even_share,
    unsigned int                current_bit)
{
    // Constants
    enum
    {
        TILE_ITEMS      = CtaUpsweepPassPolicy::TILE_ITEMS,
        RADIX_DIGITS    = 1 << CtaUpsweepPassPolicy::RADIX_BITS,
    };

    // CTA abstraction types
    typedef CtaUpsweepPass<CtaUpsweepPassPolicy, KeyType, SizeT> CtaUpsweepPassT;

    // Shared data structures
    __shared__ typename CtaUpsweepPassT::SmemStorage smem_storage;

    // Determine our threadblock's work range
    cta_even_share.Init();

    // Compute bin-count for each radix digit (valid in the first RADIX_DIGITS threads)
    SizeT bin_count;
    CtaUpsweepPassT::UpsweepPass(
        smem_storage,
        d_keys_in + cta_even_share.cta_offset,
        current_bit,
        cta_even_share.cta_items,
        bin_count);

    // Write out the bin_count reductions
    if (threadIdx.x < RADIX_DIGITS)
    {
        int spine_bin_offset = (gridDim.x * threadIdx.x) + blockIdx.x;
        d_spine[spine_bin_offset] = bin_count;
    }
}


/**
 * Upsweep kernel properties
 */
template <
    typename KeyType,
    typename SizeT>
struct UpsweepKernelProps : cub::KernelProps
{
    // Kernel function type
    typedef void (*KernelFunc)(
        KeyType*,
        SizeT*,
        cub::CtaEvenShare<SizeT>,
        unsigned int);

    // Fields
    KernelFunc              kernel_func;
    int                     tile_items;
    cudaSharedMemConfig     sm_bank_config;
    int                     radix_bits;

    /**
     * Initializer
     */
    template <
        typename CtaUpsweepPassPolicy,
        typename OpaqueCtaUpsweepPassPolicy,
        int MIN_CTA_OCCUPANCY>
    cudaError_t Init(const cub::CudaProps &cuda_props)    // CUDA properties for a specific device
    {
        // Initialize fields
        kernel_func             = UpsweepKernel<OpaqueCtaUpsweepPassPolicy, MIN_CTA_OCCUPANCY>;
        tile_items             = CtaUpsweepPassPolicy::TILE_ITEMS;
        sm_bank_config             = CtaUpsweepPassPolicy::SMEM_CONFIG;
        radix_bits                = CtaUpsweepPassPolicy::RADIX_BITS;

        // Initialize super class
        return cub::KernelProps::Init(
            kernel_func,
            CtaUpsweepPassPolicy::CTA_THREADS,
            cuda_props);
    }

    /**
     * Initializer
     */
    template <
        typename CtaUpsweepPassPolicy,
        int MIN_CTA_OCCUPANCY>
    cudaError_t Init(const cub::CudaProps &cuda_props)    // CUDA properties for a specific device
    {
        return Init<CtaUpsweepPassPolicy, CtaUpsweepPassPolicy, MIN_CTA_OCCUPANCY>(cuda_props);
    }
};


} // namespace radix_sort
} // namespace back40
BACK40_NS_POSTFIX
