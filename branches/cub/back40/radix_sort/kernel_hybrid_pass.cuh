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

#include "sort_utils.cuh"

#include "cta_single_tile.cuh"
//#include "cta_downsweep_pass.cuh"
//#include "cta_upsweep_pass.cuh"


BACK40_NS_PREFIX
namespace back40 {
namespace radix_sort {

//---------------------------------------------------------------------
// Tuning policy types
//---------------------------------------------------------------------


/**
 * Hybrid CTA tuning policy
 */
template <
    int                         RADIX_BITS,                     // The number of radix bits, i.e., log2(bins)
    int                         _CTA_THREADS,                   // The number of threads per CTA

    int                         UPSWEEP_THREAD_ITEMS,           // The number of consecutive upsweep items to load per thread per tile
    int                         DOWNSWEEP_THREAD_ITEMS,         // The number of consecutive downsweep items to load per thread per tile
    ScatterStrategy             DOWNSWEEP_SCATTER_STRATEGY,     // Downsweep strategy
    int                         SINGLE_TILE_THREAD_ITEMS,       // The number of consecutive single-tile items to load per thread per tile

    cub::PtxLoadModifier        LOAD_MODIFIER,                  // Load cache-modifier
    cub::PtxStoreModifier       STORE_MODIFIER,                 // Store cache-modifier
    cudaSharedMemConfig         _SMEM_CONFIG>                   // Shared memory bank size
struct CtaHybridPassPolicy
{
    enum
    {
        CTA_THREADS = _CTA_THREADS,
    };

    static const cudaSharedMemConfig SMEM_CONFIG = _SMEM_CONFIG;
/*
    // Upsweep pass policy
    typedef CtaUpsweepPassPolicy<
        RADIX_BITS,
        CTA_THREADS,
        UPSWEEP_THREAD_ITEMS,
        LOAD_MODIFIER,
        STORE_MODIFIER,
        SMEM_CONFIG> CtaUpsweepPassPolicyT;

    // Downsweep pass policy
    typedef CtaDownsweepPassPolicy<
        RADIX_BITS,
        CTA_THREADS,
        DOWNSWEEP_THREAD_ITEMS,
        DOWNSWEEP_SCATTER_STRATEGY,
        LOAD_MODIFIER,
        STORE_MODIFIER,
        SMEM_CONFIG> CtaDownsweepPassPolicyT;

    // Single tile policy
    typedef CtaSingleTilePolicy<
        RADIX_BITS,
        CTA_THREADS,
        SINGLE_TILE_THREAD_ITEMS,
        LOAD_MODIFIER,
        STORE_MODIFIER,
        SMEM_CONFIG> CtaSingleTilePolicyT;
*/
};

//---------------------------------------------------------------------
// CTA-wide abstractions
//---------------------------------------------------------------------


/**
 * Kernel params
 */
template <
    typename    SizeT,
    typename    KeyType,
    typename    ValueType>
struct HybridKernelParams
{
    KeyType                 *d_keys[2];
    ValueType               *d_values[2];
    SizeT                   *d_spine;

    BinDescriptor<SizeT>    *d_global[2];
    BinDescriptor<SizeT>    *d_local[2];

    cub::GridQueue<int>     q_global[2];
    cub::GridQueue<int>     q_local[2];

    unsigned int            global_cutoff;
};




/**
 * Kernel entry point
 */
template <
    typename    CtaHybridPassPolicy,
    int         MIN_CTA_OCCUPANCY,
    typename    SizeT,
    typename    KeyType,
    typename    ValueType>
__launch_bounds__ (
    CtaHybridPassPolicy::CTA_THREADS,
    MIN_CTA_OCCUPANCY)
__global__
void HybridKernel(
    HybridKernelParams<SizeT, KeyType, ValueType> params)
{
    enum
    {
        CTA_THREADS         = CtaHybridPassPolicy::CTA_THREADS,
/*
        GLOBAL_RADIX_BITS   = CtaHybridPassPolicy::CtaUpsweepPassPolicyT::RADIX_BITS,
        GLOBAL_RADIX_DIGITS = 1 << RADIX_BITS,

        LARGE_RADIX_BITS    = CtaHybridPassPolicy::CtaUpsweepPassPolicyT::RADIX_BITS,
        LARGE_RADIX_DIGITS  = 1 << RADIX_BITS,

        SMALL_RADIX_BITS    = CtaHybridPassPolicy::CtaUpsweepPassPolicyT::RADIX_BITS,
        SMALL_RADIX_DIGITS  = 1 << RADIX_BITS,
*/
        WARP_THREADS        = cub::DeviceProps::WARP_THREADS,

//        SINGLE_TILE_ITEMS   = CtaHybridPassPolicy::CtaSingleTilePolicyT::TILE_ITEMS,
    };

/*
    // CTA upsweep abstraction
    typedef CtaUpsweepPass<
        typename CtaHybridPassPolicy::CtaUpsweepPassPolicyT,
        KeyType,
        SizeT> CtaUpsweepPassT;

    // CTA downsweep abstraction
    typedef CtaDownsweepPass<
        typename CtaHybridPassPolicy::CtaDownsweepPassPolicyT,
        KeyType,
        ValueType,
        SizeT> CtaDownsweepPassT;

    // CTA single-tile abstraction
    typedef CtaSingleTile<
        typename CtaHybridPassPolicy::CtaSingleTilePolicyT,
        KeyType,
        ValueType> CtaSingleTileT;

    // CTA single-tile abstraction
    typedef CtaSingleTile<
        typename CtaHybridPassPolicy::CtaSingleTilePolicyT1,
        KeyType,
        ValueType> CtaSingleTileT1;

    // Warp scan abstraction
    typedef cub::WarpScan<
        SizeT,
        1,
        RADIX_DIGITS> WarpScanT;

    // CTA scan abstraction
    typedef cub::CtaScan<
        SizeT,
        CTA_THREADS> CtaScanT;
*/

    /**
     * Shared memory storage layout
     */
    struct SmemStorage
    {
/*
        union
        {
            typename CtaUpsweepPassT::SmemStorage       upsweep_storage;
            typename CtaDownsweepPassT::SmemStorage     downsweep_storage;
            typename CtaSingleTileT::SmemStorage        single_storage;
            typename WarpScanT::SmemStorage             warp_scan_storage;
            typename CtaScanT::SmemStorage              cta_scan_storage;
        };

        volatile SizeT                       shared_bin_count[RADIX_DIGITS];
        volatile bool                        shared_bin_active[RADIX_DIGITS];
        unsigned int                         current_bit;
        unsigned int                         bits_remaining;
        SizeT                                num_elements;
        SizeT                                cta_offset;
        volatile unsigned int                enqueue_offset;
        unsigned int                         queue_descriptors;
*/

        BinDescriptor   input_descriptor;
    };

    // Shared data structures
    __shared__ SmemStorage smem_storage;

    // Get length of global queue
    int global_queue_length;
    params.q_global[0].FillSize(global_queue_length);

    // Process queued bins from global queue
    for (int bin = 0; bin < global_queue_length; ++bin)
    {
        // Get bin descriptor
        if (threadIdx.x == 0)
        {
            smem_storage.input_descriptor = params.d_global[0][i];
        }

        __syncthreads();

        if (smem_storage.input_descriptor.bin_state == UPSWEEP)
        {
            // Upsweep pass
        }
        else if (smem_storage.input_descriptor.bin_state == SCAN)
        {
            // Scan pass
        }
        else
        {
            // Downsweep pass
        }

        // Prepare global queue counters for next round
        if ((blockIdx.x == 0) && (threadIdx.x == 0))
        {
            params.q_global[0].PrepareFill();   // Prepare drain counter for filling next round
            params.q_global[0].PrepareDrain();  // Prepare filling counter for draining next round
        }
    }

    // Get length of global queue
    int local_queue_length;
    params.q_local[0].FillSize(local_queue_length);

    // Process queued bins from local queue
    while (true)
    {
        // Steal a bin
        if (threadIdx.x == 0)
        {
            smem_storage.input_descriptor.bin_state = INVALID;
            int descriptor_offset = params.q_local[0].Drain(1);
            if (descriptor_offset < local_queue_length)
            {
                smem_storage.input_descriptor = params.d_local[descriptor_offset];
            }
        }

        __syncthreads();

        // Quit if done
        if (smem_storage.input_descriptor.bin_state == INVALID) break;


    }










/*

    if (threadIdx.x == 0)
    {
        queue_descriptors = d_queue_counters[(iteration - 1) & 3];

        if (blockIdx.x == 0)
        {
            // Reset next work steal counter
            d_steal_counters[(iteration + 1) & 1] = 0;

            // Reset next queue counter
            d_queue_counters[(iteration + 1) & 3] = 0;
        }
    }

    while (true)
    {
        // Retrieve work
        if (threadIdx.x == 0)
        {
            unsigned int descriptor_offset = atomicAdd(d_steal_counters + (iteration & 1), 1);

            num_elements = 0;

            if (descriptor_offset < queue_descriptors)
            {
                BinDescriptor bin             = d_bins_in[descriptor_offset];
                current_bit                 = bin.current_bit - RADIX_BITS;
                bits_remaining                 = bin.current_bit - low_bit;
                num_elements                 = bin.num_elements;
                cta_offset                    = bin.offset;
            }
        }

        __syncthreads();

        if (num_elements <= 1)
        {
            return;
        }
        else if (num_elements <= SINGLE_TILE_ITEMS)
        {
            // Sort input tile
            CtaSingleTileT::Sort(
                smem_storage.single_storage,
                d_keys_in + cta_offset,
                d_keys_final + cta_offset,
                d_values_in + cta_offset,
                d_values_final + cta_offset,
                low_bit,
                bits_remaining,
                num_elements);
        }
        else
        {
            SizeT bin_count = 0;
            SizeT bin_prefix = 0;

            // Compute bin-count for each radix digit (valid in tid < RADIX_DIGITS)
            CtaUpsweepPassT::UpsweepPass(
                smem_storage.upsweep_storage,
                d_keys_in + cta_offset,
                current_bit,
                num_elements,
                bin_count);

            // Scan bin counts and output new partitions for next pass
            if (threadIdx.x < RADIX_DIGITS)
            {
                // Warp prefix sum
                WarpScanT::ExclusiveSum(
                    smem_storage.warp_scan_storage,
                    bin_count,
                    bin_prefix);

                unsigned int     my_current_bit         = current_bit;
                bool             bin_active             = true;

                // Update shared state
                shared_bin_count[threadIdx.x]         = bin_count;
                shared_bin_active[threadIdx.x]         = bin_active;

                // Progressively coalesce peer bins having stride 1, 2, 4, 8, etc.
                #pragma unroll
                for (int BIT = 0; BIT < RADIX_BITS - 1; BIT++)
                {
                    // Attempt to merge bins whose indices are a multiple of current stride
                    const int PEER_STRIDE = 1 << BIT;
                    const int MASK = (1 << (BIT + 1)) - 1;
                    if (bin_active && (threadIdx.x & MASK) == 0)
                    {
                        SizeT merged_bin_count = bin_count + shared_bin_count[threadIdx.x + PEER_STRIDE];

                        if (merged_bin_count < SINGLE_TILE_ITEMS)
                        {
                            // Coalesce peer bin
                            bin_count = merged_bin_count;
                            shared_bin_count[threadIdx.x + PEER_STRIDE] = 0;
                            my_current_bit++;

                            // I am still active if the peer was still active
                            bin_active = shared_bin_active[threadIdx.x + PEER_STRIDE];
                        }
                        else
                        {
                            // I could not coalesce my peer; I am no longer active
                            bin_active = false;
                        }

                        // Share my state
                        shared_bin_count[threadIdx.x] = bin_count;
                        shared_bin_active[threadIdx.x] = bin_active;
                    }

                }

                // Recover my state
                if (shared_bin_active[threadIdx.x])
                {
                    bin_count = shared_bin_count[threadIdx.x];
                }


                unsigned int active_bins_vote         = __ballot(bin_count > 0);
                unsigned int thread_mask             = (1 << threadIdx.x) - 1;
                unsigned int active_bins             = __popc(active_bins_vote);
                unsigned int active_bins_prefix     = __popc(active_bins_vote & thread_mask);

                // Increment enqueue offset
                if (threadIdx.x == 0)
                {
                    enqueue_offset = atomicAdd(d_queue_counters + (iteration & 3), active_bins);
                }

                // Output bin
                if (bin_count > 0)
                {
                    BinDescriptor bin(
                        cta_offset + bin_prefix,
                        bin_count,
                        my_current_bit);

                    d_bins_out[enqueue_offset + active_bins_prefix] = bin;
                }
            }

            // Distribute keys
            CtaDownsweepPassT::DownsweepPass(
                smem_storage.downsweep_storage,
                bin_prefix,
                d_keys_in + cta_offset,
                d_keys_out + cta_offset,
                d_values_in + cta_offset,
                d_values_out + cta_offset,
                current_bit,
                num_elements);
        }
    }
*/
}


/**
 * Hybrid kernel props
 */
template <
    typename KeyType,
    typename ValueType,
    typename SizeT>
struct HybridKernelProps : cub::KernelProps
{
    // Kernel function type
    typedef void (*KernelFunc)(
        cub::GridQueue<SizeT>   q_drain,
        cub::GridQueue<SizeT>   q_fill,
        BinDescriptor           *d_drain,
        BinDescriptor           *d_fill,
        KeyType                 *d_keys0,
        KeyType                 *d_keys1,
        KeyType                 *d_keys_final,
        ValueType               *d_values0,
        ValueType               *d_values1,
        ValueType               *d_values_final,
        unsigned int            global_cutoff);

    // Fields
    KernelFunc                  kernel_func;
    cudaSharedMemConfig         sm_bank_config;

    /**
     * Initializer
     */
    template <
        typename CtaHybridPassPolicy,
        typename OpaqueCtaHybridPassPolicy,
        int MIN_CTA_OCCUPANCY>
    cudaError_t Init(const cub::CudaProps &cuda_props)    // CUDA properties for a specific device
    {
        // Initialize fields
        kernel_func             = HybridKernel<OpaqueCtaHybridPassPolicy, MIN_CTA_OCCUPANCY, SizeT>;
        sm_bank_config          = CtaHybridPassPolicy::SMEM_CONFIG;

        // Initialize super class
        return cub::KernelProps::Init(
            kernel_func,
            CtaHybridPassPolicy::CTA_THREADS,
            cuda_props);
    }

    /**
     * Initializer
     */
    template <typename CtaHybridPassPolicy>
    cudaError_t Init(const cub::CudaProps &cuda_props)    // CUDA properties for a specific device
    {
        return Init<CtaHybridPassPolicy, CtaHybridPassPolicy>(cuda_props);
    }

};


} // namespace radix_sort
} // namespace back40
BACK40_NS_POSTFIX
