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
 * CTA Work management.
 *
 * A given CTA may receive one of three different amounts of
 * work: "big", "normal", and "last".  The big workloads are one
 * grain greater than the normal, and the last workload
 * does the extra work.
 *
 ******************************************************************************/

#pragma once

#include "../ns_wrapper.cuh"
#include "../macro_utils.cuh"
#include "../host/allocator.cuh"

CUB_NS_PREFIX
namespace cub {


/**
 * Queue abstraction.
 *
 * Filling works by atomically-incrementing a zero-initialized counter, giving the
 * offset for writing items.
 *
 * Draining works by atomically-incrementing a different zero-initialized counter until
 * the previous fill-size is exceeded.
 */
template <typename SizeT>
struct Queue
{
    enum
    {
        FILL    = 0,
        DRAIN   = 1,
    };

    SizeT *d_counters;
    SizeT fill_size;

    /// Constructor
    Queue() : d_counters(NULL), fill_size(0) {}

    /******************************************************************//**
     * \name Host implementation
     *********************************************************************/

    /// Allocate
    __host__ __forceinline__ cudaError_t Allocate()
    {
        return CubDebug(CubCachedAllocator<void>()->Allocate((void**)&d_counters, sizeof(SizeT) * 2));
    }

    /// Deallocate
    __host__ __forceinline__ cudaError_t Deallocate()
    {
        cudaError_t error = CubDebug(CubCachedAllocator<void>()->Deallocate(d_counters));
        d_counters = NULL;
        return error;
    }

    /// Initializes a full-queue
    __host__ __forceinline__ cudaError_t Init(SizeT fill_size)
    {
        this->fill_size = fill_size;
        SizeT counters[2] = { fill_size, 0};
        return cudaMemcpy(d_counters, counters, sizeof(SizeT) * 2, cudaMemcpyHostToDevice);
    }

    /// Initializes an empty queue
    __host__ __forceinline__ cudaError_t Init()
    {
        this->fill_size = 0;
        return cudaMemset(d_counters, 0, sizeof(SizeT) * 2);
    }

    /// Returns number of items filled
    __host__ __forceinline__ cudaError_t Size(SizeT &num_items)
    {
        return cudaMemcpy(&num_items, d_counters + FILL, sizeof(SizeT), cudaMemcpyDeviceToHost);
    }


    /******************************************************************//**
     * \name Device implementation
     *********************************************************************/

    /// Allocate
    __device__ __forceinline__ cudaError_t Allocate()
    {
        cudaError_t error = CubDebug(cudaMalloc(&d_counters,  sizeof(SizeT) * 2));
        d_counters = NULL;
        return error;
    }
    /// Deallocate
    __device__ __forceinline__ cudaError_t Deallocate()
    {
        return CubDebug(cudaFree(d_counters));
    }

    /// Initializer
    __device__ __forceinline__ cudaError_t Init(SizeT fill_size)
    {
        this->fill_size = fill_size;
        *d_counters = 0;
        return cudaSuccess;
    }

    /// Dequeue num_items items.  Returns offset from which to read items.
    __device__ __forceinline__ SizeT Dequeue(SizeT num_items)
    {
        return atomicAdd(d_counters, num_items);
    }

    /// Returns number of items previously filled.  (Only valid in kernels not actively filling.)
    __device__ __forceinline__ cudaError_t Size(SizeT &num_items)
    {
        num_items = *d_counters;
        return cudaSuccess;
    }

};


/**
 * Fill.  Works by atomically-incrementing a zero-initialized counter.
 */
template <typename SizeT>
struct Fill
{
    SizeT *d_counters;

    /// Constructor
    Fill() : d_counters(NULL) {}

    /******************************************************************//**
     * \name Host implementation
     *********************************************************************/

    /// Allocate
    __host__ __forceinline__ cudaError_t Allocate()
    {
        return CubDebug(CubCachedAllocator<void>()->Allocate((void**)&d_counters, sizeof(SizeT) * 2));
    }

    /// Deallocate
    __host__ __forceinline__ cudaError_t Deallocate()
    {
        cudaError_t error = CubDebug(CubCachedAllocator<void>()->Deallocate(d_counters));
        d_counters = NULL;
        return error;
    }

    /// Initializer
    __host__ __forceinline__ cudaError_t Init()
    {
        return cudaMemset(d_counters, 0, sizeof(SizeT) * 2);
    }



    /******************************************************************//**
     * \name Device implementation
     *********************************************************************/

    /// Allocate
    __device__ __forceinline__ cudaError_t Allocate()
    {
        cudaError_t error = CubDebug(cudaMalloc(&d_counters,  sizeof(SizeT) * 2));
        d_counters = NULL;
        return error;
    }
    /// Deallocate
    __device__ __forceinline__ cudaError_t Deallocate()
    {
        return CubDebug(cudaFree(d_counters));
    }

    /// Initializer
    __device__ __forceinline__ cudaError_t Init()
    {
        if ((blockIdx.x == 0) && (threadIdx.x == 0)) *d_counters = 0;
        return cudaSuccess;
    }

    /// Enqueue num_items items.  Returns offset from which to write items.
    __device__ __forceinline__ SizeT Enqueue(SizeT num_items)
    {
        return atomicAdd(d_counters, num_items);
    }

};



template <typename SizeT>
class Queue
{
    enum
    {
        ENQUEUE = 0,
        DEQUEUE = 1,
    };

    // Two counters
    SizeT *d_counters;

    Queue()
    {

    }

    __device__ __forceinline__

};


/**
 * Description of work distribution amongst CTAs
 */
template <typename SizeT>
class CtaProgress
{
private:

    SizeT   total_items;
    SizeT   total_grains;
    int     grid_size;
    int     big_ctas;
    SizeT   big_share;
    SizeT   normal_share;
    SizeT   normal_base_offset;

    // CTA-specific fields
    SizeT   cta_offset;
    SizeT   cta_oob;

public:

    /**
     * Constructor.
     *
     * Generally constructed in host code one time.
     */
    __host__ __device__ __forceinline__ CtaProgress(
        SizeT   total_items,
        int     grid_size,
        int     schedule_granularity) :
            // initializers
            total_items(total_items),
            grid_size(grid_size),
            cta_offset(0),
            cta_oob(0)
    {
        total_grains            = (total_items + schedule_granularity - 1) / schedule_granularity;
        SizeT grains_per_cta    = total_grains / grid_size;
        big_ctas                = total_grains - (grains_per_cta * grid_size);        // leftover grains go to big blocks
        normal_share            = grains_per_cta * schedule_granularity;
        normal_base_offset      = big_ctas * schedule_granularity;
        big_share               = normal_share + schedule_granularity;
    }


    /**
     * Initializer.
     *
     * Generally initialized by each CTA after construction on the host.
     */
    __device__ __forceinline__ void Init()
    {
        if (blockIdx.x < big_ctas)
        {
            // This CTA gets a big share of grains (grains_per_cta + 1)
            cta_offset = (blockIdx.x * big_share);
            cta_oob = cta_offset + big_share;
        }
        else if (blockIdx.x < total_grains)
        {
            // This CTA gets a normal share of grains (grains_per_cta)
            cta_offset = normal_base_offset + (blockIdx.x * normal_share);
            cta_oob = cta_offset + normal_share;
        }

        // Last CTA
        if (blockIdx.x == grid_size - 1)
        {
            cta_oob = total_items;
        }
    }


    /**
     *
     */
    __device__ __forceinline__ SizeT TotalItems()
    {
        return total_items;
    }


    /**
     *
     */
    __device__ __forceinline__ SizeT GuardedItems()
    {
        return cta_oob - cta_offset;
    }


    /**
     *
     */
    __device__ __forceinline__ bool NextFull(
        int tile_size,
        SizeT &cta_offset)
    {
        if (this->cta_offset + tile_size <= cta_oob)
        {
            cta_offset = this->cta_offset;
            this->cta_offset += tile_size;
            return true;
        }
        else
        {
            return false;
        }
    }


    /**
     *
     */
    __device__ __forceinline__ bool NextPartial(
        SizeT &cta_offset)
    {
        cta_offset = this->cta_offset;
        return (cta_offset < cta_oob);
    }


    /**
     * Print to stdout
     */
    __host__ __device__ __forceinline__ void Print()
    {
        printf(
#ifdef __CUDA_ARCH__
            "\tCTA(%d) "
            "cta_offset(%lu) "
            "cta_oob(%lu) "
#endif
            "total_items(%lu)  "
            "big_ctas(%lu)  "
            "big_share(%lu)  "
            "normal_share(%lu)\n",
#ifdef __CUDA_ARCH__
                blockIdx.x,
                (unsigned long) cta_offset,
                (unsigned long) cta_oob,
#endif
                (unsigned long) total_items,
                (unsigned long) big_ctas,
                (unsigned long) big_share,
                (unsigned long) normal_share);
    }
};




} // namespace cub
CUB_NS_POSTFIX

