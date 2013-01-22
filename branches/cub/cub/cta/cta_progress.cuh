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
 * Description of work distribution amongst CTAs
 */
template <
    typename    SizeT,
    bool        WORK_STEALING = false>
class CtaProgress
{
private:

    // Grid-specific fields
    SizeT   total_items;
    SizeT   total_grains;
    int     grid_size;
    int     big_ctas;
    SizeT   big_share;
    SizeT   normal_share;
    SizeT   normal_base_offset;

    SizeT   *d_steal_counter;

    /// Shared memory storage layout type
    struct SmemStorage
    {
        SizeT   cta_offset;
        SizeT   cta_oob;
    };

    SmemStorage &smem_storage;

public:

    /// The operations exposed by CtaScan require shared memory of this
    /// type.  This opaque storage can be allocated directly using the
    /// <tt>__shared__</tt> keyword.  Alternatively, it can be aliased to
    /// externally allocated shared memory or <tt>union</tt>'d with other types
    /// to facilitate shared memory reuse.
    typedef SmemStorage SmemStorage;

public:

    /**
     * Constructor.
     *
     * Generally constructed in host code.
     */
    __host__ __device__ __forceinline__ CtaProgress(
        SizeT   total_items,
        int     grid_size,
        int     schedule_granularity,
        SizeT   *d_steal_counter = NULL) :
            // initializers
            total_items(total_items),
            grid_size(grid_size),
            cta_offset(0),
            cta_oob(0),
            d_steal_counter(d_steal_counter)
    {
        if (!WORK_STEALING)
        {
            total_grains            = (total_items + schedule_granularity - 1) / schedule_granularity;
            SizeT grains_per_cta    = total_grains / grid_size;
            big_ctas                = total_grains - (grains_per_cta * grid_size);        // leftover grains go to big blocks
            normal_share            = grains_per_cta * schedule_granularity;
            normal_base_offset      = big_ctas * schedule_granularity;
            big_share               = normal_share + schedule_granularity;
        }
    }



    /**
     * Initializer.
     *
     * Generally initialized by each CTA after construction on the host.
     */
    __device__ __forceinline__ void Init(SmemStorage &smem_storage)
    {
        if (!WORK_STEALING)
        {
            if (threadIdx.x == 0)
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
        if (!WORK_STEALING)
        {
            return cta_oob - cta_offset;
        }
        else
        {
            // TODO
            return 0;
        }
    }


    /**
     *
     */
    __device__ __forceinline__ bool NextFull(
        SmemStorage     &smem_storage,      ///< [in] Shared reference to opaque SmemStorage layout
        int             tile_size,
        SizeT           &cta_offset)
    {
        if (WORK_STEALING)
        {
            if (threadIdx.x == 0)
            {
                smem_storage.cta_offset = atomicAdd(d_steal_counter, (SizeT) tile_size);
            }

            __syncthreads();

            cta_offset = smem_storage.cta_offset;
            return (cta_offset + tile_size <= total_items);
        }
        else
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
    }


    /**
     *
     */
    __device__ __forceinline__ bool NextPartial(
        SizeT &cta_offset)
    {
        if (!WORK_STEALING)
        {
            cta_offset = this->cta_offset;
            return (cta_offset < cta_oob);
        }
        else
        {
            // TODO
            return false;
        }
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

