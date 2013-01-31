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
 * "Spine-scan" CTA abstraction for scanning radix digit histograms
 ******************************************************************************/

#pragma once

#include "../../cub/cub.cuh"
#include "../ns_wrapper.cuh"

BACK40_NS_PREFIX
namespace back40 {
namespace radix_sort {


//---------------------------------------------------------------------
// Tuning policy types
//---------------------------------------------------------------------


/**
 * Spine scan CTA tuning policy
 */
template <
    int                     _CTA_THREADS,           // The number of threads per CTA
    int                     _ITEMS_PER_THREAD,      // The number of consecutive keys to process per thread per global load
    int                     _TILE_STRIPS,           // The number of loads to process per thread per tile
    cub::PtxLoadModifier    _LOAD_MODIFIER,         // Load cache-modifier
    cub::PtxStoreModifier   _STORE_MODIFIER,        // Store cache-modifier
    cudaSharedMemConfig     _SMEM_CONFIG>           // Shared memory bank size
struct CtaScanPassPolicy
{
    enum
    {
        CTA_THREADS         = _CTA_THREADS,
        ITEMS_PER_THREAD    = _ITEMS_PER_THREAD,
        TILE_ITEMS          = CTA_THREADS * ITEMS_PER_THREAD,
    };

    static const cub::PtxLoadModifier   LOAD_MODIFIER   = _LOAD_MODIFIER;
    static const cub::PtxStoreModifier  STORE_MODIFIER  = _STORE_MODIFIER;
    static const cudaSharedMemConfig    SMEM_CONFIG     = _SMEM_CONFIG;
};


//---------------------------------------------------------------------
// CTA-wide abstractions
//---------------------------------------------------------------------


/**
 * CTA-wide abstraction for computing a prefix scan over a range of input tiles
 */
template <
    typename CtaScanPassPolicy,
    typename T>
class CtaScanPass
{
private:

    //---------------------------------------------------------------------
    // Constants and type definitions
    //---------------------------------------------------------------------

    enum
    {
        CTA_THREADS         = CtaScanPassPolicy::CTA_THREADS,
        ITEMS_PER_THREAD    = CtaScanPassPolicy::ITEMS_PER_THREAD,
        TILE_ITEMS          = CtaScanPassPolicy::TILE_ITEMS,
    };

    /// CtaScan utility type
    typedef cub::CtaScan<T, CTA_THREADS> CtaScan;

    /// Stateful callback functor to provide the running tile-prefix to CtaScan
    struct CtaPrefixOp
    {
        T prefix;

        /// Constructor
        __device__ __forceinline__ CtaPrefixOp() : prefix(0) {}

        /**
         * CTA-wide prefix callback functor called by thread-0 in CtaScan::ExclusiveScan().
         * Returns the CTA-wide prefix to apply to all scan inputs.
         */
        __device__ __forceinline__ T operator()(
            const T &local_aggregate)              ///< The aggregate sum of the local prefix sum inputs
        {
            T retval = prefix;
            prefix += local_aggregate;
            return retval;
        }
    };

public:

    /**
     * Shared memory storage layout
     */
    typedef typename CtaScan::SmemStorage SmemStorage;

private:

    //---------------------------------------------------------------------
    // Methods
    //---------------------------------------------------------------------

    /**
     * Process a single tile
     */
    template <typename SizeT>
    static __device__ __forceinline__ void ProcessTile(
        SmemStorage     &smem_storage,
        T               *d_in,
        T               *d_out,
        SizeT           cta_offset,
        CtaPrefixOp     &carry)
    {
        // Tile of scan elements
        T partials[ITEMS_PER_THREAD];

        // Load tile
        cub::CtaLoadVectorized<CtaScanPassPolicy::LOAD_MODIFIER>(
            partials, d_in, cta_offset);

        // Scan tile with carry in thread-0
        T aggregate;
        CtaScan::ExclusiveSum(smem_storage, partials, partials, aggregate, carry);

        // Store tile
        cub::CtaStoreVectorized<CtaScanPassPolicy::STORE_MODIFIER>(
            partials, d_in, cta_offset);
    }

public:

    //---------------------------------------------------------------------
    // Interface
    //---------------------------------------------------------------------

    /**
     * Scan a range of input tiles
     */
    template <typename SizeT>
    static __device__ __forceinline__ void ScanPass(
        SmemStorage     &smem_storage,
        T               *d_in,
        T               *d_out,
        SizeT           &num_elements)
    {
        // Running partial accumulated by the CTA over its tile-processing
        // lifetime (managed in each raking thread)
        CtaPrefixOp carry;

        SizeT cta_offset = 0;
        while (cta_offset + TILE_ITEMS <= num_elements)
        {
            // Process full tiles of tile_items
            ProcessTile(smem_storage, d_in, d_out, cta_offset, carry);

            cta_offset += TILE_ITEMS;
        }
    }

};


} // namespace radix_sort
} // namespace back40
BACK40_NS_POSTFIX
