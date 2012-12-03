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
 * Experimental CSR implementation of SPMV
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <iterator>
#include <stdio.h>
#include <test_util.h>

#include "../cub.cuh"

using namespace cub;

//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

bool g_verbose = false;


//---------------------------------------------------------------------
// Types and constants
//---------------------------------------------------------------------

/// Pairing of dot product partial sums and corresponding row-id
struct PartialSum
{
    float   partial;        /// PartialSum sum
    int     row;            /// Row-id

    /// Default Constructor
    PartialSum() {}

    /// Constructor
    PartialSum(float partial, int row) : partial(partial), row(flag) {}

    /// Tags indicating this structure provides overloaded ThreadLoad and ThreadStore operations
    typedef void ThreadLoadTag;
    typedef void ThreadStoreTag;

    /// ThreadLoad (simply defer to loading individual items)
    template <PtxLoadModifier MODIFIER>
    __device__ __forceinline__
    void ThreadLoad(PartialSum *ptr)
    {
        partial = ThreadLoad<MODIFIER>(&(ptr->partial));
        row = ThreadLoad<MODIFIER>(&(ptr->row));
    }

     /// ThreadStore (simply defer to storing individual items)
    template <PtxStoreModifier MODIFIER>
    __device__ __forceinline__ void ThreadStore(PartialSum *ptr) const
    {
        // Always write partial first
        ThreadStore<MODIFIER>(&(ptr->partial), partial);
        ThreadStore<MODIFIER>(&(ptr->row), row);
    }
};

/// Reduce-by-row scan operator
struct ScanOp
{
    __device__ __forceinline__ PartialSum operator()(
        const PartialSum &first,
        const PartialSum &second)
    {
        return PartialSum(
            (second.row != first.row) ?
                second.partial :
                first.partial + second.partial,
            second.row);
    }
};


/// Returns true if row_b is the start of a new row
struct NewRowOp
{
    bool operator()(const int &row_a, const int &row_b)
    {
        return (row_a != row_b);
    }
};


//---------------------------------------------------------------------
// GPU kernels
//---------------------------------------------------------------------


/**
 * COO SpMV kernel
 */
template <
    int     CTA_THREADS,
    int     ITEMS_PER_THREAD>
__launch_bounds__ (CTA_THREADS)
__global__ void Kernel(
    PartialSum*     d_cta_aggregates,
    int*            d_rows,
    int*            d_columns,
    float*          d_values,
    int             num_vertices,
    float*          d_vector,
    float*          d_output,
    )
{
    //---------------------------------------------------------------------
    // Types and constants
    //---------------------------------------------------------------------

    // Head flag type
    typedef int HeadFlag;

    // Parameterize cooperative CUB types for use in the current problem context
    typedef CtaScan<int, CTA_THREADS>                   CtaScan;
    typedef CtaExchange<PartialSum, CTA_THREADS>        CtaExchange;
    typedef CtaDiscontinuity<HeadFlag, CTA_THREADS>     CtaDiscontinuity;

    // Shared memory type for this CTA (the union of the smem required for CUB primitives)
    union SmemStorage
    {
        typename CtaScan::SmemStorage           scan;
        typename CtaExchange::SmemStorage       exchange;
        typename CtaDiscontinuity::SmemStorage  discontinuity;
    };


    //---------------------------------------------------------------------
    // Kernel body
    //---------------------------------------------------------------------

    // Declare shared items
    __shared__ SmemStorage  s_storage;       // Shared storage needed for CUB primitives
    __shared__ PartialSum   s_prev_aggregate;          // Aggregate from previous CTA

    int                     columns[ITEMS_PER_THREAD];
    int                     rows[ITEMS_PER_THREAD];
    float                   values[ITEMS_PER_THREAD];
    PartialSum              partial_sums[ITEMS_PER_THREAD];
    HeadFlag                head_flags[ITEMS_PER_THREAD];

    // The CTA's offset in d_columns and d_values
    int cta_offset = blockIdx.x * CTA_THREADS * ITEMS_PER_THREAD;

    // Load a CTA-striped tile of A (sparse row-ids, column-ids, and values)
    CtaLoadDirectStriped(rows, d_rows, cta_offset);
    CtaLoadDirectStriped(columns, d_columns, cta_offset);
    CtaLoadDirectStriped(values, d_values, cta_offset);

    // Fence to prevent hoisting any dependent code below into the loads above
    threadfence_block();

    // Load the referenced values from x and compute dot product partial_sums
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
    {
        partial_sums[ITEM].partial = values[ITEM] * d_vector[columns[ITEM]];
        partial_sums[ITEM].row = rows[ITEM];
    }

    // Transpose from CTA-striped to CTA-blocked arrangement
    CtaExchange::StripedToBlocked(s_storage.exchange, partial_sums);

    // Barrier for smem reuse
    __syncthreads();

    // Save rows separately.  We will use them to compute the row head flags
    // later.  (After the exclusive scan, the row fields in partial_sums will
    // be shifted by one element.)
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
    {
        rows[ITEM] = partial_sums[ITEM].row;
    }

    // Compute exclusive scan of partial_sums
    ScanOp          scan_op;                // Reduce-by-row scan operator
    PartialSum      aggregate;              // CTA-wide aggregate in thread0
    PartialSum      identity(0.0, -1);      // Identity partial sum

    CtaScan::ExclusiveScan(
        s_storage.scan,
        partial_sums,
        partial_sums,           // Out
        identity,
        scan_op,
        aggregate);             // Out

    // Thread0 communicates the CTA-wide aggregate with other CTAs
    if (threadIdx.x == 0)
    {
        // Write CTA-wide aggregate to global list
        ThreadStore<PTX_STORE_CG>(d_cta_aggregates + blockIdx.x, aggregate);

        // Get aggregate from prior CTA
        PartialSum prev_aggregate;
        if (blockIdx.x == 0)
        {
            // First CTA has no prior aggregate
            prev_aggregate = identity;
        }
        else
        {
            // Keep loading prior CTA's aggregate until valid
            do
            {
                prev_aggregate = ThreadLoad<PTX_LOAD_CG>(d_cta_aggregates + blockIdx.x - 1);
            }
            while (prev_aggregate.row < 0);
        }

        // Share prev_aggregate with other threads
        s_prev_aggregate = prev_aggregate;
    }

    // Barrier for smem reuse and coherence
    __syncthreads();

    // Incorporate previous CTA's aggregate into partial_sums
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
    {
        partial_sums[ITEM] = scan_op(s_prev_aggregate, partial_sums[ITEM]);
    }

    // First partial in first thread should be prev_aggregate (currently is identity)
    if (threadIdx.x == 0)
    {
        partial_sums[0] = s_prev_aggregate;
    }

    // Flag row heads using saved row ids
    int last_row;
    CtaDiscontinuity::Flag(
        s_storage.discontinuity,
        rows,                           // Original row ids
        s_prev_aggregate.row,           // Last row id from previous CTA
        NewRowOp(),
        head_flags,                     // (out) Head flags
        last_row);                      // (out) The last row_id in this CTA (discard)

    // Scatter dot products if a row head of a valid row
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
    {
        if (head_flags[ITEM] && (partial_sums[ITEM].row > 0))
        {
            d_output[partial_sums[ITEM].row] = partial_sums[ITEM].partial;
        }
    }

    // Last CTA scatters the final value (if it has a valid row id)
    if ((blockIdx.x == gridDim.x - 1) && (threadIdx.x == 0) && (aggregate.row > 0))
    {
        d_output[aggregate.row] = aggregate.partial;
    }
}


//---------------------------------------------------------------------
// Host subroutines
//---------------------------------------------------------------------



/**
 * Main
 */
int main(int argc, char** argv)
{
    // Initialize command line
    CommandLineArgs args(argc, argv);
    g_verbose = args.CheckCmdLineFlag("v");

    // Print usage
    if (args.CheckCmdLineFlag("help"))
    {
        printf("%s "
            "[--device=<device-id>] "
            "[--v] "
            "\n", argv[0]);
        exit(0);
    }

    // Initialize device
    CubDebugExit(args.DeviceInit());


    return 0;
}



