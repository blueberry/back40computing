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
#include <vector>
#include <stdio.h>
#include <test_util.h>

#include "../cub.cuh"

using namespace cub;
using namespace std;

//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

bool g_verbose = false;


//---------------------------------------------------------------------
// Graph building types and utilities
//---------------------------------------------------------------------

/**
 * COO sparse format edge.  (A COO graph is just a list/array/vector of these.)
 */
template<typename _VertexId, typename _Value>
struct CooTuple
{
    typedef _VertexId   VertexId;
    typedef _Value      Value;

    VertexId            row;
    VertexId            col;
    Value               val;

    CooTuple(VertexId row, VertexId col) : row(row), col(col) {}
    CooTuple(VertexId row, VertexId col, Value val) : row(row), col(col), val(val) {}
};

/**
 * Comparator for sorting COO sparse format edges
 */
template<typename CooTuple>
bool DimacsTupleCompare (
    CooTuple elem1,
    CooTuple elem2)
{
    if (elem1.row < elem2.row) {
        // Sort edges by source node (to make rows)
        return true;
    } else if ((elem1.row == elem2.row) && (elem1.col < elem2.col)) {
        // Sort edgelists as well for coherence
        return true;
    }

    return false;
}

/**
 * Builds a square 3D grid COO sparse graph.  Interior nodes have degree 7 (including
 * a self-loop).  Values are unintialized, tuples are sorted.
 */
template<typename CooTuple>
void BuildGrid3dGraph(
    int width,
    vector<CooTuple> &tuples)
{
    typedef typename CooTuple::VertexId VertexId;

    VertexId interior_nodes        = (width - 2) * (width - 2) * (width - 2);
    VertexId face_nodes            = (width - 2) * (width - 2) * 6;
    VertexId edge_nodes            = (width - 2) * 12;
    VertexId corner_nodes          = 8;
    VertexId nodes                 = width * width * width;
    VertexId edges                 = (interior_nodes * 6) + (face_nodes * 5) + (edge_nodes * 4) + (corner_nodes * 3) + nodes;

    tuples.reserve(edges);

    int total = 0;
    for (VertexId i = 0; i < width; i++) {
        for (VertexId j = 0; j < width; j++) {
            for (VertexId k = 0; k < width; k++) {

                VertexId me = (i * width * width) + (j * width) + k;

                VertexId neighbor = (i * width * width) + (j * width) + (k - 1);
                if (k - 1 >= 0) {
                    tuples.push_back(CooTuple(me, neighbor));
                }

                neighbor = (i * width * width) + (j * width) + (k + 1);
                if (k + 1 < width) {
                    tuples.push_back(CooTuple(me, neighbor));
                }

                neighbor = (i * width * width) + ((j - 1) * width) + k;
                if (j - 1 >= 0) {
                    tuples.push_back(CooTuple(me, neighbor));
                }

                neighbor = (i * width * width) + ((j + 1) * width) + k;
                if (j + 1 < width) {
                    tuples.push_back(CooTuple(me, neighbor));
                }

                neighbor = ((i - 1) * width * width) + (j * width) + k;
                if (i - 1 >= 0) {
                    tuples.push_back(CooTuple(me, neighbor));
                }

                neighbor = ((i + 1) * width * width) + (j * width) + k;
                if (i + 1 < width) {
                    tuples.push_back(CooTuple(me, neighbor));
                }

                neighbor = me;
                tuples.push_back(CooTuple(me, neighbor));
            }
        }
    }
}



//---------------------------------------------------------------------
// Kernel types and constants
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
    int             num_rows,
    int             num_cols,
    int             num_vertices,
    float*          d_vector,
    float*          d_output)
{
    //---------------------------------------------------------------------
    // Types and constants
    //---------------------------------------------------------------------

    /// Constants
    enum
    {
        TILE_ITEMS = CTA_THREADS * ITEMS_PER_THREAD,
    };

    /// Head flag type
    typedef int HeadFlag;

    /// Parameterize cooperative CUB types for use in the current problem context
    typedef CtaScan<int, CTA_THREADS>                   CtaScan;
    typedef CtaExchange<PartialSum, CTA_THREADS>        CtaExchange;
    typedef CtaDiscontinuity<HeadFlag, CTA_THREADS>     CtaDiscontinuity;

    /// Shared memory type for this CTA
    struct SmemStorage
    {
        union
        {
            typename CtaScan::SmemStorage           scan;               // Smem needed for reduce-value-by-row scan
            typename CtaExchange::SmemStorage       exchange;           // Smem needed for striped->blocked transpose
            typename CtaDiscontinuity::SmemStorage  discontinuity;      // Smem needed for head-flagging
        };

        PartialSum                                  prev_aggregate;     // Aggregate from previous CTA
    };


    //---------------------------------------------------------------------
    // Kernel body
    //---------------------------------------------------------------------

    // Declare shared items
    __shared__ SmemStorage  s_storage;       // Shared storage needed for CUB primitives

    int         columns[ITEMS_PER_THREAD];
    int         rows[ITEMS_PER_THREAD];
    float       values[ITEMS_PER_THREAD];
    PartialSum  partial_sums[ITEMS_PER_THREAD];
    HeadFlag    head_flags[ITEMS_PER_THREAD];

    int         cta_offset      = blockIdx.x * TILE_ITEMS;              // The CTA's offset in d_columns and d_values
    int         guarded_items   = (blockIdx.x == gridDim.x - 1) ?       // The number of guarded items in the last tile
                                    num_vertices % TILE_ITEMS :
                                    0;

    // Load a CTA-striped tile of A (sparse row-ids, column-ids, and values)
    if (guarded_items)
    {
        // Last tile has guarded loads.  Extend the coordinates of the last
        // vertex for out-of-bound items, but zero-valued
        int last_row = d_rows[num_vertices - 1];
        int last_column = d_columns[num_vertices - 1];

        CtaLoadDirectStriped(rows, d_rows, cta_offset, guarded_items, last_row);
        CtaLoadDirectStriped(columns, d_columns, cta_offset, guarded_items, last_column);
        CtaLoadDirectStriped(values, d_values, cta_offset, guarded_items, 0.0);
    }
    else
    {
        // Unguarded loads
        CtaLoadDirectStriped(rows, d_rows, cta_offset);
        CtaLoadDirectStriped(columns, d_columns, cta_offset);
        CtaLoadDirectStriped(values, d_values, cta_offset);
    }

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

    // Save a copy of the original rows.  We will use them to compute the row head flags
    // later.  (After the exclusive scan, the row fields in partial_sums will
    // be shifted by one element.)
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
    {
        rows[ITEM] = partial_sums[ITEM].row;
    }

    // Compute exclusive scan of partial_sums
    ScanOp          scan_op;                                // Reduce-by-row scan operator
    PartialSum      aggregate;                              // CTA-wide aggregate in thread0
    PartialSum      identity(0.0, d_rows[cta_offset]);      // Zero-valued identity (with row-id of first item)

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
        s_storage.prev_aggregate = prev_aggregate;
    }

    // Barrier for smem reuse and coherence
    __syncthreads();

    // Incorporate previous CTA's aggregate into partial_sums
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
    {
        partial_sums[ITEM] = scan_op(s_storage.prev_aggregate, partial_sums[ITEM]);
    }

    // Flag row heads using saved row ids
    CtaDiscontinuity::Flag(
        s_storage.discontinuity,
        rows,                           // Original row ids
        s_storage.prev_aggregate.row,   // Last row id from previous CTA
        NewRowOp(),
        head_flags);                    // (out) Head flags

    // Scatter dot products if a row head of a valid row
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
    {
        if (head_flags[ITEM] && (partial_sums[ITEM].row > 0))
        {
            d_output[partial_sums[ITEM].row] = partial_sums[ITEM].partial;
        }
    }

    // Last tile scatters the final value (if it has a valid row id), which is the aggregate
    if ((blockIdx.x == gridDim.x - 1) && (threadIdx.x == 0) && (aggregate.row > 0))
    {
        d_output[aggregate.row] = aggregate.partial;
    }
}


//---------------------------------------------------------------------
// Host subroutines
//---------------------------------------------------------------------


/**
 * Simple test of device
 */
template <int CTA_THREADS, int ITEMS_PER_THREAD>
void TestDevice(
    int*            h_rows,
    int*            h_columns,
    float*          h_values,
    int             num_rows,
    int             num_cols,
    int             num_vertices,
    float*          h_vector,
    float*          h_output,
    int             iterations)
{
    if (iterations <= 0) return;

    const int TILE_SIZE = CTA_THREADS * ITEMS_PER_THREAD;

    int*            d_rows;
    int*            d_columns;
    float*          d_values;
    float*          d_vector;
    float*          d_output;

    // Allocate device arrays
    CachedAllocator *allocator = CubCachedAllocator();
    CubDebugExit(allocator->Allocate((void**)&d_rows, sizeof(int) * num_vertices));
    CubDebugExit(allocator->Allocate((void**)&d_columns, sizeof(int) * num_vertices));
    CubDebugExit(allocator->Allocate((void**)&d_values, sizeof(float) * num_vertices));
    CubDebugExit(allocator->Allocate((void**)&d_vector, sizeof(float) * num_cols));
    CubDebugExit(allocator->Allocate((void**)&d_output, sizeof(float) * num_rows));

    // Copy host arrays to device
    CubDebugExit(cudaMemcpy(d_rows, h_rows, sizeof(int) * num_vertices, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemcpy(d_columns, h_columns, sizeof(int) * num_vertices, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemcpy(d_values, h_values, sizeof(float) * num_vertices, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemcpy(d_vector, h_vector, sizeof(float) * num_cols, cudaMemcpyHostToDevice));

    // Zero-out the output array
    CubDebugExit(cudaMemset(d_output, 0, sizeof(float) * num_rows));

    // Run kernel
    int grid_size = (num_vertices + TILE_SIZE - 1) / TILE_SIZE;

    GpuTimer gpu_timer;
    float elapsed_millis = 0;
    for (int i = 0; i < iterations; i++)
    {
        gpu_timer.Start();

        Kernel<CTA_THREADS, ITEMS_PER_THREAD><<<grid_size, CTA_THERADS>>>(
            d_cta_aggregates,
            d_rows,
            d_columns,
            d_values,
            num_rows,
            num_cols,
            num_vertices,
            d_vector,
            d_output);

        gpu_timer.Stop();
        elapsed_millis = gpu_timer.ElapsedMillis();
    }

    // Display timing
    float avg_elapsed = elapsed_millis / iterations;
    int total_bytes = ((sizeof(int) + sizeof(int) + sizeof(float)) * num_vertices) + (sizeof(float) * 2 * num_rows);
    printf("Average elapsed (%.3f ms), utilized bandwidth (%.3f GB/s), GFLOPS(%.3f)\n",
        avg_elapsed,
        total_bytes / avg_elapsed / 1000.0 / 1000.0,
        num_vertices * 2 / avg_elapsed / 1000.0 / 1000.0);

    // Check results
    AssertEquals(0, CompareDeviceResults(h_output, d_output, num_rows, g_verbose, g_verbose));

    // Cleanup
    CubDebugExit(allocator->Deallocate(d_rows));
    CubDebugExit(allocator->Deallocate(d_columns));
    CubDebugExit(allocator->Deallocate(d_columns));
    CubDebugExit(allocator->Deallocate(d_vector));
    CubDebugExit(allocator->Deallocate(d_output));
}


/**
 * Simple test of device
 */
void ComputeReference(
    int*            h_rows,
    int*            h_columns,
    float*          h_values,
    int             num_rows,
    int             num_cols,
    int             num_vertices,
    float*          h_vector,
    float*          h_output)
{
    for (int i = 0; i < num_rows; i++)
    {
        h_output[i] = 0.0;
    }

    for (int i = 0; i < num_vertices; i++)
    {
        h_output[h_rows[i]] += h_values[i] * h_vector[columns[i]];
    }
}


/**
 *
 */
void GenerateProblem(
    int*            h_rows,
    int*            h_columns,
    float*          h_values,
    int             num_rows,
    int             num_cols,
    int             num_vertices,
    float*          h_vector)
{



}



/**
 * Main
 */
int main(int argc, char** argv)
{
    typedef CooTuple<int, float> CooTuple;

    // Initialize command line
    CommandLineArgs args(argc, argv);
    g_verbose = args.CheckCmdLineFlag("v");

    // Print usage
    if (args.CheckCmdLineFlag("help"))
    {
        printf("%s\n [--device=<device-id>] [--v]\n"
            "\t--type=grid2d --width=<width>\n"
            "\t--type=grid3d --width=<width>\n"
            "\t--type=metis --file=<file>\n"
            "\n", argv[0]);
        exit(0);
    }

    // Initialize device
    CubDebugExit(args.DeviceInit());

    // Get graph type
    string type;
    args.GetCmdLineArgument("type", type);

    // Generate problem
    vector<CooTuple> tuples;
    if (type == string("grid3d"))
    {
        int width;
        args.GetCmdLineArgument("width", width);
        BuildGrid3dGraph(width, tuples);
    }
    else
    {
        printf("Unsupported graph type\n");
        exit(1);
    }


    // Force any kernel stdio to screen
    CubDebugExit(cudaThreadSynchronize());

    return 0;
}



