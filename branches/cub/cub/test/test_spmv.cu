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
 * Experimental reduce-value-by-row COO implementation of SPMV
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
int g_iterations = 1;


//---------------------------------------------------------------------
// Graph building types and utilities
//---------------------------------------------------------------------

/**
 * COO graph type
 */
template<typename VertexId, typename Value>
struct CooGraph
{
    /**
     * COO edge tuple.  (A COO graph is just a list/array/vector of these.)
     */
    struct CooTuple
    {
        VertexId            row;
        VertexId            col;
        Value               val;

        CooTuple(VertexId row, VertexId col) : row(row), col(col) {}
        CooTuple(VertexId row, VertexId col, Value val) : row(row), col(col), val(val) {}
    };

    /**
     * Comparator for sorting COO sparse format edges
     */
    static bool CooTupleCompare (const CooTuple &elem1, const CooTuple &elem2)
    {
        if (elem1.row < elem2.row)
        {
            return true;
        }
        else if ((elem1.row == elem2.row) && (elem1.col < elem2.col))
        {
            return true;
        }

        return false;
    }

    int                                     row_dim;        // Num rows
    int                                     col_dim;        // Num cols
    vector<CooTuple<VertexId, Value> >      coo_tuples;     // Non-zero entries

    /**
     * Update graph dims based upon COO tuples
     */
    void UpdateDims()
    {
        row_dim = -1;
        col_dim = -1;

        for (int i = 0; i < coo_tuples.size(); i++)
        {
            row_dim = CUB_MAX(row_dim, coo_tuples[i].row);
            col_dim = CUB_MAX(col_dim, coo_tuples[i].col);
        }

        row_dim++;
        col_dim++;
    }


    /**
     * Builds a square 3D grid COO sparse graph.  Interior nodes have degree 7 (including
     * a self-loop).  Values are unintialized, coo_tuples are sorted.
     */
    void InitGrid3d(VertexId width)
    {
        VertexId interior_nodes        = (width - 2) * (width - 2) * (width - 2);
        VertexId face_nodes            = (width - 2) * (width - 2) * 6;
        VertexId edge_nodes            = (width - 2) * 12;
        VertexId corner_nodes          = 8;
        VertexId nodes                 = width * width * width;
        VertexId edges                 = (interior_nodes * 6) + (face_nodes * 5) + (edge_nodes * 4) + (corner_nodes * 3) + nodes;

        coo_tuples.clear();
        coo_tuples.resize(edges);

        for (VertexId i = 0; i < width; i++) {
            for (VertexId j = 0; j < width; j++) {
                for (VertexId k = 0; k < width; k++) {

                    VertexId me = (i * width * width) + (j * width) + k;

                    VertexId neighbor = (i * width * width) + (j * width) + (k - 1);
                    if (k - 1 >= 0) {
                        coo_tuples.push_back(CooTuple(me, neighbor));
                    }

                    neighbor = (i * width * width) + (j * width) + (k + 1);
                    if (k + 1 < width) {
                        coo_tuples.push_back(CooTuple(me, neighbor));
                    }

                    neighbor = (i * width * width) + ((j - 1) * width) + k;
                    if (j - 1 >= 0) {
                        coo_tuples.push_back(CooTuple(me, neighbor));
                    }

                    neighbor = (i * width * width) + ((j + 1) * width) + k;
                    if (j + 1 < width) {
                        coo_tuples.push_back(CooTuple(me, neighbor));
                    }

                    neighbor = ((i - 1) * width * width) + (j * width) + k;
                    if (i - 1 >= 0) {
                        coo_tuples.push_back(CooTuple(me, neighbor));
                    }

                    neighbor = ((i + 1) * width * width) + (j * width) + k;
                    if (i + 1 < width) {
                        coo_tuples.push_back(CooTuple(me, neighbor));
                    }

                    neighbor = me;
                    coo_tuples.push_back(CooTuple(me, neighbor));
                }
            }
        }

        // Sort by rows, then columns
        std::stable_sort(coo_tuples, coo_tuples + edges, CooTupleCompare<CooTuple>);

        UpdateDims();
    }
};



//---------------------------------------------------------------------
// Kernel types and constants
//---------------------------------------------------------------------

/// Pairing of dot product partial sums and corresponding row-id
template <typename VertexId, typename Value>
struct PartialSum
{
    Value       partial;        /// PartialSum sum
    VertexId    row;            /// Row-id

    /// Default Constructor
    PartialSum() {}

    /// Constructor
    PartialSum(Value partial, VertexId row) : partial(partial), row(flag) {}

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
template <typename VertexId>
struct NewRowOp
{
    bool operator()(const VertexId& row_a, const VertexId& row_b)
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
    int             CTA_THREADS,
    int             ITEMS_PER_THREAD,
    typename        VertexId,
    typename        Value>
__launch_bounds__ (CTA_THREADS)
__global__ void Kernel(
    int                             row_dim,
    int                             col_dim,
    int                             num_vertices,
    PartialSum<VertexId, Value>*    d_cta_aggregates,   // Temporary storage for communicating dot product partials between CTAs
    VertexId*                       d_rows,
    VertexId*                       d_columns,
    Value*                          d_values,
    Value*                          d_vector,
    Value*                          d_result)
{
    //---------------------------------------------------------------------
    // Types and constants
    //---------------------------------------------------------------------

    // Constants
    enum
    {
        TILE_ITEMS = CTA_THREADS * ITEMS_PER_THREAD,
    };

    // Head flag type
    typedef int                                         HeadFlag;

    // Dot product partial sum type
    typedef PartialSum<VertexId, Value>                 PartialSum;

    // Parameterize cooperative CUB types for use in the current problem context
    typedef CtaScan<PartialSum, CTA_THREADS>            CtaScan;
    typedef CtaExchange<PartialSum, CTA_THREADS>        CtaExchange;
    typedef CtaDiscontinuity<HeadFlag, CTA_THREADS>     CtaDiscontinuity;

    // Shared memory type for this CTA
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

    VertexId    columns[ITEMS_PER_THREAD];
    VertexId    rows[ITEMS_PER_THREAD];
    Value       values[ITEMS_PER_THREAD];
    PartialSum  partial_sums[ITEMS_PER_THREAD];
    HeadFlag    head_flags[ITEMS_PER_THREAD];

    // Figure out this CTA's tile of graph input
    int         cta_offset      = blockIdx.x * TILE_ITEMS;              // The CTA's offset in d_columns and d_values
    int         guarded_items   = (blockIdx.x == gridDim.x - 1) ?       // The number of guarded items in the last tile
                                        num_vertices % TILE_ITEMS :
                                        0;

    // Load a CTA-striped tile of A (sparse row-ids, column-ids, and values)
    if (guarded_items)
    {
        // Last tile has guarded loads.  Extend the coordinates of the last
        // vertex for out-of-bound items, but zero-valued
        VertexId last_row = d_rows[num_vertices - 1];
        VertexId last_column = d_columns[num_vertices - 1];

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

    // First item of first thread should be the previous CTA's aggregate (and not identity)
    if (threadIdx.x == 0)
    {
        partial_sums[0] = s_storage.prev_aggregate;
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
            d_result[partial_sums[ITEM].row] = partial_sums[ITEM].partial;
        }
    }

    // Last tile scatters the final value (if it has a valid row id), which is the aggregate
    if ((blockIdx.x == gridDim.x - 1) && (threadIdx.x == 0) && (aggregate.row > 0))
    {
        d_result[aggregate.row] = aggregate.partial;
    }
}


//---------------------------------------------------------------------
// Host subroutines
//---------------------------------------------------------------------


/**
 * Simple test of device
 */
template <
    int                         CTA_THREADS,
    int                         ITEMS_PER_THREAD,
    typename                    VertexId,
    typename                    Value>
void TestDevice(
    CooGraph<VertexId, Value>&  coo_graph,
    Value*                      h_vector,
    Value*                      h_reference)
{
    if (iterations <= 0) return;

    const int TILE_SIZE = CTA_THREADS * ITEMS_PER_THREAD;

    // SOA device storage
    VertexId*                       d_rows;             // SOA graph row coordinates
    VertexId*                       d_columns;          // SOA graph col coordinates
    Value*                          d_values;           // SOA graph values
    Value*                          d_vector;           // Vector multiplicand
    Value*                          d_result;           // Output row
    PartialSum<VertexId, Value>*    d_cta_aggregates;   // Temporary storage for communicating dot product partials between CTAs

    // Create SOA version of coo_graph on host
    int                             num_vertices    = coo_graph.coo_tuples.size();
    VertexId*                       h_rows          = new VertexId[num_vertices];
    VertexId*                       h_columns       = new VertexId[num_vertices];
    Value*                          h_values        = new Value[num_vertices];

    for (int i = 0; i < num_vertices; i++)
    {
        h_rows[i]       = coo_graph.coo_tuples[i].row;
        h_columns[i]    = coo_graph.coo_tuples[i].col;
        h_values[i]     = coo_graph.coo_tuples[i].val;
    }

    // Allocate COO device arrays
    CachedAllocator *allocator = CubCachedAllocator();
    CubDebugExit(allocator->Allocate((void**)&d_rows,       sizeof(VertexId) * num_vertices));
    CubDebugExit(allocator->Allocate((void**)&d_columns,    sizeof(VertexId) * num_vertices));
    CubDebugExit(allocator->Allocate((void**)&d_values,     sizeof(Value) * num_vertices));
    CubDebugExit(allocator->Allocate((void**)&d_vector,     sizeof(Value) * coo_graph.col_dim));
    CubDebugExit(allocator->Allocate((void**)&d_result,     sizeof(Value) * coo_graph.row_dim));

    // Copy host arrays to device
    CubDebugExit(cudaMemcpy(d_rows,     h_rows,     sizeof(VertexId) * num_vertices, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemcpy(d_columns,  h_columns,  sizeof(VertexId) * num_vertices, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemcpy(d_values,   h_values,   sizeof(Value) * num_vertices, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemcpy(d_vector,   h_vector,   sizeof(Value) * coo_graph.col_dim, cudaMemcpyHostToDevice));

    // Zero-out the output array
    CubDebugExit(cudaMemset(d_result, 0, sizeof(Value) * coo_graph.row_dim));

    // Figure out launch params and allocate temporaries
    int grid_size = (num_vertices + TILE_SIZE - 1) / TILE_SIZE;
    CubDebugExit(allocator->Allocate((void**)&d_cta_aggregates, sizeof(PartialSum<VertexId, Value>) * grid_size));

    // Run kernel
    GpuTimer gpu_timer;
    float elapsed_millis = 0;
    for (int i = 0; i < g_iterations; i++)
    {
        gpu_timer.Start();

        Kernel<CTA_THREADS, ITEMS_PER_THREAD><<<grid_size, CTA_THERADS>>>(
            coo_graph.row_dim,
            coo_graph.col_dim,
            num_vertices,
            d_cta_aggregates,
            d_rows,
            d_columns,
            d_values,
            d_vector,
            d_result);

        gpu_timer.Stop();
        elapsed_millis = gpu_timer.ElapsedMillis();
    }

    // Display timing
    float avg_elapsed = elapsed_millis / iterations;
    int total_bytes = ((sizeof(VertexId) + sizeof(VertexId) + sizeof(Value)) * num_vertices) + (sizeof(Value) * 2 * coo_graph.row_dim);
    printf("Average elapsed (%.3f ms), utilized bandwidth (%.3f GB/s), GFLOPS(%.3f)\n",
        avg_elapsed,
        total_bytes / avg_elapsed / 1000.0 / 1000.0,
        num_vertices * 2 / avg_elapsed / 1000.0 / 1000.0);

    // Check results
    AssertEquals(0, CompareDeviceResults(h_reference, d_result, coo_graph.row_dim, g_verbose, g_verbose));

    // Cleanup
    CubDebugExit(allocator->Deallocate(d_rows));
    CubDebugExit(allocator->Deallocate(d_columns));
    CubDebugExit(allocator->Deallocate(d_columns));
    CubDebugExit(allocator->Deallocate(d_vector));
    CubDebugExit(allocator->Deallocate(d_result));
    delete h_rows;
    delete h_columns;
    delete h_values;
}


/**
 * Compute reference answer on CPU
 */
template <typename VertexId, typename Value>
void ComputeReference(
    CooGraph<VertexId, Value>&  coo_graph,
    Value*                      h_vector,
    Value*                      h_reference)
{
    for (VertexId i = 0; i < coo_graph.row_dim; i++)
    {
        h_reference[i] = 0.0;
    }

    for (VertexId i = 0; i < num_vertices; i++)
    {
        h_reference[coo_graph.coo_tuples[i].row] +=
            coo_graph.coo_tuples[i].value *
            h_vector[coo_graph.coo_tuples[i].col];
    }
}


/**
 * Assign arbitrary values to graph vertices
 */
template <typename CooGraph>
void AssignGraphValues(CooGraph &coo_graph)
{
    for (VertexId i = 0; i < coo_graph.coo_tuples.size(); i++)
    {
        coo_graph.coo_tuples[i].val = i;
    }
}


/**
 * Assign arbitrary values to vector items
 */
template <typename Value>
void AssignVectorValues(Value *vector, VertexId col_dim)
{
    for (VertexId i = 0; i < col_dim; i++)
    {
        coo_tuples[i] = 1.0;
    }
}


/**
 * Main
 */
int main(int argc, char** argv)
{
    // Graph of int32s as vertex ids, floats as values
    typedef int     VertexId;
    typedef float   Value;

    // Initialize command line
    CommandLineArgs args(argc, argv);
    g_verbose = args.CheckCmdLineFlag("v");
    args.GetCmdLineArgument("i", g_iterations);

    // Print usage
    if (args.CheckCmdLineFlag("help"))
    {
        printf("%s\n [--device=<device-id>] [--v] [--iterations=<test iterations>]\n"
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

    // Generate graph structure
    CooGraph<VertexId, Value> coo_graph;
    if (type == string("grid3d"))
    {
        VertexId width;
        args.GetCmdLineArgument("width", width);
        coo_graph.InitGrid3d(width, coo_graph);
    }
    else
    {
        printf("Unsupported graph type\n");
        exit(1);
    }
    AssignGraphValues(coo_graph);

    // Create vector
    Value *h_vector = new Value[coo_graph.col_dim];
    AssignVectorValues(h_vector, coo_graph.col_dim);

    // Compute reference answer
    Value *h_reference = new Value[coo_graph.row_dim];
    ComputeReference(coo_graph, h_vector, h_refernece);

    // Run GPU version
    TestDevice(coo_graph, h_vector, h_reference);

    // Force any kernel stdio to screen
    CubDebugExit(cudaThreadSynchronize());

    // Cleanup
    delete h_vector;
    delete h_reference;

    return 0;
}



