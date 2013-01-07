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
#include <algorithm>
#include <stdio.h>
#include <test_util.h>

#include "../cub.cuh"

using namespace cub;
using namespace std;


//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

bool    g_verbose       = false;
int     g_iterations    = 1;
int     g_grid_size     = -1;


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

        CooTuple() {}
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

    int                 row_dim;        // Num rows
    int                 col_dim;        // Num cols
    vector<CooTuple>    coo_tuples;     // Non-zero entries


    /**
     * CooGraph ostream operator
     */
    friend std::ostream& operator<<(std::ostream& os, const CooGraph& coo_graph)
    {
        os << "Sparse COO (" << coo_graph.row_dim << " rows, " << coo_graph.col_dim << " cols, " << coo_graph.coo_tuples.size() << " nonzeros):\n";
        os << "Ordinal, Row, Col, Val\n";
        for (int i = 0; i < coo_graph.coo_tuples.size(); i++)
        {
            os << i << ',' << coo_graph.coo_tuples[i].row << ',' << coo_graph.coo_tuples[i].col << ',' << coo_graph.coo_tuples[i].val << "\n";
        }
        return os;
    }

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
     * Builds a wheel COO sparse graph having spokes spokes.
     */
    void InitWheel(VertexId spokes)
    {
        VertexId edges  = spokes + (spokes - 1);

        coo_tuples.clear();
        coo_tuples.reserve(edges);

        // Add spoke edges
        for (VertexId i = 0; i < spokes; i++)
        {
            coo_tuples.push_back(CooTuple(0, i + 1));
        }

        // Add rim
        for (VertexId i = 0; i < spokes; i++)
        {
            VertexId dest = (i + 1) % spokes;
            coo_tuples.push_back(CooTuple(i + 1, dest + 1));
        }

        // Sort by rows, then columns, update dims
        std::stable_sort(coo_tuples.begin(), coo_tuples.end(), CooTupleCompare);
        UpdateDims();
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
        coo_tuples.reserve(edges);

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

        // Sort by rows, then columns, update dims
        std::stable_sort(coo_tuples.begin(), coo_tuples.end(), CooTupleCompare);
        UpdateDims();

    }
};




//---------------------------------------------------------------------
// CooKernel types and constants
//---------------------------------------------------------------------


/// Pairing of dot product partial sums and corresponding row-id
template <typename VertexId, typename Value>
struct PartialSum
{
    Value       partial;        /// PartialSum sum
    VertexId    row;            /// Row-id

    /// Tags indicating this structure provides overloaded ThreadLoad and ThreadStore operations
    typedef void ThreadLoadTag;
    typedef void ThreadStoreTag;

    /// ThreadLoad (simply defer to loading individual items)
    template <PtxLoadModifier MODIFIER>
    __device__ __forceinline__ void ThreadLoad(PartialSum *ptr)
    {
        partial = cub::ThreadLoad<MODIFIER>(&(ptr->partial));
        row = cub::ThreadLoad<MODIFIER>(&(ptr->row));
    }

     /// ThreadStore (simply defer to storing individual items)
    template <PtxStoreModifier MODIFIER>
    __device__ __forceinline__ void ThreadStore(PartialSum *ptr) const
    {
        // Always write partial first
        cub::ThreadStore<MODIFIER>(&(ptr->partial), partial);
        cub::ThreadStore<MODIFIER>(&(ptr->row), row);
    }

};


/// Scan progress
template <typename VertexId, typename Value>
struct ScanProgress
{
    int                             edge;
    PartialSum<VertexId, Value>     aggregate;
};



/// Reduce-by-row scan operator
struct ScanOp
{
    template <typename VertexId, typename Value>
    __device__ __forceinline__ PartialSum<VertexId, Value> operator()(
        const PartialSum<VertexId, Value> &first,
        const PartialSum<VertexId, Value> &second)
    {
        PartialSum<VertexId, Value> retval;
        retval.partial = (second.row != first.row) ?
                second.partial :
                first.partial + second.partial;
        retval.row = second.row;

        return retval;
    }
};


/// Returns true if row_b is the start of a new row
struct NewRowOp
{
    template <typename VertexId>
    __device__ __forceinline__ bool operator()(
        const VertexId& row_a,
        const VertexId& row_b)
    {
        return (row_a != row_b);
    }
};


//---------------------------------------------------------------------
// GPU kernels
//---------------------------------------------------------------------

template <typename Value>
struct TexVector
{
    // Texture reference type
    typedef texture<Value, cudaTextureType1D, cudaReadModeElementType> TexRef;

    static TexRef ref;

    /**
     * Bind textures
     */
    static void BindTexture(void *d_in, int elements)
    {
        cudaChannelFormatDesc tex_desc = cudaCreateChannelDesc<Value>();
        if (d_in)
        {
            size_t offset;
            size_t bytes = sizeof(Value) * elements;
            CubDebugExit(cudaBindTexture(&offset, ref, d_in, tex_desc, bytes));
        }
    }

    /**
     * Unbind textures
     */
    static void UnbindTexture()
    {
        CubDebugExit(cudaUnbindTexture(ref));
    }
};

// Texture reference definitions
template <typename Value>
typename TexVector<Value>::TexRef TexVector<Value>::ref = 0;



/**
 * CTA abstraction for processing sparse SPMV tiles
 */
template <
    int             CTA_THREADS,
    int             ITEMS_PER_THREAD,
    typename        VertexId,
    typename        Value>
struct SpmvCta
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
    typedef int                                                     HeadFlag;

    // Dot product partial sum type
    typedef PartialSum<VertexId, Value>                             PartialSum;

    // Parameterize cooperative CUB types for use in the current problem context
    typedef CtaScan<PartialSum, CTA_THREADS>                        CtaScan;
    typedef CtaExchange<PartialSum, CTA_THREADS, ITEMS_PER_THREAD>  CtaExchange;
    typedef CtaDiscontinuity<HeadFlag, CTA_THREADS>                 CtaDiscontinuity;

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
    // Operations
    //---------------------------------------------------------------------

    __device__ __forceinline__
    static void ProcessTile(
        SmemStorage                     &s_storage,
        ScanProgress<VertexId, Value>*  d_scan_progress,
        VertexId*                       d_rows,
        VertexId*                       d_columns,
        Value*                          d_values,
        Value*                          d_vector,
        Value*                          d_result,
        int                             num_edges,
        int                             cta_offset,
        int                             guarded_items = 0)
    {
        VertexId    columns[ITEMS_PER_THREAD];
        VertexId    rows[ITEMS_PER_THREAD];
        Value       values[ITEMS_PER_THREAD];
        PartialSum  partial_sums[ITEMS_PER_THREAD];
        HeadFlag    head_flags[ITEMS_PER_THREAD];

        // Load a CTA-striped tile of A (sparse row-ids, column-ids, and values)
        if (guarded_items)
        {
            // Last tile has guarded loads.  Extend the coordinates of the last
            // vertex for out-of-bound items, but zero-valued
            VertexId last_row = d_rows[num_edges - 1];
            VertexId last_column = d_columns[num_edges - 1];

            CtaLoadDirectStriped(rows, d_rows, cta_offset, guarded_items, last_row);
            CtaLoadDirectStriped(columns, d_columns, cta_offset, guarded_items, last_column);
            CtaLoadDirectStriped(values, d_values, cta_offset, guarded_items, Value(0.0));
        }
        else
        {
            // Unguarded loads
            CtaLoadDirectStriped(rows, d_rows, cta_offset);
            CtaLoadDirectStriped(columns, d_columns, cta_offset);
            CtaLoadDirectStriped(values, d_values, cta_offset);
        }

        // Fence to prevent hoisting any dependent code below into the loads above
        __threadfence_block();

        // Load the referenced values from x and compute dot product partial_sums
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            Value vec_item = tex1Dfetch(TexVector<Value>::ref, columns[ITEM]);
            partial_sums[ITEM].partial = values[ITEM] * vec_item;
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
        VertexId        first_row = d_rows[cta_offset];
        ScanOp          scan_op;                          // Reduce-by-row scan operator
        PartialSum      aggregate;                        // CTA-wide aggregate in thread0
        PartialSum      identity = {0.0, first_row};      // Zero-valued identity (with row-id of first item)

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
            // Get aggregate from prior CTA
            PartialSum prev_aggregate;
            if (blockIdx.x == 0)
            {
                // First tile has no prior aggregate
                prev_aggregate.row = first_row;
                prev_aggregate.partial = 0.0;
            }
            else
            {
                // Keep loading prior CTA's aggregate until valid
                int edge;
                do {
                    edge = ThreadLoad<PTX_LOAD_CG>(&d_scan_progress->edge);
                }
                while (edge != cta_offset);

                prev_aggregate = ThreadLoad<PTX_LOAD_CG>(&d_scan_progress->aggregate);
            }

            // Share prev_aggregate with other threads
            s_storage.prev_aggregate = prev_aggregate;

            // Apply prev_aggregate to our local aggregate
            aggregate = scan_op(prev_aggregate, aggregate);

            // Write updated CTA-wide aggregate
            ThreadStore<PTX_STORE_CG>(&d_scan_progress->aggregate, aggregate);

            // Signal to subsequent CTA that value is ready
            __threadfence_block();
            ThreadStore<PTX_STORE_CG>(&d_scan_progress->edge, cta_offset + TILE_ITEMS);
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

        // First item of first thread should be the previous CTA's aggregate (and not identity)
        if (threadIdx.x == 0)
        {
            partial_sums[0] = s_storage.prev_aggregate;
        }

        // Scatter dot products if a row head of a valid row
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            if (head_flags[ITEM])
            {
                d_result[partial_sums[ITEM].row] = partial_sums[ITEM].partial;
            }
        }

        // Last tile scatters the final value (if it has a valid row id), which is the aggregate
        if ((blockIdx.x == gridDim.x - 1) && (threadIdx.x == 0) && (aggregate.row >= 0))
        {
            d_result[aggregate.row] = aggregate.partial;
        }
    }

};


/**
 * COO SpMV kernel
 */
template <
    int             CTA_THREADS,
    int             ITEMS_PER_THREAD,
    typename        VertexId,
    typename        Value>
__global__ void CooKernel(
    CtaProgress<int>                cta_progress,
    ScanProgress<VertexId, Value>*  d_scan_progress,
    VertexId*                       d_rows,
    VertexId*                       d_columns,
    Value*                          d_values,
    Value*                          d_vector,
    Value*                          d_result)
{
    const int TILE_SIZE = CTA_THREADS * ITEMS_PER_THREAD;

    // CTA type
    typedef SpmvCta<CTA_THREADS, ITEMS_PER_THREAD, VertexId, Value> SpmvCta;

    // Shared memory
    __shared__ typename SpmvCta::SmemStorage s_storage;

    // Process full tiles of sparse matrix
    cta_progress.Init();

    int cta_offset;
    while (cta_progress.NextFull(TILE_SIZE, cta_offset))
    {
        if (threadIdx.x == 0) printf("block(%d) cta_offset(%d)\n",
            blockIdx.x, cta_offset);

        SpmvCta::ProcessTile(
            s_storage,
            d_scan_progress,
            d_rows,
            d_columns,
            d_values,
            d_vector,
            d_result,
            cta_progress.TotalItems(),
            cta_offset);

        // Barrier for smem reuse and coherence
        __syncthreads();
    }

    // Process last partial tile (if any)
    if (cta_progress.NextPartial(cta_offset))
    {
        if (threadIdx.x == 0) printf("block(%d) cta_offset(%d) guarded_items(%d)\n",
            blockIdx.x, cta_offset, cta_progress.GuardedItems());

        SpmvCta::ProcessTile(
            s_storage,
            d_scan_progress,
            d_rows,
            d_columns,
            d_values,
            d_vector,
            d_result,
            cta_progress.TotalItems(),
            cta_offset,
            cta_progress.GuardedItems());
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
    if (g_iterations <= 0) return;

    const int TILE_SIZE = CTA_THREADS * ITEMS_PER_THREAD;

    // SOA device storage
    VertexId*                       d_rows;             // SOA graph row coordinates
    VertexId*                       d_columns;          // SOA graph col coordinates
    Value*                          d_values;           // SOA graph values
    Value*                          d_vector;           // Vector multiplicand
    Value*                          d_result;           // Output row
    PartialSum<VertexId, Value>*    d_cta_aggregates;   // Temporary storage for communicating dot product partials between CTAs

    // Create SOA version of coo_graph on host
    int                             num_edges       = coo_graph.coo_tuples.size();
    VertexId*                       h_rows          = new VertexId[num_edges];
    VertexId*                       h_columns       = new VertexId[num_edges];
    Value*                          h_values        = new Value[num_edges];
    for (int i = 0; i < num_edges; i++)
    {
        h_rows[i]       = coo_graph.coo_tuples[i].row;
        h_columns[i]    = coo_graph.coo_tuples[i].col;
        h_values[i]     = coo_graph.coo_tuples[i].val;
    }

    // Determine launch params
    CudaProps cuda_props;
    KernelProps kernel_props;
    CubDebugExit(cuda_props.Init());
    CubDebugExit(kernel_props.Init(
        CooKernel<CTA_THREADS, ITEMS_PER_THREAD, VertexId, Value>,
        CTA_THREADS,
        cuda_props));

    g_grid_size = kernel_props.OversubscribedGridSize(TILE_SIZE, num_edges, g_grid_size);

    // Utility for tracking CTA progress
    CtaProgress<int> cta_progress(num_edges, g_grid_size, TILE_SIZE);

    // Print debug info
    printf("CooKernel<%d, %d><<<%d, %d>>>(...)\n",
        CTA_THREADS,
        ITEMS_PER_THREAD,
        g_grid_size,
        CTA_THREADS);
    printf("Max SM occupancy: %d\n", kernel_props.max_cta_occupancy);
    cta_progress.Print();

    // Allocate COO device arrays
    CachedAllocator *allocator = CubCachedAllocator<void>();
    CubDebugExit(allocator->Allocate((void**)&d_rows,       sizeof(VertexId) * num_edges));
    CubDebugExit(allocator->Allocate((void**)&d_columns,    sizeof(VertexId) * num_edges));
    CubDebugExit(allocator->Allocate((void**)&d_values,     sizeof(Value) * num_edges));
    CubDebugExit(allocator->Allocate((void**)&d_vector,     sizeof(Value) * coo_graph.col_dim));
    CubDebugExit(allocator->Allocate((void**)&d_result,     sizeof(Value) * coo_graph.row_dim));
    CubDebugExit(allocator->Allocate((void**)&d_scan_progress, sizeof(ScanProgress<VertexId, Value>)));

    // Copy host arrays to device
    CubDebugExit(cudaMemcpy(d_rows,     h_rows,     sizeof(VertexId) * num_edges, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemcpy(d_columns,  h_columns,  sizeof(VertexId) * num_edges, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemcpy(d_values,   h_values,   sizeof(Value) * num_edges, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemcpy(d_vector,   h_vector,   sizeof(Value) * coo_graph.col_dim, cudaMemcpyHostToDevice));

    // Bind textures
    TexVector<Value>::BindTexture(d_vector, coo_graph.col_dim);

    // Run kernel
    GpuTimer gpu_timer;
    float elapsed_millis = 0.0;
    for (int i = 0; i < g_iterations; i++)
    {
        gpu_timer.Start();

        // Zero-out the output array
        CubDebugExit(cudaMemset(d_result, 0, sizeof(Value) * coo_graph.row_dim));

        // Initialize temporaries
        CubDebugExit(cudaMemset(d_scan_progress, 0, sizeof(ScanProgress<VertexId, Value>)));

        // Run the COO kernel
        CooKernel<CTA_THREADS, ITEMS_PER_THREAD><<<g_grid_size, CTA_THREADS>>>(
            cta_progress,
            d_scan_progress,
            d_rows,
            d_columns,
            d_values,
            d_vector,
            d_result);

        gpu_timer.Stop();
        elapsed_millis += gpu_timer.ElapsedMillis();

        // Force any kernel stdio to screen
        CubDebugExit(cudaThreadSynchronize());
    }

    // Display timing
    float avg_elapsed = elapsed_millis / g_iterations;
    int total_bytes = ((sizeof(VertexId) + sizeof(VertexId) + sizeof(Value)) * num_edges) + (sizeof(Value) * 2 * coo_graph.row_dim);
    printf("%d iterations, average elapsed (%.3f ms), utilized bandwidth (%.3f GB/s), GFLOPS(%.3f)\n",
        g_iterations,
        avg_elapsed,
        total_bytes / avg_elapsed / 1000.0 / 1000.0,
        num_edges * 2 / avg_elapsed / 1000.0 / 1000.0);

    // Check results
    AssertEquals(0, CompareDeviceResults(h_reference, d_result, coo_graph.row_dim, g_verbose, g_verbose));

    // Cleanup
    TexVector<Value>::UnbindTexture();
    CubDebugExit(allocator->Deallocate(d_scan_progress));
    CubDebugExit(allocator->Deallocate(d_rows));
    CubDebugExit(allocator->Deallocate(d_columns));
    CubDebugExit(allocator->Deallocate(d_values));
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

    for (VertexId i = 0; i < coo_graph.coo_tuples.size(); i++)
    {
        h_reference[coo_graph.coo_tuples[i].row] +=
            coo_graph.coo_tuples[i].val *
            h_vector[coo_graph.coo_tuples[i].col];
    }
}


/**
 * Assign arbitrary values to graph vertices
 */
template <typename CooGraph>
void AssignGraphValues(CooGraph &coo_graph)
{
    for (int i = 0; i < coo_graph.coo_tuples.size(); i++)
    {
        coo_graph.coo_tuples[i].val = i % 21;
    }
}


/**
 * Assign arbitrary values to vector items
 */
template <typename Value>
void AssignVectorValues(Value *vector, int col_dim)
{
    for (int i = 0; i < col_dim; i++)
    {
        vector[i] = 1.0;
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
    args.GetCmdLineArgument("grid-size", g_grid_size);

    // Print usage
    if (args.CheckCmdLineFlag("help"))
    {
        printf("%s\n [--device=<device-id>] [--v] [--iterations=<test iterations>] [--grid-size=<grid-size>]\n"
            "\t--type=wheel --spokes=<spokes>\n"
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
        printf("Generating grid3d width(%d)... ", width); fflush(stdout);
        coo_graph.InitGrid3d(width);
        printf("Done.  %d non-zeros, %d rows, %d columns\n",
            coo_graph.coo_tuples.size(), coo_graph.row_dim, coo_graph.col_dim); fflush(stdout);
    }
    else if (type == string("wheel"))
    {
        VertexId spokes;
        args.GetCmdLineArgument("spokes", spokes);
        printf("Generating wheel spokes(%d)... ", spokes); fflush(stdout);
        coo_graph.InitWheel(spokes);
        printf("Done.  %d non-zeros, %d rows, %d columns\n",
            coo_graph.coo_tuples.size(), coo_graph.row_dim, coo_graph.col_dim); fflush(stdout);
    }
    else
    {
        printf("Unsupported graph type\n");
        exit(1);
    }
    AssignGraphValues(coo_graph);

    if (g_verbose)
    {
        cout << coo_graph << "\n";
    }

    // Create vector
    Value *h_vector = new Value[coo_graph.col_dim];
    AssignVectorValues(h_vector, coo_graph.col_dim);
    if (g_verbose)
    {
        printf("Vector[%d]: ", coo_graph.col_dim);
        DisplayResults(h_vector, coo_graph.col_dim);
        printf("\n\n");
    }

    // Compute reference answer
    Value *h_reference = new Value[coo_graph.row_dim];
    ComputeReference(coo_graph, h_vector, h_reference);
    if (g_verbose)
    {
        printf("Results[%d]: ", coo_graph.row_dim);
        DisplayResults(h_reference, coo_graph.row_dim);
        printf("\n\n");
    }

    // Run GPU version
    TestDevice<128, 5>(coo_graph, h_vector, h_reference);

    // Cleanup
    delete h_vector;
    delete h_reference;

    return 0;
}



