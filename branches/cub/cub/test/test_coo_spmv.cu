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


//---------------------------------------------------------------------
// Graph building types and utilities
//---------------------------------------------------------------------

/**
 * COO graph type.  A COO graph is just a vector of edge tuples.
 */
template<typename VertexId, typename Value>
struct CooGraph
{
    /**
     * COO edge tuple.  (A COO graph is just a vector of these.)
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
// GPU types and device functions
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
    /// The offset of the first waiting CTA
    int active_offset;

    /// The inter-CTA aggregate so far
    PartialSum<VertexId, Value> aggregate;
};


/// Templated Texture reference type for multiplicand vector
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

    // Parameterized CUB types for use in the current problem context
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

        PartialSum prev_aggregate;
        PartialSum aggregate;
    };


    /// Reduce-by-row scan operator
    struct ScanOp
    {
        __device__ __forceinline__ PartialSum operator()(
            const PartialSum &first,
            const PartialSum &second)
        {
            PartialSum retval;

            retval.partial = (second.row != first.row) ?
                    second.partial :
                    first.partial + second.partial;

            retval.row = second.row;
            return retval;
        }
    };

    // Callback functor for waiting on the previous CTA to compute its partial sum (the prefix for this CTA)
    struct CtaPrefixOp
    {
        ScanProgress<VertexId, Value>       *d_scan_progress;
        int                                 cta_offset;
        PartialSum                          identity;
        PartialSum                          prev_aggregate;
        PartialSum                          aggregate;
        ScanOp                              scan_op;

        /// Constructor
        __device__ __forceinline__ CtaPrefixOp(
            ScanProgress<VertexId, Value>   *d_scan_progress,
            int                             cta_offset,
            PartialSum                      identity,
            ScanOp                          scan_op) :
                cta_offset(cta_offset),
                d_scan_progress(d_scan_progress),
                identity(identity),
                scan_op(scan_op)
        {}


        /**
         * CTA-wide prefix callback functor called by thread-0 in CtaScan::ExclusiveScan().
         * Returns the CTA-wide prefix to apply to all scan inputs.
         */
        __device__ __forceinline__ PartialSum operator()(
            const PartialSum &local_aggregate)              ///< The aggregate sum of the local prefix sum inputs
        {
            // Get aggregate from prior CTA
            if (cta_offset == 0)
            {
                // The first tile has no prior aggregate: use identity
                prev_aggregate = identity;
            }
            else
            {
                // Keep loading prior CTA's aggregate until valid
                while (ThreadLoad<PTX_LOAD_CG>(&d_scan_progress->active_offset) != cta_offset)
                {
                    __threadfence_block();
                }

                // It's our turn: load the inter-CTA aggregate up to this point
                prev_aggregate = ThreadLoad<PTX_LOAD_CG>(&d_scan_progress->aggregate);
            }

            // Write updated CTA-wide aggregate and signal to subsequent CTA that value is ready
            aggregate = scan_op(prev_aggregate, local_aggregate);

            ThreadStore<PTX_STORE_CG>(&d_scan_progress->aggregate, aggregate);
            __threadfence_block();
            ThreadStore<PTX_STORE_CG>(&d_scan_progress->active_offset, cta_offset + TILE_ITEMS);

            // Return CTA's prefix
            return prev_aggregate;
        }
    };


    /// Functor for detecting row discontinuities.
    struct NewRowOp
    {
        /// Returns true if row_b is the start of a new row
        template <typename VertexId>
        __device__ __forceinline__ bool operator()(
            const VertexId& row_a,
            const VertexId& row_b)
        {
            return (row_a != row_b);
        }
    };

    //---------------------------------------------------------------------
    // Operations
    //---------------------------------------------------------------------

    /**
     * Processes a COO input tile of edges, outputting dot products for each row
     */
    __device__ __forceinline__
    static void ProcessTile(
        SmemStorage                     &s_storage,
        ScanProgress<VertexId, Value>   *d_scan_progress,
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
            // This is a partial-tile (e.g., the last tile of input).  Extend the coordinates of the last
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
        // Mooch
        __threadfence_block();

        // Load the referenced values from x and compute the dot product partials sums
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

        // Save a copy of the original row IDs.  We will use them to compute the row head flags
        // later.  (After the scan, the row fields in partial_sums will
        // be shifted by one element because the scan's exclusive nature.)
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            rows[ITEM] = partial_sums[ITEM].row;
        }

        // Compute exclusive scan of partial_sums
        VertexId        first_row = d_rows[cta_offset];                             // The vertex id of the first row encountered by this CTA
        ScanOp          scan_op;                                                    // Reduce-by-row scan operator
        PartialSum      local_aggregate;                                            // CTA-wide aggregate in thread0 (unused)
        PartialSum      identity = {0.0, first_row};                                // Zero-valued identity (with row-id of first item)
        CtaPrefixOp     prefix_op(d_scan_progress, cta_offset, identity, scan_op);  // Callback functor for waiting on the previous CTA to compute its partial sum

/*

        CtaScan::ExclusiveScan(
            s_storage.scan,
            partial_sums,
            partial_sums,           // (Out)
            identity,
            scan_op,
            local_aggregate,        // (Out)
            prefix_op);             // (In-out)
*/

        CtaScan::ExclusiveScan(
            s_storage.scan,
            partial_sums,
            partial_sums,           // (Out)
            identity,
            scan_op,
            local_aggregate);       // (Out)

        // Get aggregate from prior CTA
        if (threadIdx.x == 0)
        {
            if (cta_offset == 0)
            {
                // The first tile has no prior aggregate: use identity
                s_storage.prev_aggregate = identity;
            }
            else
            {
                // Keep loading prior CTA's aggregate until valid
                while (ThreadLoad<PTX_LOAD_CG>(&d_scan_progress->active_offset) != cta_offset)
                {
                    __threadfence_block();
                }

                // It's our turn: load the inter-CTA aggregate up to this point
                s_storage.prev_aggregate = ThreadLoad<PTX_LOAD_CG>(&d_scan_progress->aggregate);
            }

            // Write updated CTA-wide aggregate and signal to subsequent CTA that value is ready
            s_storage.aggregate = scan_op(s_storage.prev_aggregate, local_aggregate);

            ThreadStore<PTX_STORE_CG>(&d_scan_progress->aggregate, s_storage.aggregate);
            __threadfence_block();
            ThreadStore<PTX_STORE_CG>(&d_scan_progress->active_offset, cta_offset + TILE_ITEMS);
        }

        // Barrier for smem reuse and coherence
        __syncthreads();

        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            partial_sums[ITEM] = scan_op(s_storage.prev_aggregate, partial_sums[ITEM]);
        }

        if (threadIdx.x == 0)
        {
            partial_sums[0] = s_storage.prev_aggregate;
        }


        // Flag row heads using saved row ids
        CtaDiscontinuity::Flag(
            s_storage.discontinuity,
            rows,                           // Original row ids
            prefix_op.prev_aggregate.row,   // Last row id from previous CTA
            NewRowOp(),                     // Functor for detecting start of new rows
            head_flags);                    // (Out) Head flags

        // Scatter the dot products if they are the head of a valid row
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            if (head_flags[ITEM])
            {
                d_result[partial_sums[ITEM].row] = partial_sums[ITEM].partial;
            }
        }
/*
        // The last tile scatters its aggregate value as the last output
        if (
            (cta_offset + TILE_ITEMS >= num_edges) &&           // Last tile
            (threadIdx.x == 0) &&                               // First thread
            (prefix_op.aggregate.row >= 0))                     // Valid row ID
        {
            d_result[prefix_op.aggregate.row] = prefix_op.aggregate.partial;
        }
*/

        // The last tile scatters its aggregate value as the last output
        if (
            (cta_offset + TILE_ITEMS >= num_edges) &&           // Last tile
            (threadIdx.x == 0) &&                               // First thread
            (s_storage.aggregate.row >= 0))                     // Valid row ID
        {
            d_result[s_storage.aggregate.row] = s_storage.aggregate.partial;
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
    GridQueue<int>                  scan_queue,
    ScanProgress<VertexId, Value>*  d_scan_progress,
    VertexId*                       d_rows,
    VertexId*                       d_columns,
    Value*                          d_values,
    Value*                          d_vector,
    Value*                          d_result,
    int                             num_edges)
{
    const int TILE_SIZE = CTA_THREADS * ITEMS_PER_THREAD;

    // CTA type
    typedef SpmvCta<CTA_THREADS, ITEMS_PER_THREAD, VertexId, Value> SpmvCta;

    // Shared memory
    __shared__ typename SpmvCta::SmemStorage s_storage;
    __shared__ int cta_offset;

    // Process tiles of sparse matrix
    while (true)
    {
        // Thread0 steals a tile of work
        if (threadIdx.x == 0)
        {
            cta_offset = scan_queue.Drain(TILE_SIZE);
        }

        __syncthreads();

        if (cta_offset >= num_edges)
        {
            // Done
            break;
        }
        else if (cta_offset + TILE_SIZE < num_edges)
        {
            // Full tile
            SpmvCta::ProcessTile(
                s_storage,
                d_scan_progress,
                d_rows,
                d_columns,
                d_values,
                d_vector,
                d_result,
                num_edges,
                cta_offset);
        }
        else if (cta_offset < num_edges)
        {
            // Partial tile
            int guarded_items = num_edges - cta_offset;
            SpmvCta::ProcessTile(
                s_storage,
                d_scan_progress,
                d_rows,
                d_columns,
                d_values,
                d_vector,
                d_result,
                num_edges,
                cta_offset,
                guarded_items);
        }

        __syncthreads();
    }
}



/**
 * COO Initialization kernel.  Initializes queue counters and output vector to zero.
 */
template <typename VertexId, typename Value>
__global__ void InitKernel(
    GridQueue<int>                  scan_queue,         ///< Queue counters
    ScanProgress<VertexId, Value>*  d_scan_progress,    ///< Scan progress state
    Value*                          d_result,           ///< Output vector
    int                             vector_length)      ///< Output vector length
{
    // Reset queue counters to known state
    if ((blockIdx.x == 0) && (threadIdx.x == 0))
    {
        scan_queue.PrepareDrain();
        d_scan_progress->active_offset = 0;
    }

    // Initialize output vector elements to 0.0
    const int STRIDE = gridDim.x * blockDim.x;
    for (int idx = (blockIdx.x * blockDim.x) + threadIdx.x; idx < vector_length; idx += STRIDE)
    {
        d_result[idx] = 0;
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

    // SOA device storage
    VertexId*                       d_rows;             // SOA graph row coordinates
    VertexId*                       d_columns;          // SOA graph col coordinates
    Value*                          d_values;           // SOA graph values
    Value*                          d_vector;           // Vector multiplicand
    Value*                          d_result;           // Output row
    ScanProgress<VertexId, Value>*  d_scan_progress;    // Temporary storage for communicating dot product partials between CTAs

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

    // Allocate COO device arrays
    CubDebugExit(DeviceAllocate((void**)&d_rows,           sizeof(VertexId) * num_edges));
    CubDebugExit(DeviceAllocate((void**)&d_columns,        sizeof(VertexId) * num_edges));
    CubDebugExit(DeviceAllocate((void**)&d_values,         sizeof(Value) * num_edges));
    CubDebugExit(DeviceAllocate((void**)&d_vector,         sizeof(Value) * coo_graph.col_dim));
    CubDebugExit(DeviceAllocate((void**)&d_result,         sizeof(Value) * coo_graph.row_dim));
    CubDebugExit(DeviceAllocate((void**)&d_scan_progress,  sizeof(ScanProgress<VertexId, Value>)));

    // Copy host arrays to device
    CubDebugExit(cudaMemcpy(d_rows,     h_rows,     sizeof(VertexId) * num_edges, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemcpy(d_columns,  h_columns,  sizeof(VertexId) * num_edges, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemcpy(d_values,   h_values,   sizeof(Value) * num_edges, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemcpy(d_vector,   h_vector,   sizeof(Value) * coo_graph.col_dim, cudaMemcpyHostToDevice));

    // Bind textures
    TexVector<Value>::BindTexture(d_vector, coo_graph.col_dim);

    // Queue for managing CTA work assignment
    GridQueue<int> scan_queue;
    scan_queue.Allocate();

    // Get kernel properties
    CudaProps cuda_props;
    KernelProps init_kernel_props;
    KernelProps coo_kernel_props;
    CubDebugExit(cuda_props.Init());
    CubDebugExit(init_kernel_props.Init(InitKernel<VertexId, Value>, CTA_THREADS, cuda_props));
    CubDebugExit(coo_kernel_props.Init(CooKernel<CTA_THREADS, ITEMS_PER_THREAD, VertexId, Value>, CTA_THREADS, cuda_props));

    // Determine launch configuration from kernel properties
    int init_grid_size = init_kernel_props.OversubscribedGridSize(CTA_THREADS, coo_graph.row_dim);
    int coo_grid_size = coo_kernel_props.ResidentGridSize();

    // Print debug info
    printf("InitKernel<<<%d, %d>>>(...), Max SM occupancy: %d\n",
        init_grid_size, CTA_THREADS, init_kernel_props.max_cta_occupancy);
    printf("CooKernel<%d, %d><<<%d, %d>>>(...), Max SM occupancy: %d\n",
        CTA_THREADS, ITEMS_PER_THREAD, coo_grid_size, CTA_THREADS, coo_kernel_props.max_cta_occupancy);
    fflush(stdout);

    // Run kernel
    GpuTimer gpu_timer;
    float elapsed_millis = 0.0;
    for (int i = 0; i < g_iterations; i++)
    {
        gpu_timer.Start();

        // Initialize output and temporaries
        InitKernel<<<init_grid_size, CTA_THREADS>>>(
            scan_queue,
            d_scan_progress,
            d_result,
            coo_graph.row_dim);

        // Run the COO kernel
        CooKernel<CTA_THREADS, ITEMS_PER_THREAD><<<coo_grid_size, CTA_THREADS>>>(
            scan_queue,
            d_scan_progress,
            d_rows,
            d_columns,
            d_values,
            d_vector,
            d_result,
            num_edges);

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
    CubDebugExit(DeviceFree(d_scan_progress));
    CubDebugExit(DeviceFree(d_rows));
    CubDebugExit(DeviceFree(d_columns));
    CubDebugExit(DeviceFree(d_values));
    CubDebugExit(DeviceFree(d_vector));
    CubDebugExit(DeviceFree(d_result));
    delete[] h_rows;
    delete[] h_columns;
    delete[] h_values;
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
    delete[] h_vector;
    delete[] h_reference;

    return 0;
}



