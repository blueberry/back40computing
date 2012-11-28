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
// GPU kernels
//---------------------------------------------------------------------


/**
 *
 */
template <
    int     CTA_THREADS,
    int     ITEMS_PER_THREAD>
__launch_bounds__ (CTA_THREADS)
__global__ void Kernel(
    int*        d_columns,
    float*      d_values,
    float*      d_vec,
    float*      d_output)
{
    //---------------------------------------------------------------------
    // Types and constants
    //---------------------------------------------------------------------

    // Constants
    enum
    {
        NEW_ROW_MASK = 1 << 31,
    };


    // Returns true if flag_b is the start of a new row
    struct NewRowFunctor
    {
        bool operator()(const int &flag_a, const int &flag_b)
        {
            return column_b;
        }
    };

    // Pairing of product-partial and corresponding column-id
    struct Partial
    {
        float   partial;
        int     flag;

        // Default Constructor
        Partial() {}

        // Constructor
        Partial(float partial, int flag) : partial(partial), flag(flag) {}

    };

    // Scan operator
    struct ScanOp
    {
        __device__ __forceinline__ Partial operator()(
            const Partial &first,
            const Partial &second)
        {
            return Partial(
                (second.flag) ?
                    second.partial :
                    first.partial + second.partial,
                first.flag + second.flag);
        }
    };


    // Parameterize cooperative CUB types for use in the current problem context
    typedef cub::CtaScan<int, CTA_THREADS>              CtaScan;
    typedef cub::CtaExchange<ValFlagPair, CTA_THREADS>  CtaExchange;

    // Shared memory type
    union SmemStorage
    {
        typename CtaScan::SmemStorage           scan;
        typename CtaExchange::SmemStorage       exchange;
    };


    //---------------------------------------------------------------------
    // Kernel body
    //---------------------------------------------------------------------

    // Declare shared memory
    __shared__ SmemStorage smem_storage;

    int         columns[ITEMS_PER_THREAD];
    float       values[ITEMS_PER_THREAD];
    Partial     partials[ITEMS_PER_THREAD];


    // The CTA's offset in d_columns and d_values
    int cta_offset = blockIdx.x * CTA_THREADS * ITEMS_PER_THREAD;

    // Load a CTA-striped tile of sparse columns and values
    CtaLoadDirectStriped(columns, d_columns, cta_offset);
    CtaLoadDirectStriped(values, d_values, cta_offset);

    // Fence to prevent hoisting into loads above
    threadfence_block();

    // Compute product partials and row head flags
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
    {
        // Load the referenced values from x and compute dot product partials
        partials[ITEM].partial = partials[ITEM] * d_vec[a_columns[ITEM]];

        // Set head-flag if value starts a new row
        partials[ITEM].flag = (a_columns[ITEM] & NEW_ROW_MASK) ? 1 : 0;
    }

    // Transpose from CTA-striped to CTA-blocked arrangement
    CtaExchange::StripedToBlocked(smem_storage.exchange, partials);

    // Barrier for smem reuse
    __syncthreads();

    // Compute exclusive scan of items
    Partial aggregate;
    CtaScan::ExclusiveScan(
        smem_storage.scan,
        partials,
        partials,
        Partial(0.0, 0),        // identity
        ScanOp(),
        aggregate);






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



