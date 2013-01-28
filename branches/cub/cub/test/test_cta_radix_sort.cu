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
 * Test of CtaRadixSort utilities
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <stdio.h>
#include <algorithm>
#include <iostream>

#include <test_util.h>
#include "../cub.cuh"

using namespace cub;

//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

bool g_verbose = false;


//---------------------------------------------------------------------
// Test kernels
//---------------------------------------------------------------------


/**
 * CtaRadixSort kernel
 */
template <
    int                     CTA_THREADS,
    int                     ITEMS_PER_THREAD,
    int                     RADIX_BITS,
    cudaSharedMemConfig     SMEM_CONFIG,
    typename                KeyType,
    typename                ValueType>
__launch_bounds__ (CTA_THREADS, 1)
__global__ void Kernel(
    KeyType             *d_keys,
    ValueType           *d_values,
    unsigned int        begin_bit,
    unsigned int        end_bit)
{
    enum
    {
        TILE_SIZE = CTA_THREADS * ITEMS_PER_THREAD,
        KEYS_ONLY = Equals<ValueType, NullType>::VALUE,
    };

    // CTA load/store abstraction types
    typedef CtaRadixSort<
        KeyType,
        CTA_THREADS,
        ITEMS_PER_THREAD,
        ValueType,
        RADIX_BITS,
        SMEM_CONFIG> CtaRadixSort;

    // Shared memory
    __shared__ typename CtaRadixSort::SmemStorage smem_storage;

    // Keys per thread
    KeyType keys[ITEMS_PER_THREAD];

    if (KEYS_ONLY)
    {
        //
        // Test keys-only sorting (in striped arrangement)
        //

        CtaLoadDirectStriped<CTA_THREADS>(keys, d_keys, 0);

        CtaRadixSort::SortStriped(smem_storage, keys, begin_bit, end_bit);

        CtaStoreDirectStriped<CTA_THREADS>(keys, d_keys, 0);
    }
    else
    {
        //
        // Test keys-value sorting (in striped arrangement)
        //

        // Values per thread
        ValueType values[ITEMS_PER_THREAD];

        CtaLoadDirectStriped<CTA_THREADS>(keys, d_keys, 0);
        CtaLoadDirectStriped<CTA_THREADS>(values, d_values, 0);

        CtaRadixSort::SortStriped(smem_storage, keys, values, begin_bit, end_bit);

        CtaStoreDirectStriped<CTA_THREADS>(keys, d_keys, 0);
        CtaStoreDirectStriped<CTA_THREADS>(values, d_values, 0);
    }
}


//---------------------------------------------------------------------
// Host testing subroutines
//---------------------------------------------------------------------

// Caching allocator (caches up to 6MB in device allocations per GPU)
CachedAllocator g_allocator;


/**
 * Drive CtaRadixSort kernel
 */
template <
    int                     CTA_THREADS,
    int                     ITEMS_PER_THREAD,
    int                     RADIX_BITS,
    cudaSharedMemConfig     SMEM_CONFIG,
    typename                KeyType,
    typename                ValueType>
void TestDriver(
    bool                    keys_only,
    unsigned int            entropy_reduction,
    unsigned int            begin_bit,
    unsigned int            end_bit)
{
    enum
    {
        TILE_SIZE = CTA_THREADS * ITEMS_PER_THREAD,
    };

    // Allocate host arrays
    KeyType     *h_keys             = (KeyType*) malloc(TILE_SIZE * sizeof(KeyType));
    KeyType     *h_reference_keys   = (KeyType*) malloc(TILE_SIZE * sizeof(KeyType));
    ValueType   *h_values           = (ValueType*) malloc(TILE_SIZE * sizeof(ValueType));

    // Allocate device arrays
    KeyType     *d_keys     = NULL;
    ValueType   *d_values   = NULL;
    CubDebugExit(g_allocator.Allocate((void**)&d_keys, sizeof(KeyType) * TILE_SIZE));
    CubDebugExit(g_allocator.Allocate((void**)&d_values, sizeof(ValueType) * TILE_SIZE));

    // Initialize problem on host and device
    for (int i = 0; i < TILE_SIZE; ++i)
    {
        RandomBits(h_keys[i], entropy_reduction, begin_bit, end_bit);
        h_reference_keys[i] = h_keys[i];
        h_values[i] = i;
    }
    CubDebugExit(cudaMemcpy(d_keys, h_keys, sizeof(KeyType) * TILE_SIZE, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemcpy(d_values, h_values, sizeof(ValueType) * TILE_SIZE, cudaMemcpyHostToDevice));

    printf("%s "
        "CTA_THREADS(%d) "
        "ITEMS_PER_THREAD(%d) "
        "RADIX_BITS(%d) "
        "SMEM_CONFIG(%d) "
        "sizeof(KeyType)(%d) "
        "sizeof(ValueType)(%d) "
        "entropy_reduction(%d) "
        "begin_bit(%d) "
        "end_bit(%d)\n",
            ((keys_only) ? "keys-only" : "key-value"),
            CTA_THREADS,
            ITEMS_PER_THREAD,
            RADIX_BITS,
            SMEM_CONFIG,
            (int) sizeof(KeyType),
            (int) sizeof(ValueType),
            entropy_reduction,
            begin_bit,
            end_bit);

    // Compute reference solution
    printf("\tComputing reference solution on CPU..."); fflush(stdout);
    std::sort(h_reference_keys, h_reference_keys + TILE_SIZE);
    printf(" Done.\n"); fflush(stdout);

    cudaDeviceSetSharedMemConfig(SMEM_CONFIG);

    // Run kernel
    if (keys_only)
    {
        // Keys-only
        Kernel<CTA_THREADS, ITEMS_PER_THREAD, RADIX_BITS, SMEM_CONFIG><<<1, CTA_THREADS>>>(
            d_keys, (NullType*) d_values, begin_bit, end_bit);
    }
    else
    {
        // Key-value pairs
        Kernel<CTA_THREADS, ITEMS_PER_THREAD, RADIX_BITS, SMEM_CONFIG><<<1, CTA_THREADS>>>(
            d_keys, d_values, begin_bit, end_bit);
    }

    // Flush kernel output / errors
    CubDebugExit(cudaDeviceSynchronize());

    // Check keys results
    printf("\tKeys: ");
    AssertEquals(0, CompareDeviceResults(h_reference_keys, d_keys, TILE_SIZE, g_verbose, g_verbose));
    printf("\n");

    // Check value results (which aren't valid for 8-bit values and tile size >= 256 because they can't fully index into the starting array)
    if (!keys_only && ((sizeof(ValueType) > 1) || (TILE_SIZE < 256)))
    {
        CubDebugExit(cudaMemcpy(h_values, d_values, sizeof(ValueType) * TILE_SIZE, cudaMemcpyDeviceToHost));

        printf("\tValues: ");
        if (g_verbose)
        {
            DisplayResults(h_values, TILE_SIZE);
            printf("\n");
        }

        bool correct = true;
        for (int i = 0; i < TILE_SIZE; ++i)
        {
            if (h_keys[h_values[i]] != h_reference_keys[i])
            {
                std::cout << "Incorrect: [" << i << "]: " << h_keys[h_values[i]] << " != " << h_reference_keys[i] << std::endl << std::endl;
                correct = false;
                break;
            }
        }
        if (correct) printf("Correct\n");
        AssertEquals(true, correct);
    }
    printf("\n");

    // Cleanup
    if (h_keys)             free(h_keys);
    if (h_reference_keys)   free(h_reference_keys);
    if (h_values)           free(h_values);
    if (d_keys)             CubDebugExit(g_allocator.DeviceFree(d_keys));
    if (d_values)           CubDebugExit(g_allocator.DeviceFree(d_values));
}


/**
 * Test driver (valid tile size < 48KB)
 */
template <
    int                     CTA_THREADS,
    int                     ITEMS_PER_THREAD,
    int                     RADIX_BITS,
    cudaSharedMemConfig     SMEM_CONFIG,
    typename                KeyType,
    typename                ValueType,
    bool                    VALID = (CUB_MAX(sizeof(KeyType), sizeof(ValueType)) * CTA_THREADS * ITEMS_PER_THREAD < (1024 * 48))>
struct Valid
{
    static void Test()
    {
        // Iterate keys vs. key-value pairs
        for (unsigned int keys_only = 0; keys_only <= 1; keys_only++)
        {
            // Iterate entropy_reduction
            for (unsigned int entropy_reduction = 0; entropy_reduction <= 8; entropy_reduction += 2)
            {
                // Iterate begin_bit
                for (unsigned int begin_bit = 0; begin_bit <= 1; begin_bit++)
                {
                    // Iterate passes
                    for (unsigned int passes = 1; passes <= (sizeof(KeyType) * 8) / RADIX_BITS; passes++)
                    {
                        // Iterate relative_end
                        for (int relative_end = -1; relative_end <= 1; relative_end++)
                        {
                            int end_bit = begin_bit + (passes * RADIX_BITS) + relative_end;
                            if ((end_bit > begin_bit) && (end_bit <= sizeof(KeyType) * 8))
                            {
                                TestDriver<CTA_THREADS, ITEMS_PER_THREAD, RADIX_BITS, SMEM_CONFIG, KeyType, ValueType>(
                                    (bool) keys_only,
                                    entropy_reduction,
                                    begin_bit,
                                    end_bit);
                            }
                        }
                    }
                }
            }
        }
    }
};

/**
 * Test driver (invalid tile size)
 */
template <
    int                     CTA_THREADS,
    int                     ITEMS_PER_THREAD,
    int                     RADIX_BITS,
    cudaSharedMemConfig     SMEM_CONFIG,
    typename                KeyType,
    typename                ValueType>
struct Valid<CTA_THREADS, ITEMS_PER_THREAD, RADIX_BITS, SMEM_CONFIG, KeyType, ValueType, false>
{
    // Do nothing
    static void Test() {}
};


/**
 * Test value type
 */
template <
    int CTA_THREADS,
    int ITEMS_PER_THREAD,
    int RADIX_BITS,
    cudaSharedMemConfig SMEM_CONFIG,
    typename KeyType>
void Test()
{
    Valid<CTA_THREADS, ITEMS_PER_THREAD, RADIX_BITS, SMEM_CONFIG, KeyType, unsigned char>::Test();
    Valid<CTA_THREADS, ITEMS_PER_THREAD, RADIX_BITS, SMEM_CONFIG, KeyType, unsigned short>::Test();
    Valid<CTA_THREADS, ITEMS_PER_THREAD, RADIX_BITS, SMEM_CONFIG, KeyType, unsigned int>::Test();
    Valid<CTA_THREADS, ITEMS_PER_THREAD, RADIX_BITS, SMEM_CONFIG, KeyType, unsigned long long>::Test();
}


/**
 * Test key type
 */
template <
    int CTA_THREADS,
    int ITEMS_PER_THREAD,
    int RADIX_BITS,
    cudaSharedMemConfig SMEM_CONFIG>
void Test()
{
    Test<CTA_THREADS, ITEMS_PER_THREAD, RADIX_BITS, SMEM_CONFIG, unsigned char>();
    Test<CTA_THREADS, ITEMS_PER_THREAD, RADIX_BITS, SMEM_CONFIG, unsigned short>();
    Test<CTA_THREADS, ITEMS_PER_THREAD, RADIX_BITS, SMEM_CONFIG, unsigned int>();
    Test<CTA_THREADS, ITEMS_PER_THREAD, RADIX_BITS, SMEM_CONFIG, unsigned long long>();
}


/**
 * Test smem config
 */
template <
    int CTA_THREADS,
    int ITEMS_PER_THREAD,
    int RADIX_BITS>
void Test()
{
    Test<CTA_THREADS, ITEMS_PER_THREAD, RADIX_BITS, cudaSharedMemBankSizeFourByte>();
    Test<CTA_THREADS, ITEMS_PER_THREAD, RADIX_BITS, cudaSharedMemBankSizeEightByte>();
}


/**
 * Test radix bits
 */
template <
    int CTA_THREADS,
    int ITEMS_PER_THREAD>
void Test()
{
    Test<CTA_THREADS, ITEMS_PER_THREAD, 1>();
    Test<CTA_THREADS, ITEMS_PER_THREAD, 3>();
    Test<CTA_THREADS, ITEMS_PER_THREAD, 4>();
    Test<CTA_THREADS, ITEMS_PER_THREAD, 5>();
}


/**
 * Test items per thread
 */
template <int CTA_THREADS>
void Test()
{
    Test<CTA_THREADS, 1>();
    Test<CTA_THREADS, 8>();
    Test<CTA_THREADS, 15>();
    Test<CTA_THREADS, 19>();
}

/**
 * Test threads
 */
void Test()
{
    Test<32>();
    Test<64>();
    Test<128>();
    Test<256>();
}


/**
 * Main
 */
int main(int argc, char** argv)
{
    // Initialize command line
    CommandLineArgs args(argc, argv);
    g_verbose = args.CheckCmdLineFlag("v");
    bool quick = args.CheckCmdLineFlag("quick");

    // Print usage
    if (args.CheckCmdLineFlag("help"))
    {
        printf("%s [--device=<device-id>] [--v] [--quick]\n", argv[0]);
        exit(0);
    }

    // Initialize device
    CubDebugExit(args.DeviceInit());

    if (quick)
    {
        // Quick test
        TestDriver<64, 2, 5, cudaSharedMemBankSizeFourByte, unsigned int, unsigned int>(
            true,
            0,
            0,
            sizeof(unsigned int) * 8);
    }
    else
    {
        Test();
    }

    return 0;
}



