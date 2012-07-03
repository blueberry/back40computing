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
 * Test evaluation for caching allocator of device memory
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <stdio.h>

#include <stdio.h>
#include "../cub.cuh"
#include <test_util.h>

using namespace cub;


//---------------------------------------------------------------------
// Main
//---------------------------------------------------------------------

/**
 * Main
 */
int main(int argc, char** argv)
{
    // Initialize command line
    CommandLineArgs args(argc, argv);

    // Print usage
    if (args.CheckCmdLineFlag("help"))
    {
    	printf("%s "
    		"[--device=<device-id>]"
    		"\n", argv[0]);
    	exit(0);
    }

    // Initialize device
    CubDebugExit(args.DeviceInit());

    // Get number of GPUs and current GPU
    int num_gpus, initial_gpu;
	if (CubDebug(cudaGetDeviceCount(&num_gpus))) exit(1);
	if (CubDebug(cudaGetDevice(&initial_gpu))) exit(1);

	// Create default allocator (caches up to 6MB in device allocations per GPU)
    CachedAllocator allocator;

	printf("Running single-gpu tests...\n"); fflush(stdout);

	//
    // Test1
    //

    // Allocate 5 bytes on the current gpu
    char *d_5B;
    allocator.Allocate((void **) &d_5B, 5);

    // Check that that we have zero bytes allocated on the initial GPU
    AssertEquals(allocator.cached_bytes[initial_gpu], 0);

    // Check that that we have 1 live block on the initial GPU
    AssertEquals(allocator.live_blocks.size(), 1);

    //
    // Test2
    //

    // Allocate 4096 bytes on the current gpu
    char *d_4096B;
    allocator.Allocate((void **) &d_4096B, 4096);

    // Check that that we have 2 live blocks on the initial GPU
    AssertEquals(allocator.live_blocks.size(), 2);

    //
    // Test3
    //

    // Deallocate d_5B
    allocator.Deallocate(d_5B);

    // Check that that we have min_bin_bytes free bytes cached on the initial gpu
    AssertEquals(allocator.cached_bytes[initial_gpu], allocator.min_bin_bytes);

    // Check that that we have 1 live block on the initial GPU
    AssertEquals(allocator.live_blocks.size(), 1);

    // Check that that we have 1 cached block on the initial GPU
    AssertEquals(allocator.cached_blocks.size(), 1);

    //
    // Test4
    //

    // Deallocate d_4096B
    allocator.Deallocate(d_4096B);

    // Check that that we have the 4096 + min_bin free bytes cached on the initial gpu
    AssertEquals(allocator.cached_bytes[initial_gpu], allocator.min_bin_bytes + 4096);

    // Check that that we have 0 live block on the initial GPU
    AssertEquals(allocator.live_blocks.size(), 0);

    // Check that that we have 2 cached block on the initial GPU
    AssertEquals(allocator.cached_blocks.size(), 2);

    //
    // Test5
    //

    // Allocate 768 bytes on the current gpu
    char *d_768B;
    allocator.Allocate((void **) &d_768B, 768);

    // Check that that we have the min_bin free bytes cached on the initial gpu (4096 was reused)
    AssertEquals(allocator.cached_bytes[initial_gpu], allocator.min_bin_bytes);

    // Check that that we have 1 live block on the initial GPU
    AssertEquals(allocator.live_blocks.size(), 1);

    // Check that that we have 1 cached block on the initial GPU
    AssertEquals(allocator.cached_blocks.size(), 1);

    //
    // Test6
    //

    // Allocate max_cached_bytes on the current gpu
    char *d_max_cached;
    allocator.Allocate((void **) &d_max_cached, allocator.max_cached_bytes);

    // Deallocate d_max_cached
    allocator.Deallocate(d_max_cached);

    // Check that that we have the min_bin free bytes cached on the initial gpu (max cached was not returned because we went over)
    AssertEquals(allocator.cached_bytes[initial_gpu], allocator.min_bin_bytes);

    // Check that that we have 1 live block on the initial GPU
    AssertEquals(allocator.live_blocks.size(), 1);

    // Check that that we still have 1 cached block on the initial GPU
    AssertEquals(allocator.cached_blocks.size(), 1);

    //
    // Test7
    //

    // Free all cached blocks on all GPUs
    allocator.FreeAllCached();

    // Check that that we have 0 bytes cached on the initial GPU
    AssertEquals(allocator.cached_bytes[initial_gpu], 0);

    // Check that that we have 0 cached blocks across all GPUs
    AssertEquals(allocator.cached_blocks.size(), 0);

    // Check that that still we have 1 live block across all GPUs
    AssertEquals(allocator.live_blocks.size(), 1);

    //
    // Test8
    //

    // Allocate max cached bytes on the current gpu
    allocator.Allocate((void **) &d_max_cached, allocator.max_cached_bytes);

    // Deallocate max cached bytes
    allocator.Deallocate(d_max_cached);

    // Deallocate d_768B
    allocator.Deallocate(d_768B);

    unsigned int power;
    size_t rounded_bytes;
    allocator.NearestPowerOf(power, rounded_bytes, allocator.bin_growth, 768);

    // Check that that we have 4096 free bytes cached on the initial gpu
    AssertEquals(allocator.cached_bytes[initial_gpu], rounded_bytes);

    // Check that that we have 1 cached blocks across all GPUs
    AssertEquals(allocator.cached_blocks.size(), 1);

    // Check that that still we have 0 live block across all GPUs
    AssertEquals(allocator.live_blocks.size(), 0);

    if (num_gpus > 1)
    {
    	printf("Running multi-gpu tests...\n"); fflush(stdout);

        //
        // Test9
        //

    	// Allocate 768 bytes on the next gpu
		int next_gpu = (initial_gpu + 1) % num_gpus;
		char *d_768B_2;
		allocator.Allocate((void **) &d_768B_2, 768, next_gpu);

		// Deallocate d_768B on the next gpu
		allocator.Deallocate(d_768B_2, next_gpu);

		// Check that that we have 4096 free bytes cached on the initial gpu
		AssertEquals(allocator.cached_bytes[initial_gpu], rounded_bytes);

		// Check that that we have 4096 free bytes cached on the second gpu
		AssertEquals(allocator.cached_bytes[next_gpu], rounded_bytes);

	    // Check that that we have 2 cached blocks across all GPUs
	    AssertEquals(allocator.cached_blocks.size(), 2);

	    // Check that that still we have 0 live block across all GPUs
	    AssertEquals(allocator.live_blocks.size(), 0);
    }

    printf("Success\n");
    return 0;
}

