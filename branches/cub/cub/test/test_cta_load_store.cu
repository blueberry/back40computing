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
 * Test of CtaLoad and CtaStore utilities
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <iterator>
#include <stdio.h>
#include <test_util.h>

#include <thrust/iterator/counting_iterator.h>

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
 * Test unguarded load/store kernel.
 */
template <
	int 			CTA_THREADS,
	int 			ITEMS_PER_THREAD,
	CtaLoadPolicy 	LOAD_POLICY,
	CtaStorePolicy 	STORE_POLICY,
	LoadModifier 	LOAD_MODIFIER,
	StoreModifier 	STORE_MODIFIER,
	typename 		InputIterator,
	typename 		OutputIterator>
__launch_bounds__ (CTA_THREADS, 1)
__global__ void Kernel(
	InputIterator d_in,
	OutputIterator d_out_unguarded,
	OutputIterator d_out_guarded_range,
	int num_elements)
{
	enum
	{
		TILE_SIZE = CTA_THREADS * ITEMS_PER_THREAD
	};

	// Data type of input/output iterators
	typedef typename std::iterator_traits<InputIterator>::value_type T;

	// CTA load/store abstraction types
	typedef CtaLoad<InputIterator, CTA_THREADS, ITEMS_PER_THREAD, LOAD_POLICY, LOAD_MODIFIER> CtaLoad;
	typedef CtaStore<OutputIterator, CTA_THREADS, ITEMS_PER_THREAD, STORE_POLICY, STORE_MODIFIER> CtaStore;

	// Shared memory type for this CTA
	union SmemStorage
	{
		typename CtaLoad::SmemStorage 	load;
		typename CtaStore::SmemStorage 	store;
	};

	// Shared memory
	__shared__ SmemStorage smem_storage;

	// CTA work bounds
	int cta_offset = blockIdx.x * TILE_SIZE;
	int guarded_elements = num_elements - cta_offset;

	// Test unguarded
	{
		// Tile of items
		T data[ITEMS_PER_THREAD];

		// Load data
		CtaLoad::Load(smem_storage.load, data, d_in, cta_offset);

		__syncthreads();

		// Store data
		CtaStore::Store(smem_storage.store, data, d_out_unguarded, cta_offset);
	}

	__syncthreads();

	// Test guarded by range
	{
		// Tile of items
		T data[ITEMS_PER_THREAD];

		// Load data
		CtaLoad::Load(smem_storage.load, data, d_in, cta_offset, guarded_elements);

		__syncthreads();

		// Store data
		CtaStore::Store(smem_storage.store, data, d_out_guarded_range, cta_offset, guarded_elements);
	}
}


//---------------------------------------------------------------------
// Host testing subroutines
//---------------------------------------------------------------------


/**
 * Test load/store variants
 */
template <
	typename 		T,
	int 			CTA_THREADS,
	int 			ITEMS_PER_THREAD,
	CtaLoadPolicy 	LOAD_POLICY,
	CtaStorePolicy 	STORE_POLICY,
	LoadModifier 	LOAD_MODIFIER,
	StoreModifier 	STORE_MODIFIER,
	typename 		InputIterator,
	typename 		OutputIterator>
void TestKernel(
	T 				*h_in,
	InputIterator 	d_in,
	OutputIterator 	d_out_unguarded,
	OutputIterator 	d_out_guarded_range,
	int 			grid_size,
	int 			guarded_elements)
{
	int unguarded_elements = grid_size * CTA_THREADS * ITEMS_PER_THREAD;

	// Run kernel
	Kernel<CTA_THREADS, ITEMS_PER_THREAD, LOAD_POLICY, STORE_POLICY, LOAD_MODIFIER, STORE_MODIFIER>
		<<<grid_size, CTA_THREADS>>>(
			d_in,
			d_out_unguarded,
			d_out_guarded_range,
			guarded_elements);

	CubDebugExit(cudaDeviceSynchronize());

	// Check results
	printf("\tUnguarded: ");
	AssertEquals(0, CompareDeviceResults(h_in, d_out_unguarded, unguarded_elements, g_verbose, g_verbose));
	printf("\n");

	printf("\tGuarded range: ");
	AssertEquals(0, CompareDeviceResults(h_in, d_out_guarded_range, guarded_elements, g_verbose, g_verbose));
	printf("\n");
}


/**
 * Test native pointer
 */
template <
	typename 		T,
	int 			CTA_THREADS,
	int 			ITEMS_PER_THREAD,
	CtaLoadPolicy 	LOAD_POLICY,
	CtaStorePolicy 	STORE_POLICY,
	LoadModifier 	LOAD_MODIFIER,
	StoreModifier 	STORE_MODIFIER>
void TestNative(
	int grid_size,
	float fraction_valid)
{
	int unguarded_elements = grid_size * CTA_THREADS * ITEMS_PER_THREAD;
	int guarded_elements = int(fraction_valid * float(unguarded_elements));

	// Allocate host arrays
	T *h_in = (T*) malloc(unguarded_elements * sizeof(T));

	// Allocate device arrays
	T *d_in = NULL;
	T *d_out_unguarded = NULL;
	T *d_out_guarded_range = NULL;
	CubDebugExit(cudaMalloc((void**)&d_in, sizeof(T) * unguarded_elements));
	CubDebugExit(cudaMalloc((void**)&d_out_unguarded, sizeof(T) * unguarded_elements));
	CubDebugExit(cudaMalloc((void**)&d_out_guarded_range, sizeof(T) * guarded_elements));

	// Initialize problem on host and device
	for (int i = 0; i < unguarded_elements; ++i)
	{
		RandomBits(h_in[i]);
	}
	CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(T) * unguarded_elements, cudaMemcpyHostToDevice));

	printf("TestNative "
		"grid_size(%d) "
		"guarded_elements(%d) "
		"unguarded_elements(%d) "
		"CTA_THREADS(%d) "
		"ITEMS_PER_THREAD(%d) "
		"LOAD_POLICY(%d) "
		"STORE_POLICY(%d) "
		"LOAD_MODIFIER(%d) "
		"STORE_MODIFIER(%d) "
		"sizeof(T)(%d)\n",
			grid_size, guarded_elements, unguarded_elements, CTA_THREADS, ITEMS_PER_THREAD, LOAD_POLICY, STORE_POLICY, LOAD_MODIFIER, STORE_MODIFIER, (int) sizeof(T));

	TestKernel<T, CTA_THREADS, ITEMS_PER_THREAD, LOAD_POLICY, STORE_POLICY, LOAD_MODIFIER, STORE_MODIFIER>(
		h_in,
		d_in,
		d_out_unguarded,
		d_out_guarded_range,
		grid_size,
		guarded_elements);

	// Cleanup
	if (h_in) free(h_in);
	if (d_in) CubDebugExit(cudaFree(d_in));
	if (d_out_unguarded) CubDebugExit(cudaFree(d_out_unguarded));
	if (d_out_guarded_range) CubDebugExit(cudaFree(d_out_guarded_range));
}


/**
 * Test iterator
 */
template <
	typename 		T,
	int 			CTA_THREADS,
	int 			ITEMS_PER_THREAD,
	CtaLoadPolicy 	LOAD_POLICY,
	CtaStorePolicy 	STORE_POLICY>
void TestIterator(
	int grid_size,
	float fraction_valid)
{
	int unguarded_elements = grid_size * CTA_THREADS * ITEMS_PER_THREAD;
	int guarded_elements = int(fraction_valid * float(unguarded_elements));

	// Allocate host arrays
	T *h_in = (T*) malloc(unguarded_elements * sizeof(T));

	// Allocate device arrays
	T *d_out_unguarded = NULL;
	T *d_out_guarded_range = NULL;
	CubDebugExit(cudaMalloc((void**)&d_out_unguarded, sizeof(T) * unguarded_elements));
	CubDebugExit(cudaMalloc((void**)&d_out_guarded_range, sizeof(T) * guarded_elements));

	// Initialize problem on host and device
	thrust::counting_iterator<T> counting_itr(0);
	for (int i = 0; i < unguarded_elements; ++i)
	{
		h_in[i] = counting_itr[i];
	}

	printf("TestIterator "
		"grid_size(%d) "
		"guarded_elements(%d) "
		"unguarded_elements(%d) "
		"CTA_THREADS(%d) "
		"ITEMS_PER_THREAD(%d) "
		"LOAD_POLICY(%d) "
		"STORE_POLICY(%d) "
		"sizeof(T)(%d)\n",
			grid_size, guarded_elements, unguarded_elements, CTA_THREADS, ITEMS_PER_THREAD, LOAD_POLICY, STORE_POLICY, (int) sizeof(T));

	TestKernel<T, CTA_THREADS, ITEMS_PER_THREAD, LOAD_POLICY, STORE_POLICY, LOAD_NONE, STORE_NONE>(
		h_in,
		counting_itr,
		d_out_unguarded,
		d_out_guarded_range,
		grid_size,
		guarded_elements);

	// Cleanup
	if (h_in) free(h_in);
	if (d_out_unguarded) CubDebugExit(cudaFree(d_out_unguarded));
	if (d_out_guarded_range) CubDebugExit(cudaFree(d_out_guarded_range));
}


/**
 * Evaluate different pointer access types
 */
template <
	typename 		T,
	int 			CTA_THREADS,
	int 			ITEMS_PER_THREAD,
	CtaLoadPolicy 	LOAD_POLICY,
	CtaStorePolicy 	STORE_POLICY>
void TestPointerAccess(
	int grid_size,
	float fraction_valid)
{
    TestNative<T, CTA_THREADS, ITEMS_PER_THREAD, LOAD_POLICY, STORE_POLICY, LOAD_NONE, STORE_NONE>(grid_size, fraction_valid);
    TestNative<T, CTA_THREADS, ITEMS_PER_THREAD, LOAD_POLICY, STORE_POLICY, LOAD_CG, STORE_CG>(grid_size, fraction_valid);
    TestNative<T, CTA_THREADS, ITEMS_PER_THREAD, LOAD_POLICY, STORE_POLICY, LOAD_CS, STORE_CS>(grid_size, fraction_valid);
    TestIterator<T, CTA_THREADS, ITEMS_PER_THREAD, LOAD_POLICY, STORE_POLICY>(grid_size, fraction_valid);
}


/**
 * Evaluate different load/store strategies
 */
template <
	typename 		T,
	int 			CTA_THREADS,
	int 			ITEMS_PER_THREAD>
void TestStrategy(
	int grid_size,
	float fraction_valid)
{
	TestPointerAccess<T, CTA_THREADS, ITEMS_PER_THREAD, LOAD_TILE_DIRECT, STORE_TILE_DIRECT>(grid_size, fraction_valid);
	TestPointerAccess<T, CTA_THREADS, ITEMS_PER_THREAD, LOAD_TILE_TRANSPOSE, STORE_TILE_TRANSPOSE>(grid_size, fraction_valid);
	TestPointerAccess<T, CTA_THREADS, ITEMS_PER_THREAD, LOAD_TILE_VECTORIZED, STORE_TILE_VECTORIZED>(grid_size, fraction_valid);
}


/**
 * Evaluate different register blocking
 */
template <
	typename T,
	int CTA_THREADS>
void TestItemsPerThread(
	int grid_size,
	float fraction_valid)
{
	TestStrategy<T, CTA_THREADS, 1>(grid_size, fraction_valid);
	TestStrategy<T, CTA_THREADS, 3>(grid_size, fraction_valid);
	TestStrategy<T, CTA_THREADS, 4>(grid_size, fraction_valid);
	TestStrategy<T, CTA_THREADS, 8>(grid_size, fraction_valid);
}


/**
 * Evaluate different CTA sizes
 */
template <typename T>
void TestThreads(
	int grid_size,
	float fraction_valid)
{
	TestItemsPerThread<T, 15>(grid_size, fraction_valid);
	TestItemsPerThread<T, 32>(grid_size, fraction_valid);
	TestItemsPerThread<T, 96>(grid_size, fraction_valid);
	TestItemsPerThread<T, 128>(grid_size, fraction_valid);
}


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

    // Evaluate different data types
    TestThreads<int>(2, 0.8);

    return 0;
}



