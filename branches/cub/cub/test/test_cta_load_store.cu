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
__global__ void UnguardedKernel(
	InputIterator d_in,
	OutputIterator d_out)
{
	// Data type of input/output iterators
	typedef typename std::iterator_traits<InputIterator>::value_type T;

	typedef CtaLoad<
		InputIterator,
		CTA_THREADS,
		ITEMS_PER_THREAD,
		LOAD_POLICY,
		LOAD_MODIFIER> CtaLoad;

	typedef CtaStore<
		OutputIterator,
		CTA_THREADS,
		ITEMS_PER_THREAD,
		STORE_POLICY,
		STORE_MODIFIER> CtaStore;

	union SmemStorage
	{
		typename CtaLoad::SmemStorage 	load;
		typename CtaStore::SmemStorage 	store;
	};

	__shared__ SmemStorage smem_storage;

	T data[ITEMS_PER_THREAD];

	// Load data
	CtaLoad::Load(smem_storage.load, data, d_in, 0);

	__syncthreads();

	// Store data
	CtaStore::Store(smem_storage.store, data, d_out, 0);
}


/**
 * Test guarded-by-range load/store kernel.
 * /
template <
	int 			CTA_THREADS,
	int 			ITEMS_PER_THREAD,
	CtaLoadPolicy 	LOAD_POLICY,
	CtaStorePolicy 	STORE_POLICY,
	LoadModifier 	LOAD_MODIFIER,
	StoreModifier 	STORE_MODIFIER,
	typename 		InputIterator,
	typename 		OutputIterator>
__global__ void GuardedRangeKernel(
	T d_in,
	T d_out,
	int num_elements)
{
	// Data type of input/output iterators
	typedef typename std::iterator_traits<InputIterator>::value_type T;

	typedef CtaLoad<
		InputIterator,
		CTA_THREADS,
		ITEMS_PER_THREAD,
		LOAD_POLICY,
		LOAD_MODIFIER>			CtaLoad;

	typedef CtaStore<
		OutputIterator,
		CTA_THREADS,
		ITEMS_PER_THREAD,
		STORE_POLICY,
		STORE_MODIFIER>			CtaStore;

	union SmemStorage
	{
		typename CtaLoad::SmemStorage 	load;
		typename CtaStore::SmemStorage 	store;
	};

	__shared__ SmemStorage smem_storage;

	T data[ITEMS_PER_THREAD];

	// Load data
	CtaLoad::Load(smem_storage.load, data, d_in, 0, num_elements);

	__syncthreads();

	// Store data
	CtaStore::Store(smem_storage.store, data, d_out, 0, num_elements);
}


/* *
 * Test guarded-by-flag load/store kernel.
 * /
template <
	int 			CTA_THREADS,
	int 			ITEMS_PER_THREAD,
	CtaLoadPolicy 	LOAD_POLICY,
	CtaStorePolicy 	STORE_POLICY,
	LoadModifier 	LOAD_MODIFIER,
	StoreModifier 	STORE_MODIFIER,
	typename 		InputIterator,
	typename 		OutputIterator,
	typename		Flag>
__global__ void GuardedFlagKernel(
	InputIterator 	d_in,
	OutputIterator 	d_out,
	Flag 			*d_flags)
{
	typedef CtaLoad<
		InputIterator,
		CTA_THREADS,
		ITEMS_PER_THREAD,
		LOAD_POLICY,
		LOAD_MODIFIER>			CtaLoad;

	typedef CtaLoad<
		Flag*,
		CTA_THREADS,
		ITEMS_PER_THREAD,
		LOAD_POLICY,
		LOAD_MODIFIER>			CtaLoadFlags;

	typedef CtaStore<
		OutputIterator,
		CTA_THREADS,
		ITEMS_PER_THREAD,
		STORE_POLICY,
		STORE_MODIFIER>			CtaStore;

	union SmemStorage
	{
		typename CtaLoad::SmemStorage 		load;
		typename CtaLoadFlags::SmemStorage 	load_flags;
		typename CtaStore::SmemStorage 		store;
	};

	__shared__ SmemStorage smem_storage;

	T 		data[ITEMS_PER_THREAD];
	Flag 	flags[ITEMS_PER_THREAD];

	// Load flags
	CtaLoad::Load(smem_storage.load_flags, flags, d_flags, 0);

	__syncthreads();

	// Load data
	CtaLoad::Load(smem_storage.load, data, d_in, 0, flags);

	__syncthreads();

	// Store data
	CtaStore::Store(smem_storage.store, data, d_out, 0, flags);
}


//---------------------------------------------------------------------
// Host testing subroutines
//---------------------------------------------------------------------

*/

/**
 * Test unguarded load/store
 */
template <
	typename 		T,
	int 			CTA_THREADS,
	int 			ITEMS_PER_THREAD,
	CtaLoadPolicy 	LOAD_POLICY,
	CtaStorePolicy 	STORE_POLICY,
	LoadModifier 	LOAD_MODIFIER,
	StoreModifier 	STORE_MODIFIER>
void TestUnguardedKernel()
{
	const int TILE_SIZE = CTA_THREADS * ITEMS_PER_THREAD;

	// Allocate host arrays
	T h_in[TILE_SIZE];

	// Initialize problem (and solution)
	for (int i = 0; i < TILE_SIZE; ++i)
	{
		RandomBits(h_in[i]);
//		h_in[i] = i;
	}

	// Initialize device arrays
	T *d_in = NULL;
	T *d_out = NULL;
	CubDebugExit(cudaMalloc((void**)&d_in, sizeof(T) * TILE_SIZE));
	CubDebugExit(cudaMalloc((void**)&d_out, sizeof(T) * TILE_SIZE));
	CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(T) * TILE_SIZE, cudaMemcpyHostToDevice));

	printf("TestUnguardedKernel "
		"CTA_THREADS(%d) "
		"ITEMS_PER_THREAD(%d) "
		"LOAD_POLICY(%d) "
		"STORE_POLICY(%d) "
		"LOAD_MODIFIER(%d) "
		"STORE_MODIFIER(%d) "
		"sizeof(T)(%d):\n\t ",
			CTA_THREADS, ITEMS_PER_THREAD, LOAD_POLICY, STORE_POLICY, LOAD_MODIFIER, STORE_MODIFIER, (int) sizeof(T));

	// Run kernel
	UnguardedKernel<
		CTA_THREADS,
		ITEMS_PER_THREAD,
		LOAD_POLICY,
		STORE_POLICY,
		LOAD_MODIFIER,
		STORE_MODIFIER>
			<<<1, CTA_THREADS>>>(d_in, d_out);

	CubDebugExit(cudaDeviceSynchronize());

	// Copy out and display results
	AssertEquals(0, CompareDeviceResults(h_in, d_out, TILE_SIZE, g_verbose, g_verbose));
	printf("\n");

	// Cleanup
	if (d_in) CubDebugExit(cudaFree(d_in));
	if (d_out) CubDebugExit(cudaFree(d_out));
}


/**
 * Test guarded load/store
 * /
template <
	typename 		T,
	int 			CTA_THREADS,
	int 			ITEMS_PER_THREAD,
	CtaLoadPolicy 	LOAD_POLICY,
	CtaStorePolicy 	STORE_POLICY,
	LoadModifier 	LOAD_MODIFIER,
	StoreModifier 	STORE_MODIFIER>
void TestGuardedRangeKernel(int num_elements)
{
	const int TILE_SIZE = CTA_THREADS * ITEMS_PER_THREAD;

	// Allocate host arrays
	T h_in[TILE_SIZE];

	// Initialize problem (and solution)
	for (int i = 0; i < TILE_SIZE; ++i)
	{
		RandomBits(h_in[i]);
	}

	// Initialize device arrays
	T 		*d_in = NULL;
	T 		*d_out = NULL;

	CubDebugExit(cudaMalloc((void**)&d_in, sizeof(T) * TILE_SIZE));
	CubDebugExit(cudaMalloc((void**)&d_out, sizeof(T) * TILE_SIZE));
	CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(T) * TILE_SIZE, cudaMemcpyHostToDevice));

	printf("TestGuardedRangeKernel "
		"num_elements(%d) "
		"CTA_THREADS(%d) "
		"ITEMS_PER_THREAD(%d) "
		"LOAD_POLICY(%d) "
		"STORE_POLICY(%d) "
		"LOAD_MODIFIER(%d) "
		"STORE_MODIFIER(%d) "
		"sizeof(T)(%d):\n\t ",
			num_elements, CTA_THREADS, ITEMS_PER_THREAD, LOAD_POLICY, STORE_POLICY, LOAD_MODIFIER, STORE_MODIFIER, (int) sizeof(T));

	// Run kernel
	GuardedRangeKernel<
		CTA_THREADS,
		ITEMS_PER_THREAD,
		LOAD_POLICY,
		STORE_POLICY,
		LOAD_MODIFIER,
		STORE_MODIFIER>
			<<<1, CTA_THREADS>>>(d_in, d_out, num_elements);

	CubDebugExit(cudaDeviceSynchronize());

	// Copy out and display results
	AssertEquals(0, CompareDeviceResults(h_in, d_out, num_elements, g_verbose, g_verbose));
	printf("\n\t");

	// Cleanup
	if (d_in) CubDebugExit(cudaFree(d_in));
	if (d_out) CubDebugExit(cudaFree(d_out));
}


/* *
 * Test guarded load/store
 * /
template <
	typename 		T,
	int 			CTA_THREADS,
	int 			ITEMS_PER_THREAD,
	CtaLoadPolicy 	LOAD_POLICY,
	CtaStorePolicy 	STORE_POLICY,
	LoadModifier 	LOAD_MODIFIER,
	StoreModifier 	STORE_MODIFIER>
void TestGuardedFlagKernel(int num_elements)
{
	const int TILE_SIZE = CTA_THREADS * ITEMS_PER_THREAD;

	// Allocate host arrays
	T 		h_in[TILE_SIZE];
	bool 	h_flags[TILE_SIZE];

	// Initialize problem (and solution)
	for (int i = 0; i < CTA_THREADS; ++i)
	{
		RandomBits(h_in[i]);
		h_flags[i] = (i < num_elements);
	}

	// Initialize device arrays
	T 		*d_in = NULL;
	T 		*d_out = NULL;
	bool 	*d_flags = NULL;

	CubDebugExit(cudaMalloc((void**)&d_in, sizeof(T) * TILE_SIZE));
	CubDebugExit(cudaMalloc((void**)&d_out, sizeof(T) * TILE_SIZE));
	CubDebugExit(cudaMalloc((void**)&d_flags, sizeof(bool) * TILE_SIZE));
	CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(T) * TILE_SIZE, cudaMemcpyHostToDevice));

	printf("TestGuardedFlagKernel "
		"num_elements(%d) "
		"CTA_THREADS(%d) "
		"ITEMS_PER_THREAD(%d) "
		"LOAD_POLICY(%d) "
		"STORE_POLICY(%d) "
		"LOAD_MODIFIER(%d) "
		"STORE_MODIFIER(%d) "
		"sizeof(T)(%d):\n\t ",
			num_elements, CTA_THREADS, ITEMS_PER_THREAD, LOAD_POLICY, STORE_POLICY, LOAD_MODIFIER, STORE_MODIFIER, (int) sizeof(T));

	// Run kernel
	GuardedFlagKernel<
		CTA_THREADS,
		ITEMS_PER_THREAD,
		LOAD_POLICY,
		STORE_POLICY,
		LOAD_MODIFIER,
		STORE_MODIFIER>
			<<<1, CTA_THREADS>>>(d_in, d_out, d_flags);

	CubDebugExit(cudaDeviceSynchronize());

	// Copy out and display results
	AssertEquals(0, CompareDeviceResults(h_in, d_out, num_elements, g_verbose, g_verbose));
	printf("\n\t");

	// Cleanup
	if (d_in) CubDebugExit(cudaFree(d_in));
	if (d_out) CubDebugExit(cudaFree(d_out));
	if (d_flags) CubDebugExit(cudaFree(d_flags));
}
*/


/**
 * Evaluate different LoadTile variants
 */
template <
	typename 		T,
	int 			CTA_THREADS,
	int 			ITEMS_PER_THREAD,
	CtaLoadPolicy 	LOAD_POLICY,
	CtaStorePolicy 	STORE_POLICY,
	LoadModifier 	LOAD_MODIFIER,
	StoreModifier 	STORE_MODIFIER>
void TestVariants()
{
    TestUnguardedKernel<T, CTA_THREADS, ITEMS_PER_THREAD, LOAD_POLICY, STORE_POLICY, LOAD_MODIFIER, STORE_MODIFIER>();
}


/**
 * Evaluate different cache modifiers
 */
template <
	typename 		T,
	int 			CTA_THREADS,
	int 			ITEMS_PER_THREAD,
	CtaLoadPolicy 	LOAD_POLICY,
	CtaStorePolicy 	STORE_POLICY>
void TestModifiers()
{
	TestVariants<T, CTA_THREADS, ITEMS_PER_THREAD, LOAD_POLICY, STORE_POLICY, LOAD_NONE, STORE_NONE>();
	TestVariants<T, CTA_THREADS, ITEMS_PER_THREAD, LOAD_POLICY, STORE_POLICY, LOAD_CG, STORE_CG>();
	TestVariants<T, CTA_THREADS, ITEMS_PER_THREAD, LOAD_POLICY, STORE_POLICY, LOAD_CS, STORE_CS>();
}


/**
 * Evaluate different load/store strategies
 */
template <
	typename 		T,
	int 			CTA_THREADS,
	int 			ITEMS_PER_THREAD>
void TestStrategy()
{
	TestModifiers<T, CTA_THREADS, ITEMS_PER_THREAD, LOAD_TILE_DIRECT, STORE_TILE_DIRECT>();
	TestModifiers<T, CTA_THREADS, ITEMS_PER_THREAD, LOAD_TILE_TRANSPOSE, STORE_TILE_TRANSPOSE>();
	TestModifiers<T, CTA_THREADS, ITEMS_PER_THREAD, LOAD_TILE_VECTORIZED, STORE_TILE_VECTORIZED>();
}


/**
 * Evaluate different register blocking
 */
template <
	typename T,
	int CTA_THREADS>
void TestItemsPerThread()
{
	TestStrategy<T, CTA_THREADS, 1>();
	TestStrategy<T, CTA_THREADS, 3>();
	TestStrategy<T, CTA_THREADS, 4>();
	TestStrategy<T, CTA_THREADS, 8>();
}


/**
 * Evaluate different CTA sizes
 */
template <typename T>
void TestThreads()
{
	TestItemsPerThread<T, 15>();
	TestItemsPerThread<T, 32>();
	TestItemsPerThread<T, 96>();
	TestItemsPerThread<T, 128>();
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
    TestThreads<int>();




    return 0;
}



