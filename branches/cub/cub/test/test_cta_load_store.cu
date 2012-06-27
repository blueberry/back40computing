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

#define CUB_STDERR

#include <stdio.h>
#include "../cub.cuh"
#include <test_util.h>

using namespace cub;

bool g_verbose = false;



/**
 * Test unguarded load/store kernel.
 */
template <
	int 			CTA_THREADS,
	int 			SEGMENTS,
	int 			ELEMENTS,
	LoadModifier 	LOAD_MODIFIER,
	StoreModifier 	STORE_MODIFIER,
	typename 		T>
__global__ void UnguardedKernel(T *d_in, T *d_out)
{
	typedef CtaLoad<CTA_THREADS, LOAD_MODIFIER> 	CtaLoad;
	typedef CtaStore<CTA_THREADS, STORE_MODIFIER> 	CtaStore;

	T data[SEGMENTS][ELEMENTS];

	// Load data
	CtaLoad::LoadUnguarded(data, d_in, 0);

	__syncthreads();

	// Store data
	CtaStore::StoreUnguarded(data, d_out, 0);
}


/**
 * Test guarded load/store kernel.
 */
template <
	int 			CTA_THREADS,
	int 			SEGMENTS,
	int 			ELEMENTS,
	LoadModifier 	LOAD_MODIFIER,
	StoreModifier 	STORE_MODIFIER,
	typename 		T>
__global__ void GuardedKernel(
	T *d_in,
	T *d_out,
	bool *d_flags,
	int num_elements)
{
	typedef CtaLoad<CTA_THREADS, LOAD_MODIFIER> 	CtaLoad;
	typedef CtaStore<CTA_THREADS, STORE_MODIFIER> 	CtaStore;

	T 		data[SEGMENTS][ELEMENTS];
	bool 	flags[SEGMENTS][ELEMENTS];

	// Load data
	CtaLoad::LoadGuarded(flags, data, d_in, 0, num_elements);

	__syncthreads();

	// Store data
	CtaStore::StoreGuarded(data, d_out, 0, num_elements);

	// Store flags
	CtaStore::StoreUnguarded(flags, d_flags, 0);
}



/**
 * Test unguarded load/store
 */
template <
	int 			CTA_THREADS,
	int 			SEGMENTS,
	int 			ELEMENTS,
	LoadModifier 	LOAD_MODIFIER,
	StoreModifier 	STORE_MODIFIER,
	typename 		T>
void TestUnguarded()
{
	// Allocate host arrays
	T h_in[CTA_THREADS];

	// Initialize problem (and solution)
	for (int i = 0; i < CTA_THREADS; ++i)
	{
		RandomBits(h_in[i]);
	}

	// Initialize device arrays
	T *d_in = NULL;
	T *d_out = NULL;
	DebugExit(cudaMalloc((void**)&d_in, sizeof(T) * CTA_THREADS));
	DebugExit(cudaMalloc((void**)&d_out, sizeof(T) * CTA_THREADS));
	DebugExit(cudaMemcpy(d_in, h_in, sizeof(T) * CTA_THREADS, cudaMemcpyHostToDevice));

	printf("Unguarded test "
		"CTA_THREADS(%d) "
		"SEGMENTS(%d) "
		"ELEMENTS(%d) "
		"LOAD_MODIFIER(%d) "
		"STORE_MODIFIER(%d) "
		"sizeof(T)(%d):\n\t ",
			CTA_THREADS, SEGMENTS, ELEMENTS, LOAD_MODIFIER, STORE_MODIFIER, (int) sizeof(T));

	// Run kernel
	UnguardedKernel<CTA_THREADS, SEGMENTS, ELEMENTS, LOAD_MODIFIER, STORE_MODIFIER>
		<<<1, CTA_THREADS>>>(d_in, d_out);

	DebugExit(cudaDeviceSynchronize());

	// Copy out and display results
	AssertEquals(0, CompareDeviceResults(h_in, d_out, CTA_THREADS, g_verbose, g_verbose));
	printf("\n");

	// Cleanup
	if (d_in) DebugExit(cudaFree(d_in));
	if (d_out) DebugExit(cudaFree(d_out));
}


/**
 * Test guarded load/store
 */
template <
	int 			CTA_THREADS,
	int 			SEGMENTS,
	int 			ELEMENTS,
	LoadModifier 	LOAD_MODIFIER,
	StoreModifier 	STORE_MODIFIER,
	typename 		T>
void TestGuarded(int num_elements)
{
	// Allocate host arrays
	T 		h_in[CTA_THREADS];
	bool 	h_flags[CTA_THREADS];

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

	DebugExit(cudaMalloc((void**)&d_in, sizeof(T) * CTA_THREADS));
	DebugExit(cudaMalloc((void**)&d_out, sizeof(T) * CTA_THREADS));
	DebugExit(cudaMalloc((void**)&d_flags, sizeof(bool) * CTA_THREADS));
	DebugExit(cudaMemcpy(d_in, h_in, sizeof(T) * CTA_THREADS, cudaMemcpyHostToDevice));

	printf("Guarded test "
		"CTA_THREADS(%d) "
		"SEGMENTS(%d) "
		"ELEMENTS(%d) "
		"LOAD_MODIFIER(%d) "
		"STORE_MODIFIER(%d) "
		"sizeof(T)(%d):\n\t",
			CTA_THREADS, SEGMENTS, ELEMENTS, LOAD_MODIFIER, STORE_MODIFIER, (int) sizeof(T));

	// Run kernel
	GuardedKernel<CTA_THREADS, SEGMENTS, ELEMENTS, LOAD_MODIFIER, STORE_MODIFIER>
		<<<1, CTA_THREADS>>>(d_in, d_out, d_flags, num_elements);

	DebugExit(cudaDeviceSynchronize());

	// Copy out and display results
	AssertEquals(0, CompareDeviceResults(h_in, d_out, num_elements, g_verbose, g_verbose));
	printf("\n\t");

	AssertEquals(0, CompareDeviceResults(h_flags, d_flags, CTA_THREADS, g_verbose, g_verbose));
	printf("\n");

	// Cleanup
	if (d_in) DebugExit(cudaFree(d_in));
	if (d_out) DebugExit(cudaFree(d_out));
	if (d_flags) DebugExit(cudaFree(d_flags));
}


/**
 * Main
 */
int main(int argc, char** argv)
{
    // Initialize command line
    CommandLineArgs args(argc, argv);
    DeviceInit(args);
    g_verbose = args.CheckCmdLineFlag("v");

    TestUnguarded<128, 2, 2, LOAD_NONE, STORE_NONE, int>();
    TestUnguarded<128, 2, 2, LOAD_NONE, STORE_NONE, double>();
    TestUnguarded<128, 2, 2, LOAD_NONE, STORE_NONE, char>();
    TestUnguarded<128, 2, 2, LOAD_CG, STORE_CS, int>();

    TestUnguarded<128, 2, 3, LOAD_NONE, STORE_NONE, int>();
    TestUnguarded<128, 2, 8, LOAD_NONE, STORE_NONE, int>();
    TestUnguarded<128, 2, 9, LOAD_NONE, STORE_NONE, int>();

    TestGuarded<128, 1, 1, LOAD_NONE, STORE_NONE, int>(56);
    TestGuarded<128, 1, 1, LOAD_CS, STORE_WT, float>(127);
    TestGuarded<128, 2, 2, LOAD_NONE, STORE_NONE, int>(56);
    TestGuarded<128, 2, 2, LOAD_CS, STORE_WT, float>(127);

	return 0;
}



