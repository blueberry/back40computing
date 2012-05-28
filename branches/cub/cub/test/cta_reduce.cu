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
 * Test of CtaReduce utilities
 ******************************************************************************/

#define CUB_STDERR

#include <stdio.h>
#include <cub/cub.cuh>
#include <test_util.h>

using namespace cub;

bool g_verbose = false;


//---------------------------------------------------------------------
// Kernels
//---------------------------------------------------------------------

/**
 * Test unguarded load/store kernel.
 */
template <
	int CTA_THREADS,
	int CTA_STRIDES,
	typename T,
	typename ReductionOp>
__global__ void UnguardedKernel(
	T *d_in,
	T *d_out,
	ReductionOp reduction_op,
	int iterations)
{
	typedef CtaLoad<CTA_THREADS> CtaLoad;
	typedef CtaReduce<CTA_THREADS, T, CTA_STRIDES> CtaReduce;

	__shared__ typename CtaReduce::SmemStorage smem_storage;

	// Data
	T data[CTA_STRIDES][1];

	// Load data
	int cta_offset = 0;
	CtaLoad::LoadUnguarded(data, d_in, cta_offset);
	cta_offset += CTA_THREADS;

	// Cooperative reduce
	T partial = CtaReduce::Reduce(smem_storage, data, reduction_op);

	while (cta_offset < CTA_THREADS * iterations)
	{
		// Load data
		T next = d_in[cta_offset + threadIdx.x];
		cta_offset += CTA_THREADS;

		// Cooperative reduce
		next = CtaReduce::Reduce(smem_storage, data, reduction_op);
		partial = reduction_op(partial, next);
	}

	// Store data
	if (threadIdx.x == 0)
	{
		d_out[0] = partial;
	}
}


/**
 * Test guarded load/store kernel.
 */
template <int CTA_THREADS, typename T, typename ReductionOp>
__global__ void GuardedKernel(
	T *d_in,
	T *d_out,
	int num_elements,
	ReductionOp reduction_op)
{
	typedef CtaLoad<CTA_THREADS> CtaLoad;
	typedef CtaReduce<CTA_THREADS, T> CtaReduce;

	__shared__ typename CtaReduce::SmemStorage smem_storage;

	T partial;

	// Load data
	if (threadIdx.x < num_elements) {
		partial = d_in[threadIdx.x];
	}

	// Cooperative reduce
	partial = CtaReduce::Reduce(smem_storage, partial, num_elements, reduction_op);

	// Store data
	if (threadIdx.x == 0)
	{
		d_out[0] = partial;
	}
}


//---------------------------------------------------------------------
// Test routines
//---------------------------------------------------------------------

/**
 * Uint2 reduction operator
 */
struct Uint2Sum
{
	__host__ __device__ __forceinline__ uint2 operator()(uint2 a, uint2 b)
	{
		a.x += b.x;
		a.y += b.y;
		return a;
	}
};


/**
 * Initialize problem (and solution)
 */
template <typename T, typename ReductionOp>
void Initialize(
	T *h_in,
	T h_result[1],
	ReductionOp reduction_op,
	int num_elements)
{
	for (int i = 0; i < num_elements; ++i)
	{
		RandomBits(h_in[i]);
//		h_in[i] = 1;
//		h_in[i] = i;
		if (i == 0)
			h_result[0] = h_in[0];
		else
			h_result[0] = reduction_op(h_result[0], h_in[i]);
	}
}


/**
 * Test reduction
 */
template <int CTA_THREADS, int CTA_STRIDES, typename T, typename ReductionOp>
void Test(int num_elements, ReductionOp reduction_op)
{
	const int TILE_SIZE = CTA_THREADS * CTA_STRIDES;

	// Allocate host arrays
	T h_in[TILE_SIZE];
	T h_result[1];

	// Initialize problem
	Initialize(h_in, h_result, reduction_op, num_elements);

	// Initialize device arrays
	T *d_in = NULL;
	T *d_out = NULL;
	DebugExit(cudaMalloc((void**)&d_in, sizeof(T) * TILE_SIZE));
	DebugExit(cudaMalloc((void**)&d_out, sizeof(T) * 1));
	DebugExit(cudaMemcpy(d_in, h_in, sizeof(T) * TILE_SIZE, cudaMemcpyHostToDevice));

	// Run kernel
	if (num_elements == TILE_SIZE)
	{
		// Test unguarded
		printf("Unguarded test CTA_THREADS(%d) CTA_STRIDES(%d) sizeof(T)(%d):\n\t ",
			CTA_THREADS, CTA_STRIDES, (int) sizeof(T));
		fflush(stdout);

		UnguardedKernel<CTA_THREADS, CTA_STRIDES><<<1, CTA_THREADS>>>(
			d_in, d_out, reduction_op, 1);
	}
	else
	{
		// Test guarded
		printf("Guarded test CTA_THREADS(%d) num_elements(%d) sizeof(T)(%d):\n\t ",
			CTA_THREADS, num_elements, (int) sizeof(T));
		fflush(stdout);

		GuardedKernel<CTA_THREADS><<<1, CTA_THREADS>>>(
			d_in, d_out, num_elements, reduction_op);
	}

	DebugExit(cudaDeviceSynchronize());

	// Copy out and display results
	AssertEquals(0, CompareDeviceResults(h_result, d_out, 1, g_verbose, g_verbose));
	printf("\n");

	// Cleanup
	if (d_in) DebugExit(cudaFree(d_in));
	if (d_out) DebugExit(cudaFree(d_out));
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

    Test<32, 	1, int>(32, 	Sum<int>());
    Test<8, 	1, int>(8, 		Sum<int>());
    Test<23, 	1, int>(23, 	Sum<int>());
    Test<512, 	1, int>(512, 	Sum<int>());
    Test<121,	1, int>(121, 	Sum<int>());
    Test<133, 	1, int>(133, 	Sum<int>());
    Test<96, 	1, int>(96, 	Sum<int>());
    Test<32, 	1, int>(12, 	Sum<int>());
    Test<512, 	1, int>(509,	Sum<int>());
    Test<32,	1, uint2>(32, 	Uint2Sum());
    Test<512,	1, uint2>(512, 	Uint2Sum());
    Test<512, 	1, uint2>(509,	Uint2Sum());
    Test<128, 	2, int>(256, 	Sum<int>());
    Test<32, 	2, int>(64, 	Sum<int>());
    Test<16, 	2, int>(32, 	Sum<int>());
    Test<55, 	2, int>(110, 	Sum<int>());
    Test<23, 	2, int>(46, 	Sum<int>());

	return 0;
}



