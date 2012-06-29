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
 * Test of WarpScan utilities
 ******************************************************************************/

#define CUB_STDERR

#include <stdio.h>
#include <test_util.h>
#include "../cub.cuh"

using namespace cub;

//---------------------------------------------------------------------
// Globals , constants and typedefs
//---------------------------------------------------------------------

/**
 * Verbose output
 */
bool g_verbose = false;


/**
 * Primitive variant to test
 */
enum TestMode
{
	BASIC,
	AGGREGATE,
	PREFIX_AGGREGATE,
};


/**
 * Test problem generation options
 */
enum GenMode
{
	UNIFORM,			// All 1s
	SEQ_INC,			// Sequentially incrementing
	RANDOM,				// Random

	GEN_MODE_END,
};


/**
 * Uint2 summation operator
 */
__host__ __device__ __forceinline__ uint2 operator+(uint2 a, uint2 b)
{
	a.x += b.x;
	a.y += b.y;
	return a;
}



//---------------------------------------------------------------------
// Test kernels
//---------------------------------------------------------------------

/**
 * Exclusive WarpScan test kernel.
 */
template <
	int 		LOGICAL_WARP_THREADS,
	TestMode	TEST_MODE,
	typename 	T,
	typename 	ScanOp,
	typename 	IdentityT>
__global__ void WarpScanKernel(
	T 			*d_in,
	T 			*d_out,
	ScanOp 		scan_op,
	IdentityT 	identity,
	T			prefix)
{
	// Cooperative warp-scan utility type (1 warp)
	typedef WarpScan<T, 1, LOGICAL_WARP_THREADS> WarpScan;

	// Shared memory
	__shared__ typename WarpScan::SmemStorage smem_storage;

	// Per-thread tile data
	T data = d_in[threadIdx.x];

	T aggregate;
	if (TEST_MODE == BASIC)
	{
		// Test basic warp scan
		WarpScan::ExclusiveScan(smem_storage, data, data, scan_op, identity);
	}
	else if (TEST_MODE == AGGREGATE)
	{
		// Test with warp-prefix and cumulative aggregate
		WarpScan::ExclusiveScan(smem_storage, data, data, scan_op, identity, aggregate);
	}
	else if (TEST_MODE == PREFIX_AGGREGATE)
	{
		// Test with warp-prefix and cumulative aggregate
		WarpScan::ExclusiveScan(smem_storage, data, data, scan_op, identity, aggregate, prefix);
	}

	// Store data
	d_out[threadIdx.x] = data;

	// Store aggregate
	if (threadIdx.x == 0)
	{
		d_out[blockDim.x] = aggregate;
	}
}


/**
 * Inclusive WarpScan test kernel.
 */
template <
	int 		LOGICAL_WARP_THREADS,
	TestMode	TEST_MODE,
	typename 	T,
	typename 	ScanOp>
__global__ void WarpScanKernel(
	T 			*d_in,
	T 			*d_out,
	ScanOp 		scan_op,
	NullType,
	T			prefix)
{
	// Cooperative warp-scan utility type (1 warp)
	typedef WarpScan<T, 1, LOGICAL_WARP_THREADS> WarpScan;

	// Shared memory
	__shared__ typename WarpScan::SmemStorage smem_storage;

	// Per-thread tile data
	T data = d_in[threadIdx.x];

	T aggregate;
	if (TEST_MODE == BASIC)
	{
		// Test basic warp scan
		WarpScan::InclusiveScan(smem_storage, data, data, scan_op);
	}
	else if (TEST_MODE == AGGREGATE)
	{
		// Test with warp-prefix and cumulative aggregate
		WarpScan::InclusiveScan(smem_storage, data, data, scan_op, aggregate);
	}
	else if (TEST_MODE == PREFIX_AGGREGATE)
	{
		// Test with warp-prefix and cumulative aggregate
		WarpScan::InclusiveScan(smem_storage, data, data, scan_op, aggregate, prefix);
	}

	// Store data
	d_out[threadIdx.x] = data;

	// Store aggregate
	if (threadIdx.x == 0)
	{
		d_out[blockDim.x] = aggregate;
	}
}


//---------------------------------------------------------------------
// Host utility subroutines
//---------------------------------------------------------------------

/**
 * Initialize value at a given index
 */
template <typename T>
void InitValue(int gen_mode, T &value, int index)
{
	switch (gen_mode)
	{
	case UNIFORM:
		value = 1;
		break;
	case SEQ_INC:
		value = index;
		break;
	case RANDOM:
	default:
		RandomBits(value);
	}
}

/**
 * Initialize value at a given index.  Specialized for uint2.
 */
void InitValue(int gen_mode, uint2 &value, int index)
{
	InitValue(gen_mode, value.x, index);
	value.y = value.x;
}


/**
 * Initialize exclusive-scan problem (and solution)
 */
template <
	typename 	T,
	typename 	ScanOp,
	typename 	IdentityT>
T Initialize(
	int		 	gen_mode,
	T 			*h_in,
	T 			*h_reference,
	int 		num_elements,
	ScanOp 		scan_op,
	IdentityT 	identity,
	T			*prefix)
{
	T inclusive = (prefix != NULL) ? *prefix : identity;

	for (int i = 0; i < num_elements; ++i)
	{
		InitValue(gen_mode, h_in[i], i);
		h_reference[i] = inclusive;
		inclusive = scan_op(inclusive, h_in[i]);
	}

	return inclusive;
}


/**
 * Initialize inclusive-scan problem (and solution)
 */
template <
	typename 	T,
	typename 	ScanOp>
T Initialize(
	int		 	gen_mode,
	T 			*h_in,
	T 			*h_reference,
	int 		num_elements,
	ScanOp 		scan_op,
	NullType,
	T			*prefix)
{
	T inclusive;
	for (int i = 0; i < num_elements; ++i)
	{
		InitValue(gen_mode, h_in[i], i);
		if (i == 0)
		{
			inclusive = (prefix != NULL) ?
				scan_op(*prefix, h_in[0]) :
				h_in[0];
		}
		else
		{
			inclusive = scan_op(inclusive, h_in[i]);
		}
		h_reference[i] = inclusive;
	}

	return inclusive;
}


/**
 * Test warp scan
 */
template <
	int 		LOGICAL_WARP_THREADS,
	TestMode 	TEST_MODE,
	typename 	ScanOp,
	typename 	IdentityT,		// NullType implies inclusive-scan, otherwise inclusive scan
	typename 	T>
void Test(
	int 		gen_mode,
	ScanOp 		scan_op,
	IdentityT 	identity,
	T			prefix)
{
	// Allocate host arrays
	T *h_in = new T[LOGICAL_WARP_THREADS];
	T *h_reference = new T[LOGICAL_WARP_THREADS];

	// Initialize problem
	T *p_prefix = (TEST_MODE == PREFIX_AGGREGATE) ? &prefix : NULL;
	T aggregate = Initialize(gen_mode, h_in, h_reference, LOGICAL_WARP_THREADS, scan_op, identity, p_prefix);

	// Initialize device arrays
	T *d_in = NULL;
	T *d_out = NULL;
	CubDebugExit(cudaMalloc((void**)&d_in, sizeof(T) * LOGICAL_WARP_THREADS));
	CubDebugExit(cudaMalloc((void**)&d_out, sizeof(T) * (LOGICAL_WARP_THREADS + 1)));
	CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(T) * LOGICAL_WARP_THREADS, cudaMemcpyHostToDevice));

	// Run kernel
	printf("%s warpscan LOGICAL_WARP_THREADS(%d) sizeof(T)(%d):\n",
		(Equals<IdentityT, NullType>::VALUE) ? "Inclusive" : "Exclusive",
		LOGICAL_WARP_THREADS,
		(int) sizeof(T));
	fflush(stdout);

	// Run aggregate/prefix kernel
	WarpScanKernel<LOGICAL_WARP_THREADS, TEST_MODE><<<1, LOGICAL_WARP_THREADS>>>(
		d_in,
		d_out,
		scan_op,
		identity,
		prefix);

	CubDebugExit(cudaDeviceSynchronize());

	// Copy out and display results
	AssertEquals(0, CompareDeviceResults(h_reference, d_out, LOGICAL_WARP_THREADS, g_verbose, g_verbose));
	printf("\n");

	// Copy out and display aggregate
	if ((TEST_MODE == AGGREGATE) || (TEST_MODE == PREFIX_AGGREGATE))
	{
		AssertEquals(0, CompareDeviceResults(&aggregate, d_out + LOGICAL_WARP_THREADS, 1, g_verbose, g_verbose));
		printf("\n");
	}

	// Cleanup
	if (h_in) delete h_in;
	if (h_reference) delete h_in;
	if (d_in) CubDebugExit(cudaFree(d_in));
	if (d_out) CubDebugExit(cudaFree(d_out));
}

/**
 * Run battery of tests for different logical warp widths (which
 * must be less than or equal to the device warp width)
 */
template <int LOGICAL_WARP_THREADS>
void Test(int gen_mode)
{
/*
    // int sum
    {
    	typedef int T;
    	Sum<T> scan_op;
    	T identity = 0;
    	T prefix = 99;

    	// Exclusive
    	Test<LOGICAL_WARP_THREADS, BASIC>(gen_mode, scan_op, identity, prefix);
    	Test<LOGICAL_WARP_THREADS, AGGREGATE>(gen_mode, scan_op, identity, prefix);
    	Test<LOGICAL_WARP_THREADS, PREFIX_AGGREGATE>(gen_mode, scan_op, identity, prefix);

    	// Inclusive
    	Test<LOGICAL_WARP_THREADS, BASIC>(gen_mode, scan_op, NullType(), prefix);
    	Test<LOGICAL_WARP_THREADS, AGGREGATE>(gen_mode, scan_op, NullType(), prefix);
    	Test<LOGICAL_WARP_THREADS, PREFIX_AGGREGATE>(gen_mode, scan_op, NullType(), prefix);
    }

    // uint max
    {
    	typedef unsigned int T;
    	Max<T> scan_op;
    	T identity = 0;
    	T prefix = 99;

    	// Exclusive
    	Test<LOGICAL_WARP_THREADS, BASIC>(gen_mode, scan_op, identity, prefix);
    	Test<LOGICAL_WARP_THREADS, AGGREGATE>(gen_mode, scan_op, identity, prefix);
    	Test<LOGICAL_WARP_THREADS, PREFIX_AGGREGATE>(gen_mode, scan_op, identity, prefix);

    	// Inclusive
    	Test<LOGICAL_WARP_THREADS, BASIC>(gen_mode, scan_op, NullType(), prefix);
    	Test<LOGICAL_WARP_THREADS, AGGREGATE>(gen_mode, scan_op, NullType(), prefix);
    	Test<LOGICAL_WARP_THREADS, PREFIX_AGGREGATE>(gen_mode, scan_op, NullType(), prefix);

    }
*/
    // uint2 sum
//    {
    	typedef uint2 T;
    	Sum<T> scan_op;
    	T identity = {0, 0};
    	T prefix = {14, 21};

    	// Exclusive
    	Test<LOGICAL_WARP_THREADS, BASIC>(gen_mode, scan_op, identity, prefix);
/*
    	Test<LOGICAL_WARP_THREADS, AGGREGATE>(gen_mode, scan_op, identity, prefix);
    	Test<LOGICAL_WARP_THREADS, PREFIX_AGGREGATE>(gen_mode, scan_op, identity, prefix);

    	// Inclusive
    	Test<LOGICAL_WARP_THREADS, BASIC>(gen_mode, scan_op, NullType(), prefix);
    	Test<LOGICAL_WARP_THREADS, AGGREGATE>(gen_mode, scan_op, NullType(), prefix);
    	Test<LOGICAL_WARP_THREADS, PREFIX_AGGREGATE>(gen_mode, scan_op, NullType(), prefix);
    }
*/
}


/**
 * Run battery of tests for different problem generation options
 */
template <int LOGICAL_WARP_THREADS>
void Test()
{
	for (
		int gen_mode = UNIFORM;
		gen_mode < UNIFORM + 1;
//		gen_mode < GEN_MODE_END;
		gen_mode++)
	{
		Test<LOGICAL_WARP_THREADS>(gen_mode);
	}
}


/**
 * Main
 */
int main(int argc, char** argv)
{
    // Initialize command line
    CommandLineArgs args(argc, argv);
    g_verbose = args.CheckCmdLineFlag("v");

    // Initialize device
    CubDebugExit(args.DeviceInit());

    // Test logical warp sizes
    Test<32>();
//    Test<16>();
//    Test<9>();
//    Test<7>();

    return 0;
}



