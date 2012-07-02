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
#include <iostream>
#include <test_util.h>
#include "../cub.cuh"

using namespace cub;

//---------------------------------------------------------------------
// Globals, constants and typedefs
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

//---------------------------------------------------------------------
// Complex data type Foo
//---------------------------------------------------------------------

/**
 * Foo complex data type
 */
struct Foo
{
	long long 	x;
	int 		y;
	short 		z;
	char 		w;

	// Constructor
	__host__ __device__ __forceinline__ Foo() {}

	// Constructor
	__host__ __device__ __forceinline__ Foo(long long x, int y, short z, char w) : x(x), y(y), z(z), w(w) {}

	// Summation operator
	__host__ __device__ __forceinline__ Foo operator+(const Foo &b) const
	{
		return Foo(x + b.x, y + b.y, z + b.z, w + b.w);
	}

	// Inequality operator
	__host__ __device__ __forceinline__ bool operator !=(const Foo &b)
	{
		return (x != b.x) && (y != b.y) && (z != b.z) && (w != b.w);
	}
};

/**
 * Foo ostream operator
 */
std::ostream& operator<<(std::ostream& os, const Foo& val)
{
	os << '(' << val.x << ',' << val.y << ',' << val.z << ',' << CoutCast(val.w) << ')';
	return os;
}

/**
 * Foo test initialization
 */
void InitValue(int gen_mode, Foo &value, int index = 0)
{
	InitValue(gen_mode, value.x, index);
	InitValue(gen_mode, value.y, index);
	InitValue(gen_mode, value.z, index);
	InitValue(gen_mode, value.w, index);
}


//---------------------------------------------------------------------
// Complex data type Bar (with optimizations for fence-free warp-synchrony)
//---------------------------------------------------------------------

/**
 * Bar complex data type
 */
struct Bar
{
	typedef void ThreadLoadTag;
	typedef void ThreadStoreTag;

	long long 	x;
	int 		y;

	// Constructor
	__host__ __device__ __forceinline__ Bar() {}

	// Constructor
	__host__ __device__ __forceinline__ Bar(long long x, int y) : x(x), y(y) {}

	// Summation operator
	__host__ __device__ __forceinline__ Bar operator+(const Bar &b) const
	{
		return Bar(x + b.x, y + b.y);
	}

	// Inequality operator
	__host__ __device__ __forceinline__ bool operator !=(const Bar &b)
	{
		return (x != b.x) && (y != b.y);
	}

	// Volatile shared load
	template <LoadModifier MODIFIER>
	__device__ __forceinline__
	typename EnableIf<(MODIFIER == LOAD_VS), void>::Type ThreadLoad(Bar *ptr)
	{
		volatile long long *x_ptr = &(ptr->x);
		volatile int *y_ptr = &(ptr->y);

		x = *x_ptr;
		y = *y_ptr;
	}

	 // Volatile shared store
	template <StoreModifier MODIFIER>
	__device__ __forceinline__
	typename EnableIf<(MODIFIER == STORE_VS), void>::Type ThreadStore(Bar *ptr) const
	{
		volatile long long *x_ptr = &(ptr->x);
		volatile int *y_ptr = &(ptr->y);

		*x_ptr = x;
		*y_ptr = y;
	}
};

/**
 * Bar ostream operator
 */
std::ostream& operator<<(std::ostream& os, const Bar& val)
{
	os << '(' << val.x << ',' << val.y << ')';
	return os;
}

/**
 * Bar test initialization
 */
void InitValue(int gen_mode, Bar &value, int index = 0)
{
	InitValue(gen_mode, value.x, index);
	InitValue(gen_mode, value.y, index);
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
	T			prefix,
	clock_t		*d_elapsed)
{
	// Cooperative warp-scan utility type (1 warp)
	typedef WarpScan<T, 1, LOGICAL_WARP_THREADS> WarpScan;

	// Shared memory
	__shared__ typename WarpScan::SmemStorage smem_storage;

	// Per-thread tile data
	T data = d_in[threadIdx.x];

	// Record elapsed clocks
	clock_t start = clock();

	// Test scan
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

	// Record elapsed clocks
	*d_elapsed = clock() - start;

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
	T			prefix,
	clock_t		*d_elapsed)
{
	// Cooperative warp-scan utility type (1 warp)
	typedef WarpScan<T, 1, LOGICAL_WARP_THREADS> WarpScan;

	// Shared memory
	__shared__ typename WarpScan::SmemStorage smem_storage;

	// Per-thread tile data
	T data = d_in[threadIdx.x];

	// Record elapsed clocks
	clock_t start = clock();

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

	// Record elapsed clocks
	*d_elapsed = clock() - start;

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
	clock_t *d_elapsed = NULL;
	CubDebugExit(cudaMalloc((void**)&d_in, sizeof(T) * LOGICAL_WARP_THREADS));
	CubDebugExit(cudaMalloc((void**)&d_out, sizeof(T) * (LOGICAL_WARP_THREADS + 1)));
	CubDebugExit(cudaMalloc((void**)&d_elapsed, sizeof(clock_t)));
	CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(T) * LOGICAL_WARP_THREADS, cudaMemcpyHostToDevice));

	// Run kernel
	printf("Test-mode %d, gen-mode %d, %s warpscan, %d warp threads, %d-byte elements:\n",
		TEST_MODE,
		gen_mode,
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
		prefix,
		d_elapsed);

	if (g_verbose)
	{
		printf("\tElapsed clocks: ");
		DisplayDeviceResults(d_elapsed, 1);
	}

	CubDebugExit(cudaDeviceSynchronize());

	// Copy out and display results
	printf("\tScan results: ");
	AssertEquals(0, CompareDeviceResults(h_reference, d_out, LOGICAL_WARP_THREADS, g_verbose, g_verbose));
	printf("\n");

	// Copy out and display aggregate
	if ((TEST_MODE == AGGREGATE) || (TEST_MODE == PREFIX_AGGREGATE))
	{
		printf("\tScan aggregate: ");
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
 * Run battery of tests for different primitive variants
 */
template <
	int 		LOGICAL_WARP_THREADS,
	typename 	ScanOp,
	typename 	T>
void Test(
	int 		gen_mode,
	ScanOp 		scan_op,
	T 			identity,
	T			prefix)
{
	// Exclusive
	Test<LOGICAL_WARP_THREADS, BASIC>(gen_mode, scan_op, identity, prefix);
	Test<LOGICAL_WARP_THREADS, AGGREGATE>(gen_mode, scan_op, identity, prefix);
	Test<LOGICAL_WARP_THREADS, PREFIX_AGGREGATE>(gen_mode, scan_op, identity, prefix);

	// Inclusive
	Test<LOGICAL_WARP_THREADS, BASIC>(gen_mode, scan_op, NullType(), prefix);
	Test<LOGICAL_WARP_THREADS, AGGREGATE>(gen_mode, scan_op, NullType(), prefix);
	Test<LOGICAL_WARP_THREADS, PREFIX_AGGREGATE>(gen_mode, scan_op, NullType(), prefix);
}


/**
 * Run battery of tests for different data types and scan ops
 */
template <int LOGICAL_WARP_THREADS>
void Test(int gen_mode)
{

	// primitive
	Test<LOGICAL_WARP_THREADS>(gen_mode, Sum<unsigned char>(), (unsigned char) 0, (unsigned char) 99);
	Test<LOGICAL_WARP_THREADS>(gen_mode, Sum<unsigned short>(), (unsigned short) 0, (unsigned short) 99);
	Test<LOGICAL_WARP_THREADS>(gen_mode, Sum<unsigned int>(), (unsigned int) 0, (unsigned int) 99);
	Test<LOGICAL_WARP_THREADS>(gen_mode, Sum<unsigned long long>(), (unsigned long long) 0, (unsigned long long) 99);

	// primitive (alternative scan op)
	Test<LOGICAL_WARP_THREADS>(gen_mode, Max<unsigned char>(), (unsigned char) 0, (unsigned char) 99);
	Test<LOGICAL_WARP_THREADS>(gen_mode, Max<unsigned short>(), (unsigned short) 0, (unsigned short) 99);
	Test<LOGICAL_WARP_THREADS>(gen_mode, Max<unsigned int>(), (unsigned int) 0, (unsigned int) 99);
	Test<LOGICAL_WARP_THREADS>(gen_mode, Max<unsigned long long>(), (unsigned long long) 0, (unsigned long long) 99);

	// vec-2
	Test<LOGICAL_WARP_THREADS>(gen_mode, Sum<uchar2>(), make_uchar2(0, 0), make_uchar2(17, 21));
	Test<LOGICAL_WARP_THREADS>(gen_mode, Sum<ushort2>(), make_ushort2(0, 0), make_ushort2(17, 21));
	Test<LOGICAL_WARP_THREADS>(gen_mode, Sum<uint2>(), make_uint2(0, 0), make_uint2(17, 21));
	Test<LOGICAL_WARP_THREADS>(gen_mode, Sum<ulonglong2>(), make_ulonglong2(0, 0), make_ulonglong2(17, 21));

	// vec-4
	Test<LOGICAL_WARP_THREADS>(gen_mode, Sum<uchar4>(), make_uchar4(0, 0, 0, 0), make_uchar4(17, 21, 32, 85));
	Test<LOGICAL_WARP_THREADS>(gen_mode, Sum<ushort4>(), make_ushort4(0, 0, 0, 0), make_ushort4(17, 21, 32, 85));
	Test<LOGICAL_WARP_THREADS>(gen_mode, Sum<uint4>(), make_uint4(0, 0, 0, 0), make_uint4(17, 21, 32, 85));
	Test<LOGICAL_WARP_THREADS>(gen_mode, Sum<ulonglong4>(), make_ulonglong4(0, 0, 0, 0), make_ulonglong4(17, 21, 32, 85));

	// complex
	Test<LOGICAL_WARP_THREADS>(gen_mode, Sum<Foo>(), Foo(0, 0, 0, 0), Foo(17, 21, 32, 85));
	Test<LOGICAL_WARP_THREADS>(gen_mode, Sum<Bar>(), Bar(0, 0), Bar(17, 21));
}


/**
 * Run battery of tests for different problem generation options
 */
template <int LOGICAL_WARP_THREADS>
void Test()
{
	for (int gen_mode = UNIFORM; gen_mode < GEN_MODE_END; gen_mode++)
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
    bool quick = args.CheckCmdLineFlag("quick");

    // Initialize device
    CubDebugExit(args.DeviceInit());

    if (quick)
    {
        // Quick test
        Test<32, BASIC>(UNIFORM, Sum<int>(), int(0), int(10));
    }
    else
    {
        // Test logical warp sizes
        Test<32>();
        Test<16>();
        Test<9>();
        Test<7>();
    }

    return 0;
}



