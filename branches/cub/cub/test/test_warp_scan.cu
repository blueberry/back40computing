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
#include <cub/cub.cuh>

using namespace cub;

bool g_verbose = false;


//---------------------------------------------------------------------
// Warp scan test kernels
//---------------------------------------------------------------------

/**
 * WarpScan test kernel (exclusive variant).
 */
template <
	typename 	T,
	typename 	ScanOp,
	typename 	IdentityT>
__global__ void WarpScanKernel(
	T 				*d_in,
	T 				*d_out,
	ScanOp 			scan_op,
	IdentityT 		identity)
{
	// Cooperative warp-scan utility type (returns aggregate in all threads)
	typedef WarpScan<1, T> WarpScan;

	// Shared memory
	__shared__ typename WarpScan::SmemStorage smem_storage;

	// Per-thread tile data
	T data = d_in[threadIdx.x];

	// Exclusive warp scan
	data = WarpScan::ExclusiveScan(smem_storage, data, scan_op, identity);

	// Store data
	d_out[threadIdx.x] = data;
}


/**
 * WarpScan test kernel (inclusive variant).
 */
template <
	typename 	T,
	typename 	ScanOp>
__global__ void WarpScanKernel(
	T 				*d_in,
	T 				*d_out,
	ScanOp 			scan_op,
	NullType)
{
	// Cooperative warp-scan utility type (returns aggregate in all threads)
	typedef WarpScan<1, T> WarpScan;

	// Shared memory
	__shared__ typename WarpScan::SmemStorage smem_storage;

	// Per-thread tile data
	T data = d_in[threadIdx.x];

	// Inclusive warp scan
	data = WarpScan::InclusiveScan(smem_storage, data, scan_op);

	// Store data
	d_out[threadIdx.x] = data;
}


//---------------------------------------------------------------------
// Test routines
//---------------------------------------------------------------------

/**
 * Uint2 reduction operator
 */
struct Uint2Sum
{
	// Scan op
	__host__ __device__ __forceinline__ uint2 operator()(uint2 a, uint2 b)
	{
		a.x += b.x;
		a.y += b.y;
		return a;
	}

	// Identity
	__host__ __device__ __forceinline__ uint2 operator()()
	{
		uint2 retval;
		retval.x = retval.y = 0;
		return retval;
	}
};


/**
 * Initialize value at a given index
 */
template <typename T>
void InitValue(T &value, int index)
{
	RandomBits(value);
//	value = 1;
//	value = index;
}

/**
 * Initialize value at a given index
 */
void InitValue(uint2 &value, int index)
{
	RandomBits(value.x);
//	value.x = 1;
//	value.x = index;

	value.y = value.x;
}


/**
 * Initialize exclusive-scan problem (and solution)
 */
template <
	typename 	T,
	typename 	ScanOp,
	typename 	IdentityT>
void Initialize(
	T 			*h_in,
	T 			*h_reference,
	int 		num_elements,
	ScanOp 		scan_op,
	IdentityT 	identity)
{
	T prefix = identity;

	for (int i = 0; i < num_elements; ++i)
	{
		InitValue(h_in[i], i);
		h_reference[i] = prefix;
		prefix = scan_op(prefix, h_in[i]);
	}
}


/**
 * Initialize inclusive-scan problem (and solution)
 */
template <
	typename 	T,
	typename 	ScanOp>
void Initialize(
	T 			*h_in,
	T 			*h_reference,
	int 		num_elements,
	ScanOp 		scan_op,
	NullType)
{
	T prefix;
	for (int i = 0; i < num_elements; ++i)
	{
		InitValue(h_in[i], i);
		if (i == 0) {
			prefix = h_in[0];
		} else {
			prefix = scan_op(prefix, h_in[i]);
		}
		h_reference[i] = prefix;
	}
}


/**
 * Test warp scan
 */
template <
	typename 	T,
	typename 	ScanOp,
	typename 	IdentityT>		// NullType for inclusive-scan
void Test(
	int 		warp_size,
	ScanOp 		scan_op,
	IdentityT 	identity)
{
	// Allocate host arrays
	T *h_in = new T[warp_size];
	T *h_reference = new T[warp_size];

	// Initialize problem
	Initialize(h_in, h_reference, warp_size, scan_op, identity);

	// Initialize device arrays
	T *d_in = NULL;
	T *d_out = NULL;
	DebugExit(cudaMalloc((void**)&d_in, sizeof(T) * warp_size));
	DebugExit(cudaMalloc((void**)&d_out, sizeof(T) * warp_size));
	DebugExit(cudaMemcpy(d_in, h_in, sizeof(T) * warp_size, cudaMemcpyHostToDevice));

	// Run kernel
	printf("%s warpscan warp_size(%d) sizeof(T)(%d):\n\t ",
		(Equals<IdentityT, NullType>::VALUE) ? "Inclusive" : "Exclusive",
		warp_size,
		(int) sizeof(T));
	fflush(stdout);

	WarpScanKernel<<<1, warp_size>>>(
		d_in,
		d_out,
		scan_op,
		identity);

	DebugExit(cudaDeviceSynchronize());

	// Copy out and display results
	AssertEquals(0, CompareDeviceResults(h_reference, d_out, warp_size, g_verbose, g_verbose));
	printf("\n");

	// Cleanup
	if (h_in) delete h_in;
	if (h_reference) delete h_in;
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

    // int exclusive sum
    {
    	typedef int T;
    	Sum<T> scan_op;
    	T identity = scan_op();
    	Test<T>(32, scan_op, identity);
    }

    // int inclusive sum
    {
    	typedef int T;
    	Sum<T> scan_op;
    	NullType identity;
    	Test<T>(32, scan_op, identity);
    }

    // int inclusive max
    {
    	typedef int T;
    	Max<T> scan_op;
    	NullType identity;
    	Test<T>(32, scan_op, identity);
    }

    // uint2 exclusive sum
    {
    	typedef uint2 T;
    	Uint2Sum scan_op;
    	T identity = scan_op();
    	Test<T>(32, scan_op, identity);
    }

    // uint2 inclusive sum
    {
    	typedef uint2 T;
    	Uint2Sum scan_op;
    	NullType identity;
    	Test<T>(32, scan_op, identity);
    }

    return 0;
}



