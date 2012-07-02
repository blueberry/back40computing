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
#include "../cub.cuh"
#include <test_util.h>

using namespace cub;


//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

bool g_verbose = false;


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
 * Test full-tile reduction kernel (where num_elements is an even
 * multiple of CTA_THREADS)
 */
template <
	int 		CTA_THREADS,
	int 		STRIPS,
	int 		ELEMENTS,
	typename 	T,
	typename 	ReductionOp>
__launch_bounds__ (CTA_THREADS, 1)
__global__ void FullTileReduceKernel(
	T 				*d_in,
	T 				*d_out,
	ReductionOp 	reduction_op,
	int				tiles)
{
	const int TILE_SIZE = CTA_THREADS * STRIPS * ELEMENTS;

	// Cooperative CTA tile-loading utility type
	typedef CtaLoad<CTA_THREADS> CtaLoad;

	// Cooperative CTA reduction utility type (returns aggregate in thread 0)
	typedef CtaReduce<T, CTA_THREADS, STRIPS> CtaReduce;

	// Shared memory
	__shared__ typename CtaReduce::SmemStorage smem_storage;

	// Per-thread tile data
	T data[STRIPS][ELEMENTS];

	// Load first tile of data
	int cta_offset = 0;
	CtaLoad::LoadUnguarded(data, d_in, cta_offset);
	cta_offset += TILE_SIZE;

	// Cooperative reduce first tile
	T cta_aggregate = CtaReduce::Reduce(smem_storage, data, reduction_op);

	// Loop over input tiles
	while (cta_offset < TILE_SIZE * tiles)
	{
		// Barrier between CTA reductions
		__syncthreads();

		// Load tile of data
		CtaLoad::LoadUnguarded(data, d_in, cta_offset);
		cta_offset += TILE_SIZE;

		// Cooperatively reduce the tile's aggregate
		T tile_aggregate = CtaReduce::Reduce(smem_storage, data, reduction_op);

		// Reduce CTA aggregate
		cta_aggregate = reduction_op(cta_aggregate, tile_aggregate);
	}

	// Store data
	if (threadIdx.x == 0)
	{
		d_out[0] = cta_aggregate;
	}
}



/**
 * Test partial-tile reduction kernel (where num_elements < CTA_THREADS)
 */
template <
	int 		CTA_THREADS,
	typename 	T,
	typename 	ReductionOp>
__launch_bounds__ (CTA_THREADS, 1)
__global__ void PartialTileReduceKernel(
	T 				*d_in,
	T 				*d_out,
	int 			num_elements,
	ReductionOp 	reduction_op)
{
	// Cooperative CTA reduction utility type (returns aggregate only in thread-0)
	typedef CtaReduce<T, CTA_THREADS> CtaReduce;

	// Shared memory
	__shared__ typename CtaReduce::SmemStorage smem_storage;

	// Per-thread tile data
	T partial;

	// Load partial tile data
	if (threadIdx.x < num_elements)
	{
		partial = d_in[threadIdx.x];
	}

	// Cooperatively reduce the tile's aggregate
	T tile_aggregate = CtaReduce::Reduce(
		smem_storage,
		partial,
		num_elements,
		reduction_op);

	// Store data
	if (threadIdx.x == 0)
	{
		d_out[0] = tile_aggregate;
	}
}


//---------------------------------------------------------------------
// Host utility subroutines
//---------------------------------------------------------------------

/**
 * Initialize problem (and solution)
 */
template <typename T, typename ReductionOp>
void Initialize(
	int		 		gen_mode,
	T 				*h_in,
	T 				h_reference[1],
	ReductionOp 	reduction_op,
	int 			num_elements)
{
	for (int i = 0; i < num_elements; ++i)
	{
		InitValue(gen_mode, h_in[i], i);
		if (i == 0)
			h_reference[0] = h_in[0];
		else
			h_reference[0] = reduction_op(h_reference[0], h_in[i]);
	}
}


//---------------------------------------------------------------------
// Full tile test generation
//---------------------------------------------------------------------


/**
 * Test full-tile reduction
 */
template <
	int 		CTA_THREADS,
	int 		STRIPS,
	int			ELEMENTS,
	typename 	T,
	typename 	ReductionOp>
void TestFullTile(
	int 			gen_mode,
	int 			tiles,
	ReductionOp 	reduction_op)
{
	const int TILE_SIZE = CTA_THREADS * STRIPS * ELEMENTS;

	int num_elements = TILE_SIZE * tiles;

	// Allocate host arrays
	T *h_in = new T[num_elements];
	T h_reference[1];

	// Initialize problem
	Initialize(gen_mode, h_in, h_reference, reduction_op, num_elements);

	// Initialize device arrays
	T *d_in = NULL;
	T *d_out = NULL;
	CubDebugExit(cudaMalloc((void**)&d_in, sizeof(T) * num_elements));
	CubDebugExit(cudaMalloc((void**)&d_out, sizeof(T) * 1));
	CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(T) * num_elements, cudaMemcpyHostToDevice));

	// Test multi-tile (unguarded)
	printf("TestFullTile gen-mode %d, num_elements(%d), CTA_THREADS(%d) STRIPS(%d) ELEMENTS(%d) sizeof(T)(%d):\n",
		gen_mode,
		num_elements,
		CTA_THREADS,
		STRIPS,
		ELEMENTS,
		(int) sizeof(T));
	fflush(stdout);

	FullTileReduceKernel<CTA_THREADS, STRIPS, ELEMENTS><<<1, CTA_THREADS>>>(
		d_in,
		d_out,
		reduction_op,
		tiles);

	CubDebugExit(cudaDeviceSynchronize());

	// Copy out and display results
	printf("\tReduction results: ");
	AssertEquals(0, CompareDeviceResults(h_reference, d_out, 1, g_verbose, g_verbose));
	printf("\n");

	// Cleanup
	if (h_in) free(h_in);
	if (d_in) CubDebugExit(cudaFree(d_in));
	if (d_out) CubDebugExit(cudaFree(d_out));
}

/**
 * Run battery of tests for different thread strip elements
 */
template <
	int 		CTA_THREADS,
	int 		STRIPS,
	typename 	T,
	typename 	ReductionOp>
void TestFullTile(
	int 			gen_mode,
	int 			tiles,
	ReductionOp 	reduction_op)
{
	TestFullTile<CTA_THREADS, STRIPS, 1, T>(gen_mode, tiles, reduction_op);
	TestFullTile<CTA_THREADS, STRIPS, 2, T>(gen_mode, tiles, reduction_op);
	TestFullTile<CTA_THREADS, STRIPS, 4, T>(gen_mode, tiles, reduction_op);
}


/**
 * Run battery of tests for different strips
 */
template <
	int 		CTA_THREADS,
	typename 	T,
	typename 	ReductionOp>
void TestFullTile(
	int 			gen_mode,
	int 			tiles,
	ReductionOp 	reduction_op)
{
	TestFullTile<CTA_THREADS, 1, T>(gen_mode, tiles, reduction_op);
	TestFullTile<CTA_THREADS, 2, T>(gen_mode, tiles, reduction_op);
	TestFullTile<CTA_THREADS, 4, T>(gen_mode, tiles, reduction_op);
}


/**
 * Run battery of full-tile tests for different cta sizes
 */
template <
	typename 	T,
	typename 	ReductionOp>
void TestFullTile(
	int 			gen_mode,
	int 			tiles,
	ReductionOp 	reduction_op)
{
	TestFullTile<7, T>(gen_mode, tiles, reduction_op);
	TestFullTile<31, T>(gen_mode, tiles, reduction_op);
	TestFullTile<32, T>(gen_mode, tiles, reduction_op);
	TestFullTile<65, T>(gen_mode, tiles, reduction_op);
	TestFullTile<96, T>(gen_mode, tiles, reduction_op);
	TestFullTile<128, T>(gen_mode, tiles, reduction_op);
}


/**
 * Run battery of full-tile tests for different numbers of tiles
 */
template <
	typename 	T,
	typename 	ReductionOp>
void TestFullTile(
	int 			gen_mode,
	ReductionOp 	reduction_op)
{
	for (int tiles = 1; tiles < 3; tiles++)
	{
		TestFullTile<T>(gen_mode, tiles, reduction_op);
	}
}


//---------------------------------------------------------------------
// Partial-tile test generation
//---------------------------------------------------------------------

/**
 * Test partial-tile reduction
 */
template <
	int 		CTA_THREADS,
	typename 	T,
	typename 	ReductionOp>
void TestPartialTile(
	int 			gen_mode,
	int 			num_elements,
	ReductionOp 	reduction_op)
{
	const int TILE_SIZE = CTA_THREADS;

	// Allocate host arrays
	T *h_in = new T[num_elements];
	T h_reference[1];

	// Initialize problem
	Initialize(gen_mode, h_in, h_reference, reduction_op, num_elements);

	// Initialize device arrays
	T *d_in = NULL;
	T *d_out = NULL;
	CubDebugExit(cudaMalloc((void**)&d_in, sizeof(T) * TILE_SIZE));
	CubDebugExit(cudaMalloc((void**)&d_out, sizeof(T) * 1));
	CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(T) * TILE_SIZE, cudaMemcpyHostToDevice));

	printf("TestPartialTile gen-mode %d, num_elements(%d), CTA_THREADS(%d) sizeof(T)(%d):\n",
		gen_mode,
		num_elements,
		CTA_THREADS,
		(int) sizeof(T));
	fflush(stdout);

	PartialTileReduceKernel<CTA_THREADS><<<1, CTA_THREADS>>>(
		d_in,
		d_out,
		num_elements,
		reduction_op);

	CubDebugExit(cudaDeviceSynchronize());

	// Copy out and display results
	printf("\tReduction results: ");
	AssertEquals(0, CompareDeviceResults(h_reference, d_out, 1, g_verbose, g_verbose));
	printf("\n");

	// Cleanup
	if (h_in) free(h_in);
	if (d_in) CubDebugExit(cudaFree(d_in));
	if (d_out) CubDebugExit(cudaFree(d_out));
}


/**
 *  Run battery of full-tile tests for different numbers of effective threads
 */
template <
	int 		CTA_THREADS,
	typename 	T,
	typename 	ReductionOp>
void TestPartialTile(
	int 			gen_mode,
	ReductionOp 	reduction_op)
{
	for (
		int num_elements = 1;
		num_elements < CTA_THREADS;
		num_elements += CUB_MAX(1, CTA_THREADS / 5))
	{
		TestPartialTile<CTA_THREADS, T>(gen_mode, num_elements, reduction_op);
	}
}


/**
 * Run battery of full-tile tests for different cta sizes
 */
template <
	typename 	T,
	typename 	ReductionOp>
void TestPartialTile(
	int 			gen_mode,
	ReductionOp 	reduction_op)
{
	TestPartialTile<7, T>(gen_mode, reduction_op);
	TestPartialTile<31, T>(gen_mode, reduction_op);
	TestPartialTile<32, T>(gen_mode, reduction_op);
	TestPartialTile<65, T>(gen_mode, reduction_op);
	TestPartialTile<96, T>(gen_mode, reduction_op);
	TestPartialTile<128, T>(gen_mode, reduction_op);
}


//---------------------------------------------------------------------
// Main
//---------------------------------------------------------------------

/**
 * Run battery of full-tile tests for different gen modes
 */
template <typename T, typename ReductionOp>
void Test(ReductionOp reduction_op)
{
	for (int gen_mode = UNIFORM; gen_mode < GEN_MODE_END; gen_mode++)
	{
		TestFullTile<T>(gen_mode, reduction_op);
		TestPartialTile<T>(gen_mode, reduction_op);
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
        Test<int>(Sum<int>());
    }
    else
    {
		// primitives
		Test<char>(Sum<char>());
		Test<short>(Sum<short>());
		Test<int>(Sum<int>());
		Test<long long>(Sum<long long>());

		// vector types
		Test<char2>(Sum<char2>());
		Test<short2>(Sum<short2>());
		Test<int2>(Sum<int2>());
		Test<longlong2>(Sum<longlong2>());

		Test<char4>(Sum<char4>());
		Test<short4>(Sum<short4>());
		Test<int4>(Sum<int4>());
		Test<longlong4>(Sum<longlong4>());

		// Complex types
		Test<Foo>(Sum<Foo>());
		Test<Bar>(Sum<Bar>());
    }

    return 0;
}



