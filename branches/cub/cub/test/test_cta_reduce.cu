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

// Ensure printing of CUDA runtime errors to console
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

	// Factory
	static __host__ __device__ __forceinline__ Foo MakeFoo(long long x, int y, short z, char w)
	{
		Foo retval = {x, y, z, w};
		return retval;
	}

	// Summation operator
	__host__ __device__ __forceinline__ Foo operator+(const Foo &b) const
	{
		return MakeFoo(x + b.x, y + b.y, z + b.z, w + b.w);
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

	// Factory
	static __host__ __device__ __forceinline__ Bar MakeBar(long long x, int y)
	{
		Bar retval = {x, y};
		return retval;
	}

	// Summation operator
	__host__ __device__ __forceinline__ Bar operator+(const Bar &b) const
	{
		return MakeBar(x + b.x, y + b.y);
	}

	// Inequality operator
	__host__ __device__ __forceinline__ bool operator !=(const Bar &b)
	{
		return (x != b.x) && (y != b.y);
	}

	// ThreadLoad
	template <LoadModifier MODIFIER>
	__device__ __forceinline__
	void ThreadLoad(Bar *ptr)
	{
		x = cub::ThreadLoad<MODIFIER>(&(ptr->x));
		y = cub::ThreadLoad<MODIFIER>(&(ptr->y));
	}

	 // ThreadStore
	template <StoreModifier MODIFIER>
	__device__ __forceinline__ void ThreadStore(Bar *ptr) const
	{
		cub::ThreadStore<MODIFIER>(&(ptr->x), x);
		cub::ThreadStore<MODIFIER>(&(ptr->y), y);
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

	// Cooperative CTA reduction utility type (returns aggregate in thread 0)
	typedef CtaReduce<T, CTA_THREADS, STRIPS> CtaReduce;

	// Shared memory
	__shared__ typename CtaReduce::SmemStorage smem_storage;

	// Per-thread tile data
	T data[STRIPS][ELEMENTS];

	// Load first tile of data
	int cta_offset = 0;
	CtaLoad<CTA_THREADS>::LoadUnguarded(data, d_in, cta_offset);
	cta_offset += TILE_SIZE;

	// Cooperative reduce first tile
	T cta_aggregate = CtaReduce::Reduce(smem_storage, data, reduction_op);

	// Loop over input tiles
	while (cta_offset < TILE_SIZE * tiles)
	{
		// Barrier between CTA reductions
		__syncthreads();

		// Load tile of data
		CtaLoad<CTA_THREADS>::LoadUnguarded(data, d_in, cta_offset);
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
	ReductionOp 	reduction_op,
	char			*type_string)
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
	printf("TestFullTile, gen-mode %d, num_elements(%d), CTA_THREADS(%d), STRIPS(%d), ELEMENTS(%d), %s (%d bytes) elements:\n",
		gen_mode,
		num_elements,
		CTA_THREADS,
		STRIPS,
		ELEMENTS,
		type_string,
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
	ReductionOp 	reduction_op,
	char			*type_string)
{
	TestFullTile<CTA_THREADS, STRIPS, 1, T>(gen_mode, tiles, reduction_op, type_string);
//	TestFullTile<CTA_THREADS, STRIPS, 4, T>(gen_mode, tiles, reduction_op, type_string);
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
	ReductionOp 	reduction_op,
	char			*type_string)
{
	TestFullTile<CTA_THREADS, 1, T>(gen_mode, tiles, reduction_op, type_string);
	TestFullTile<CTA_THREADS, 4, T>(gen_mode, tiles, reduction_op, type_string);
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
	ReductionOp 	reduction_op,
	char			*type_string)
{
	TestFullTile<7, T>(gen_mode, tiles, reduction_op, type_string);
	TestFullTile<32, T>(gen_mode, tiles, reduction_op, type_string);
	TestFullTile<63, T>(gen_mode, tiles, reduction_op, type_string);
	TestFullTile<65, T>(gen_mode, tiles, reduction_op, type_string);
	TestFullTile<128, T>(gen_mode, tiles, reduction_op, type_string);
}


/**
 * Run battery of full-tile tests for different numbers of tiles
 */
template <
	typename 	T,
	typename 	ReductionOp>
void TestFullTile(
	int 			gen_mode,
	ReductionOp 	reduction_op,
	char			*type_string)
{
	for (int tiles = 1; tiles < 3; tiles++)
	{
		TestFullTile<T>(gen_mode, tiles, reduction_op, type_string);
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
	ReductionOp 	reduction_op,
	char			*type_string)
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

	printf("TestPartialTile, gen-mode %d, num_elements(%d), CTA_THREADS(%d), %s (%d bytes) elements:\n",
		gen_mode,
		num_elements,
		CTA_THREADS,
		type_string,
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
 *  Run battery of partial-tile tests for different numbers of effective threads
 */
template <
	int 		CTA_THREADS,
	typename 	T,
	typename 	ReductionOp>
void TestPartialTile(
	int 			gen_mode,
	ReductionOp 	reduction_op,
	char			*type_string)
{
	for (
		int num_elements = 1;
		num_elements < CTA_THREADS;
		num_elements += CUB_MAX(1, CTA_THREADS / 5))
	{
		TestPartialTile<CTA_THREADS, T>(gen_mode, num_elements, reduction_op, type_string);
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
	ReductionOp 	reduction_op,
	char			*type_string)
{
	TestPartialTile<7, T>(gen_mode, reduction_op, type_string);
	TestPartialTile<32, T>(gen_mode, reduction_op, type_string);
	TestPartialTile<63, T>(gen_mode, reduction_op, type_string);
	TestPartialTile<65, T>(gen_mode, reduction_op, type_string);
	TestPartialTile<128, T>(gen_mode, reduction_op, type_string);
}


//---------------------------------------------------------------------
// Main
//---------------------------------------------------------------------

/**
 * Run battery of full-tile tests for different gen modes
 */
template <typename T, typename ReductionOp>
void Test(ReductionOp reduction_op, char *type_string)
{
	for (int gen_mode = UNIFORM; gen_mode < GEN_MODE_END; gen_mode++)
	{
		TestFullTile<T>(gen_mode, reduction_op, type_string);
		TestPartialTile<T>(gen_mode, reduction_op, type_string);
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

    // Print usage
    if (args.CheckCmdLineFlag("help"))
    {
    	printf("%s "
    		"[--device=<device-id>] "
    		"[--v] "
    		"[--quick]"
    		"\n", argv[0]);
    	exit(0);
    }

    // Initialize device
    CubDebugExit(args.DeviceInit());

    if (quick)
    {
        // Quick test
    	typedef int T;
    	TestFullTile<128, 1, 4, T>(UNIFORM, 1, Sum<T>(), CUB_TYPE_STRING(T));
    }
    else
    {
		// primitives
		Test<char>(Sum<char>(), CUB_TYPE_STRING(char));
		Test<short>(Sum<short>(), CUB_TYPE_STRING(short));
		Test<int>(Sum<int>(), CUB_TYPE_STRING(int));
		Test<long long>(Sum<long long>(), CUB_TYPE_STRING(long long));

		// vector types
		Test<char2>(Sum<char2>(), CUB_TYPE_STRING(char2));
		Test<short2>(Sum<short2>(), CUB_TYPE_STRING(short2));
		Test<int2>(Sum<int2>(), CUB_TYPE_STRING(int2));
		Test<longlong2>(Sum<longlong2>(), CUB_TYPE_STRING(longlong2));

		Test<char4>(Sum<char4>(), CUB_TYPE_STRING(char4));
		Test<short4>(Sum<short4>(), CUB_TYPE_STRING(short4));
		Test<int4>(Sum<int4>(), CUB_TYPE_STRING(int4));
		Test<longlong4>(Sum<longlong4>(), CUB_TYPE_STRING(longlong4));

		// Complex types
		Test<Foo>(Sum<Foo>(), CUB_TYPE_STRING(Foo));
		Test<Bar>(Sum<Bar>(), CUB_TYPE_STRING(Bar));
    }

    return 0;
}



