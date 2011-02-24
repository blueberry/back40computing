/******************************************************************************
 * 
 * Copyright 2010-2011 Duane Merrill
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
 * For more information, see our Google Code project site: 
 * http://code.google.com/p/back40computing/
 * 
 ******************************************************************************/


/******************************************************************************
 * Tuning tool for establishing optimal reduction granularity configuration types
 ******************************************************************************/

#include <stdio.h> 

// Reduction includes
#include "reduction_api_granularity.cuh"
#include "reduction_api_enactor.cuh"

// Test utils
#include "b40c_util.h"

using namespace b40c;
using namespace reduction;

/******************************************************************************
 * Defines, constants, globals, and utility types
 ******************************************************************************/

bool g_verbose;
int g_max_ctas = 0;
int g_iterations = 0;


template <typename _T, typename _OpType>
struct Detail
{
	typedef _T T;
	typedef _OpType OpType;

	ReductionEnactor<> enactor;
	T *d_dest;
	T *d_src;
	T *h_data;
	T *h_reference;
	size_t num_elements;

	Detail(size_t num_elements) :
		d_dest(NULL), d_src(NULL), h_data(NULL), h_reference(NULL), num_elements(num_elements) {}
};


/******************************************************************************
 * Utility routines
 ******************************************************************************/

/**
 * Displays the commandline usage for this tool
 */
void Usage()
{
	printf("\ntune_reduction [--device=<device index>] [--v] [--i=<num-iterations>] "
			"[--max-ctas=<max-thread-blocks>] [--n=<num-elements>]\n");
	printf("\n");
	printf("\t--v\tDisplays verbose configuration to the console.\n");
	printf("\n");
	printf("\t--i\tPerforms the reduction operation <num-iterations> times\n");
	printf("\t\t\ton the device. Re-copies original input each time. Default = 1\n");
	printf("\n");
	printf("\t--n\tThe number of elements to comprise the sample problem\n");
	printf("\t\t\tDefault = 512\n");
	printf("\n");
}



/**
 * Timed reduction for a specific granularity configuration type
 */
template <typename Detail, typename Config>
void TimedReduction(Detail &detail)
{
	Config::Print();
	fflush(stdout);

	// Perform a single iteration to allocate any memory if needed, prime code caches, etc.
	detail.enactor.DEBUG = g_verbose;

	if (detail.enactor.template Enact<Config>(detail.d_dest, detail.d_src, detail.num_elements, g_max_ctas)) {
		exit(1);
	}

	detail.enactor.DEBUG = false;

	// Perform the timed number of iterations

	cudaEvent_t start_event, stop_event;
	cudaEventCreate(&start_event);
	cudaEventCreate(&stop_event);

	double elapsed = 0;
	float duration = 0;
	for (int i = 0; i < g_iterations; i++) {

		// Start cuda timing record
		cudaEventRecord(start_event, 0);

		// Call the reduction API routine
		if (detail.enactor.template Enact<Config>(detail.d_dest, detail.d_src, detail.num_elements, g_max_ctas)) {
			exit(1);
		}

		// End cuda timing record
		cudaEventRecord(stop_event, 0);
		cudaEventSynchronize(stop_event);
		cudaEventElapsedTime(&duration, start_event, stop_event);
		elapsed += (double) duration;

		// Flushes any stdio from the GPU
		cudaThreadSynchronize();
	}

	// Display timing information
	double avg_runtime = elapsed / g_iterations;
	double throughput =  0.0;
	if (avg_runtime > 0.0) throughput = ((double) detail.num_elements) / avg_runtime / 1000.0 / 1000.0; 
    printf(", %f, %f, %f, ",
		avg_runtime, throughput, throughput * sizeof(typename Detail::T));
    fflush(stdout);

    // Clean up events
	cudaEventDestroy(start_event);
	cudaEventDestroy(stop_event);

    // Copy out data
    if (B40CPerror(cudaMemcpy(detail.h_data, detail.d_dest, sizeof(typename Detail::T), cudaMemcpyDeviceToHost),
		"TimedReduction cudaMemcpy d_dest failed: ", __FILE__, __LINE__)) exit(1);

    // Verify solution
	CompareResults<typename Detail::T>(detail.h_data, detail.h_reference, 1, true);
	printf("\n");
	fflush(stdout);

}


/******************************************************************************
 * Kernel configuration sweep types
 ******************************************************************************/




template <int CUDA_ARCH, typename T, typename Detail>
struct SweepConfig
{
	enum
	{
		MIN_LOG_THREADS 			= B40C_LOG_WARP_THREADS(CUDA_ARCH),
		MAX_LOG_THREADS 			= B40C_LOG_CTA_THREADS(CUDA_ARCH) + 1,

		MIN_LOG_LOAD_VEC_SIZE 		= 0,
		MAX_LOG_LOAD_VEC_SIZE 		= 2 + 1,

		MIN_LOG_LOADS_PER_TILE 		= 0,
		MAX_LOG_LOADS_PER_TILE 		= 2 + 1,

		MIN_CACHE_MODIFIER 			= (CUDA_ARCH < 200) ? NONE : NONE + 1,
//		MAX_CACHE_MODIFIER 			= (CUDA_ARCH < 200) ? NONE + 1 : LIMIT,			// No cache modifiers exist pre-fermi
		MAX_CACHE_MODIFIER 			= (CUDA_ARCH < 200) ? NONE + 1 : CS,			// CS seems to break things on windows with 128-bit vector loads

		MIN_WORK_STEALING 			= 0,
		MAX_WORK_STEALING 			= (CUDA_ARCH < 200) ? 0 + 1 : 1 + 1,				// Atomics needed for pre-Fermi work-stealing are too painful

		MIN_UNIFORM_GRID_SIZE		= 0,
		MAX_UNIFORM_GRID_SIZE		= 1 + 1,

		MIN_UNIFORM_SMEM_ALLOCATION	= 0,
		MAX_UNIFORM_SMEM_ALLOCATION	= 1 + 1
	};

	// Next UNIFORM_SMEM_ALLOCATION
	template <int LOG_THREADS, int LOG_LOAD_VEC_SIZE, int LOG_LOADS_PER_TILE, int CACHE_MODIFIER, int WORK_STEALING, int UNIFORM_GRID_SIZE, int UNIFORM_SMEM_ALLOCATION,
		int MAX_LOG_THREADS, int MAX_LOG_LOAD_VEC_SIZE, int MAX_LOG_LOADS_PER_TILE, int MAX_CACHE_MODIFIER, int MAX_WORK_STEALING, int MAX_UNIFORM_SMEM_ALLOCATION, int MAX_UNIFORM_GRID_SIZE>
	struct Iterate
	{
		static void Invoke(Detail &detail)
		{
			const int CTA_OCCUPANCY = B40C_MIN(B40C_SM_CTAS(CUDA_ARCH), (B40C_SM_THREADS(CUDA_ARCH)) >> LOG_THREADS);

			// Establish the granularity configuration type
			typedef ReductionConfig <
					T, Detail::OpType::Op, Detail::OpType::Identity,														// Problem type
					size_t, (CacheModifier) CACHE_MODIFIER,	WORK_STEALING, UNIFORM_GRID_SIZE, UNIFORM_SMEM_ALLOCATION,		// Common config
					CTA_OCCUPANCY, LOG_THREADS, LOG_LOAD_VEC_SIZE, LOG_LOADS_PER_TILE, B40C_LOG_WARP_THREADS(200), LOG_LOADS_PER_TILE + LOG_LOAD_VEC_SIZE + LOG_THREADS,	// Upsweep config
					7, 1, 1, B40C_LOG_WARP_THREADS(200)>									// Generic spine config
				Config;

			// Invoke this config
			TimedReduction<Detail, Config>(detail);

			// Next UNIFORM_SMEM_ALLOCATION
			Iterate<LOG_THREADS, LOG_LOAD_VEC_SIZE, LOG_LOADS_PER_TILE, CACHE_MODIFIER, WORK_STEALING, UNIFORM_GRID_SIZE, UNIFORM_SMEM_ALLOCATION + 1,
				MAX_LOG_THREADS, MAX_LOG_LOAD_VEC_SIZE, MAX_LOG_LOADS_PER_TILE, MAX_CACHE_MODIFIER, MAX_WORK_STEALING, MAX_UNIFORM_SMEM_ALLOCATION, MAX_UNIFORM_GRID_SIZE>::Invoke(detail);
		}
	};

	// Last UNIFORM_SMEM_ALLOCATION, next UNIFORM_GRID_SIZE
	template <int LOG_THREADS, int LOG_LOAD_VEC_SIZE, int LOG_LOADS_PER_TILE, int CACHE_MODIFIER, int WORK_STEALING, int UNIFORM_GRID_SIZE,
		int MAX_LOG_THREADS, int MAX_LOG_LOAD_VEC_SIZE, int MAX_LOG_LOADS_PER_TILE, int MAX_CACHE_MODIFIER, int MAX_WORK_STEALING, int MAX_UNIFORM_SMEM_ALLOCATION, int MAX_UNIFORM_GRID_SIZE>
	struct Iterate<LOG_THREADS, LOG_LOAD_VEC_SIZE, LOG_LOADS_PER_TILE, CACHE_MODIFIER, WORK_STEALING, UNIFORM_GRID_SIZE, MAX_UNIFORM_SMEM_ALLOCATION,
		MAX_LOG_THREADS, MAX_LOG_LOAD_VEC_SIZE, MAX_LOG_LOADS_PER_TILE, MAX_CACHE_MODIFIER, MAX_WORK_STEALING, MAX_UNIFORM_SMEM_ALLOCATION, MAX_UNIFORM_GRID_SIZE>
	{
		static void Invoke(Detail &detail)
		{
			// Next UNIFORM_GRID_SIZE (reset UNIFORM_SMEM_ALLOCATION)
			Iterate<LOG_THREADS, LOG_LOAD_VEC_SIZE, LOG_LOADS_PER_TILE, CACHE_MODIFIER, WORK_STEALING, UNIFORM_GRID_SIZE + 1, MIN_UNIFORM_SMEM_ALLOCATION,
				MAX_LOG_THREADS, MAX_LOG_LOAD_VEC_SIZE, MAX_LOG_LOADS_PER_TILE, MAX_CACHE_MODIFIER, MAX_WORK_STEALING, MAX_UNIFORM_SMEM_ALLOCATION, MAX_UNIFORM_GRID_SIZE>::Invoke(detail);
		}
	};

	// Last WORK_STEALING, next UNIFORM_GRID_SIZE
	template <int LOG_THREADS, int LOG_LOAD_VEC_SIZE, int LOG_LOADS_PER_TILE, int CACHE_MODIFIER, int WORK_STEALING, int UNIFORM_SMEM_ALLOCATION,
		int MAX_LOG_THREADS, int MAX_LOG_LOAD_VEC_SIZE, int MAX_LOG_LOADS_PER_TILE, int MAX_CACHE_MODIFIER, int MAX_WORK_STEALING, int MAX_UNIFORM_SMEM_ALLOCATION, int MAX_UNIFORM_GRID_SIZE>
	struct Iterate<LOG_THREADS, LOG_LOAD_VEC_SIZE, LOG_LOADS_PER_TILE, CACHE_MODIFIER, WORK_STEALING, MAX_UNIFORM_GRID_SIZE, UNIFORM_SMEM_ALLOCATION,
		MAX_LOG_THREADS, MAX_LOG_LOAD_VEC_SIZE, MAX_LOG_LOADS_PER_TILE, MAX_CACHE_MODIFIER, MAX_WORK_STEALING, MAX_UNIFORM_SMEM_ALLOCATION, MAX_UNIFORM_GRID_SIZE>
	{
		static void Invoke(Detail &detail)
		{
			// Next UNIFORM_GRID_SIZE (reset MIN_UNIFORM_GRID_SIZE)
			Iterate<LOG_THREADS, LOG_LOAD_VEC_SIZE, LOG_LOADS_PER_TILE, CACHE_MODIFIER, WORK_STEALING + 1, MIN_UNIFORM_GRID_SIZE, MIN_UNIFORM_SMEM_ALLOCATION,
				MAX_LOG_THREADS, MAX_LOG_LOAD_VEC_SIZE, MAX_LOG_LOADS_PER_TILE, MAX_CACHE_MODIFIER, MAX_WORK_STEALING, MAX_UNIFORM_SMEM_ALLOCATION, MAX_UNIFORM_GRID_SIZE>::Invoke(detail);
		}
	};

	// Last WORK_STEALING, next CACHE_MODIFIER
	template <int LOG_THREADS, int LOG_LOAD_VEC_SIZE, int LOG_LOADS_PER_TILE, int CACHE_MODIFIER, int UNIFORM_GRID_SIZE, int UNIFORM_SMEM_ALLOCATION,
		int MAX_LOG_THREADS, int MAX_LOG_LOAD_VEC_SIZE, int MAX_LOG_LOADS_PER_TILE, int MAX_CACHE_MODIFIER, int MAX_WORK_STEALING, int MAX_UNIFORM_SMEM_ALLOCATION, int MAX_UNIFORM_GRID_SIZE>
	struct Iterate<LOG_THREADS, LOG_LOAD_VEC_SIZE, LOG_LOADS_PER_TILE, CACHE_MODIFIER, MAX_WORK_STEALING, UNIFORM_GRID_SIZE, UNIFORM_SMEM_ALLOCATION,
		MAX_LOG_THREADS, MAX_LOG_LOAD_VEC_SIZE, MAX_LOG_LOADS_PER_TILE, MAX_CACHE_MODIFIER, MAX_WORK_STEALING, MAX_UNIFORM_SMEM_ALLOCATION, MAX_UNIFORM_GRID_SIZE>
	{
		static void Invoke(Detail &detail)
		{
			// Next CACHE_MODIFIER (reset WORK_STEALING)
			Iterate<LOG_THREADS, LOG_LOAD_VEC_SIZE, LOG_LOADS_PER_TILE, CACHE_MODIFIER + 1, MIN_WORK_STEALING, MIN_UNIFORM_GRID_SIZE, MIN_UNIFORM_SMEM_ALLOCATION,
				MAX_LOG_THREADS, MAX_LOG_LOAD_VEC_SIZE, MAX_LOG_LOADS_PER_TILE, MAX_CACHE_MODIFIER, MAX_WORK_STEALING, MAX_UNIFORM_SMEM_ALLOCATION, MAX_UNIFORM_GRID_SIZE>::Invoke(detail);
		}
	};

	// Last CACHE_MODIFIER, next LOG_LOADS_PER_TILE
	template <int LOG_THREADS, int LOG_LOAD_VEC_SIZE, int LOG_LOADS_PER_TILE, int WORK_STEALING, int UNIFORM_GRID_SIZE, int UNIFORM_SMEM_ALLOCATION,
		int MAX_LOG_THREADS, int MAX_LOG_LOAD_VEC_SIZE, int MAX_LOG_LOADS_PER_TILE, int MAX_CACHE_MODIFIER, int MAX_WORK_STEALING, int MAX_UNIFORM_SMEM_ALLOCATION, int MAX_UNIFORM_GRID_SIZE>
	struct Iterate<LOG_THREADS, LOG_LOAD_VEC_SIZE, LOG_LOADS_PER_TILE, MAX_CACHE_MODIFIER, WORK_STEALING, UNIFORM_GRID_SIZE, UNIFORM_SMEM_ALLOCATION,
		MAX_LOG_THREADS, MAX_LOG_LOAD_VEC_SIZE, MAX_LOG_LOADS_PER_TILE, MAX_CACHE_MODIFIER, MAX_WORK_STEALING, MAX_UNIFORM_SMEM_ALLOCATION, MAX_UNIFORM_GRID_SIZE>
	{
		static void Invoke(Detail &detail)
		{
			// Next LOG_LOADS_PER_TILE (reset CACHE_MODIFIER)
			Iterate<LOG_THREADS, LOG_LOAD_VEC_SIZE, LOG_LOADS_PER_TILE + 1, MIN_CACHE_MODIFIER, MIN_WORK_STEALING, MIN_UNIFORM_GRID_SIZE, MIN_UNIFORM_SMEM_ALLOCATION,
				MAX_LOG_THREADS, MAX_LOG_LOAD_VEC_SIZE, MAX_LOG_LOADS_PER_TILE, MAX_CACHE_MODIFIER, MAX_WORK_STEALING, MAX_UNIFORM_SMEM_ALLOCATION, MAX_UNIFORM_GRID_SIZE>::Invoke(detail);
		}
	};

	// Last LOG_LOADS_PER_TILE, next LOG_LOAD_VEC_SIZE
	template <int LOG_THREADS, int LOG_LOAD_VEC_SIZE, int CACHE_MODIFIER, int WORK_STEALING, int UNIFORM_GRID_SIZE, int UNIFORM_SMEM_ALLOCATION,
		int MAX_LOG_THREADS, int MAX_LOG_LOAD_VEC_SIZE, int MAX_LOG_LOADS_PER_TILE, int MAX_CACHE_MODIFIER, int MAX_WORK_STEALING, int MAX_UNIFORM_SMEM_ALLOCATION, int MAX_UNIFORM_GRID_SIZE>
	struct Iterate<LOG_THREADS, LOG_LOAD_VEC_SIZE, MAX_LOG_LOADS_PER_TILE, CACHE_MODIFIER, WORK_STEALING, UNIFORM_GRID_SIZE, UNIFORM_SMEM_ALLOCATION,
		MAX_LOG_THREADS, MAX_LOG_LOAD_VEC_SIZE, MAX_LOG_LOADS_PER_TILE, MAX_CACHE_MODIFIER, MAX_WORK_STEALING, MAX_UNIFORM_SMEM_ALLOCATION, MAX_UNIFORM_GRID_SIZE>
	{
		static void Invoke(Detail &detail)
		{
			// Next LOG_LOAD_VEC_SIZE (reset LOG_LOADS_PER_TILE)
			Iterate<LOG_THREADS, LOG_LOAD_VEC_SIZE + 1, MIN_LOG_LOADS_PER_TILE, MIN_CACHE_MODIFIER, MIN_WORK_STEALING, MIN_UNIFORM_GRID_SIZE, MIN_UNIFORM_SMEM_ALLOCATION,
				MAX_LOG_THREADS, MAX_LOG_LOAD_VEC_SIZE, MAX_LOG_LOADS_PER_TILE, MAX_CACHE_MODIFIER, MAX_WORK_STEALING, MAX_UNIFORM_SMEM_ALLOCATION, MAX_UNIFORM_GRID_SIZE>::Invoke(detail);
		}
	};

	// Last LOG_LOAD_VEC_SIZE, next LOG_THREADS
	template <int LOG_THREADS, int LOG_LOADS_PER_TILE, int CACHE_MODIFIER, int WORK_STEALING, int UNIFORM_GRID_SIZE, int UNIFORM_SMEM_ALLOCATION,
		int MAX_LOG_THREADS, int MAX_LOG_LOAD_VEC_SIZE, int MAX_LOG_LOADS_PER_TILE, int MAX_CACHE_MODIFIER, int MAX_WORK_STEALING, int MAX_UNIFORM_SMEM_ALLOCATION, int MAX_UNIFORM_GRID_SIZE>
	struct Iterate<LOG_THREADS, MAX_LOG_LOAD_VEC_SIZE, LOG_LOADS_PER_TILE, CACHE_MODIFIER, WORK_STEALING, UNIFORM_GRID_SIZE, UNIFORM_SMEM_ALLOCATION,
		MAX_LOG_THREADS, MAX_LOG_LOAD_VEC_SIZE, MAX_LOG_LOADS_PER_TILE, MAX_CACHE_MODIFIER, MAX_WORK_STEALING, MAX_UNIFORM_SMEM_ALLOCATION, MAX_UNIFORM_GRID_SIZE>
	{
		static void Invoke(Detail &detail)
		{
			// Next LOG_THREADS (reset LOG_LOAD_VEC_SIZE)
			Iterate<LOG_THREADS + 1, MIN_LOG_LOAD_VEC_SIZE, MIN_LOG_LOADS_PER_TILE, MIN_CACHE_MODIFIER, MIN_WORK_STEALING, MIN_UNIFORM_GRID_SIZE, MIN_UNIFORM_SMEM_ALLOCATION,
				MAX_LOG_THREADS, MAX_LOG_LOAD_VEC_SIZE, MAX_LOG_LOADS_PER_TILE, MAX_CACHE_MODIFIER, MAX_WORK_STEALING, MAX_UNIFORM_SMEM_ALLOCATION, MAX_UNIFORM_GRID_SIZE>::Invoke(detail);
		}
	};

	// Last LOG_THREADS
	template <int LOG_LOAD_VEC_SIZE, int LOG_LOADS_PER_TILE, int CACHE_MODIFIER, int WORK_STEALING, int UNIFORM_GRID_SIZE, int UNIFORM_SMEM_ALLOCATION,
		int MAX_LOG_THREADS, int MAX_LOG_LOAD_VEC_SIZE, int MAX_LOG_LOADS_PER_TILE, int MAX_CACHE_MODIFIER, int MAX_WORK_STEALING, int MAX_UNIFORM_SMEM_ALLOCATION, int MAX_UNIFORM_GRID_SIZE>
	struct Iterate<MAX_LOG_THREADS, LOG_LOAD_VEC_SIZE, LOG_LOADS_PER_TILE, CACHE_MODIFIER, WORK_STEALING, UNIFORM_GRID_SIZE, UNIFORM_SMEM_ALLOCATION,
		MAX_LOG_THREADS, MAX_LOG_LOAD_VEC_SIZE, MAX_LOG_LOADS_PER_TILE, MAX_CACHE_MODIFIER, MAX_WORK_STEALING, MAX_UNIFORM_SMEM_ALLOCATION, MAX_UNIFORM_GRID_SIZE>
	{
		static void Invoke(Detail &detail) {}
	};

	// Interface
	static void Invoke(Detail &detail)
	{
		Iterate<MIN_LOG_THREADS, MIN_LOG_LOAD_VEC_SIZE, MIN_LOG_LOADS_PER_TILE, MIN_CACHE_MODIFIER, MIN_WORK_STEALING, MIN_UNIFORM_SMEM_ALLOCATION, MIN_UNIFORM_GRID_SIZE,
			MAX_LOG_THREADS, MAX_LOG_LOAD_VEC_SIZE, MAX_LOG_LOADS_PER_TILE, MAX_CACHE_MODIFIER, MAX_WORK_STEALING, MAX_UNIFORM_SMEM_ALLOCATION, MAX_UNIFORM_GRID_SIZE>::Invoke(detail);
	}
};




/******************************************************************************
 * ReductionTuner
 ******************************************************************************/

template <typename T>
struct Sum
{
	static __host__ __device__ __forceinline__ T Op(const T &a, const T &b)
	{
		return a + b;
	}

	static __host__ __device__ __forceinline__ T Identity()
	{
		return 0;
	}
};

template <typename T>
struct Max
{
	static __host__ __device__ __forceinline__ T Op(const T &a, const T &b)
	{
		return (a > b) ? a : b;
	}

	static __host__ __device__ __forceinline__ T Identity()
	{
		return 0;
	}
};


class ReductionTuner : public Architecture<__B40C_CUDA_ARCH__, ReductionTuner>
{
	typedef Architecture<__B40C_CUDA_ARCH__, ReductionTuner> 	BaseArchType;

	// Device properties
	const CudaProperties cuda_props;

public:

	// Constructor
	ReductionTuner() {}

	// Return the current device's sm version
	int PtxVersion()
	{
		return cuda_props.device_sm_version;
	}

	// Dispatch call-back with static CUDA_ARCH
	template <int CUDA_ARCH, typename Storage, typename Detail>
	cudaError_t Enact(Storage &problem_storage, Detail &detail)
	{
		// Run the timing tests
		SweepConfig<CUDA_ARCH, Storage, Detail>::Invoke(detail);
		return cudaSuccess;
	}

	/**
	 * Creates an example reduction problem and then dispatches the problem
	 * to the GPU for the given number of iterations, displaying runtime information.
	 */
	template<typename T, typename OpType>
	void TestReduction(size_t num_elements)
	{
		// Allocate storage and enactor
		Detail<T, OpType> detail(num_elements);

		if (B40CPerror(cudaMalloc((void**) &detail.d_src, sizeof(T) * num_elements),
			"TimedReduction cudaMalloc d_src failed: ", __FILE__, __LINE__)) exit(1);

		if (B40CPerror(cudaMalloc((void**) &detail.d_dest, sizeof(T)),
			"TimedReduction cudaMalloc d_dest failed: ", __FILE__, __LINE__)) exit(1);

		if ((detail.h_data = (T*) malloc(num_elements * sizeof(T))) == NULL) {
			fprintf(stderr, "Host malloc of problem data failed\n");
			exit(1);
		}
		if ((detail.h_reference = (T*) malloc(sizeof(T))) == NULL) {
			fprintf(stderr, "Host malloc of problem data failed\n");
			exit(1);
		}

		detail.h_reference[0] = OpType::Identity();
		for (size_t i = 0; i < num_elements; ++i) {
			// RandomBits<T>(detail.h_data[i], 0);
			detail.h_data[i] = i;
			detail.h_reference[0] = OpType::Op(detail.h_reference[0], detail.h_data[i]);
		}

		// Move a fresh copy of the problem into device storage
		if (B40CPerror(cudaMemcpy(detail.d_src, detail.h_data, sizeof(T) * num_elements, cudaMemcpyHostToDevice),
			"TimedReduction cudaMemcpy d_src failed: ", __FILE__, __LINE__)) exit(1);

		// Have the base class call back with a constant dispatch-arch for our current device
		T dummy;
		BaseArchType::Enact(dummy, detail);

	    // Free allocated memory
	    if (detail.d_src) cudaFree(detail.d_src);
	    if (detail.d_dest) cudaFree(detail.d_dest);

		// Free our allocated host memory
		if (detail.h_data) free(detail.h_data);
	    if (detail.h_reference) free(detail.h_reference);
	}

};



/******************************************************************************
 * Main
 ******************************************************************************/

int main(int argc, char** argv)
{

	CommandLineArgs args(argc, argv);
	DeviceInit(args);

	//srand(time(NULL));	
	srand(0);				// presently deterministic

    size_t num_elements 								= 1024;

	// Check command line arguments
    if (args.CheckCmdLineFlag("help")) {
		Usage();
		return 0;
	}

    args.GetCmdLineArgument("i", g_iterations);
    args.GetCmdLineArgument("n", num_elements);
    args.GetCmdLineArgument("max-ctas", g_max_ctas);
	g_verbose = args.CheckCmdLineFlag("v");

	CudaProperties cuda_props;

	printf("Test Reduction: %d iterations, %d elements", g_iterations, num_elements);
	printf("\nCodeGen: \t[device_sm_version: %d, kernel_ptx_version: %d]\n\n",
		cuda_props.device_sm_version, cuda_props.kernel_ptx_version);

	printf("SizeT bytes, CACHE_MODIFIER, WORK_STEALING, UNIFORM_SMEM_ALLOCATION, UNIFORM_GRID_SIZE, "
		"UPSWEEP_CTA_OCCUPANCY, UPSWEEP_LOG_THREADS, UPSWEEP_LOG_LOAD_VEC_SIZE, UPSWEEP_LOG_LOADS_PER_TILE, UPSWEEP_LOG_RAKING_THREADS, UPSWEEP_LOG_SCHEDULE_GRANULARITY, "
		"SPINE_LOG_THREADS, SPINE_LOG_LOAD_VEC_SIZE, SPINE_LOG_LOADS_PER_TILE, SPINE_LOG_RAKING_THREADS, "
		"elapsed time (ms), throughput (10^9 items/s), bandwidth (10^9 B/s)\n");

	ReductionTuner tuner;

//	typedef unsigned short T;
//	typedef unsigned char T;
	typedef unsigned int T;
//	typedef unsigned long long T;

	// Execute test(s)
	tuner.TestReduction<T, Sum<T>>(num_elements * sizeof(num_elements) / 4);

	return 0;
}



