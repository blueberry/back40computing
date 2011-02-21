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
 * Tuning tool for establishing optimal memcopy granularity configuration types
 ******************************************************************************/

#include <stdio.h> 

// Memcopy includes
#include "memcopy_api_granularity.cuh"
#include "memcopy_api_enactor.cuh"

// Test utils
#include "b40c_util.h"

using namespace b40c;
using namespace memcopy;

/******************************************************************************
 * Defines, constants, globals, and utility types
 ******************************************************************************/

bool g_verbose;
int g_max_ctas = 0;
int g_iterations = 0;


template <typename T>
struct Detail
{
	MemcopyEnactor<> memcopy_enactor;
	T *d_dest;
	T *d_src;
	size_t num_elements;

	Detail(size_t num_elements) : d_dest(NULL), d_src(NULL), num_elements(num_elements) {}
};


/******************************************************************************
 * Utility routines
 ******************************************************************************/

/**
 * Displays the commandline usage for this tool
 */
void Usage()
{
	printf("\ntune_memcopy_large [--device=<device index>] [--v] [--i=<num-iterations>] "
			"[--max-ctas=<max-thread-blocks>] [--n=<num-elements>]\n");
	printf("\n");
	printf("\t--v\tDisplays copied results to the console.\n");
	printf("\n");
	printf("\t--i\tPerforms the memcopy operation <num-iterations> times\n");
	printf("\t\t\ton the device. Re-copies original input each time. Default = 1\n");
	printf("\n");
	printf("\t--n\tThe number of elements to comprise the sample problem\n");
	printf("\t\t\tDefault = 512\n");
	printf("\n");
}


/**
 * Timed memcopy for a specific granularity configuration type
 */
template <typename Config>
void TimedMemcopy(Detail<typename Config::T> &detail)
{
	Config::Print();

	// Perform a single iteration to allocate any memory if needed, prime code caches, etc.
	detail.memcopy_enactor.DEBUG = g_verbose;
	detail.memcopy_enactor.template Enact<Config>(detail.d_dest, detail.d_src, detail.num_elements, g_max_ctas);
	detail.memcopy_enactor.DEBUG = false;

	// Perform the timed number of iterations

	cudaEvent_t start_event, stop_event;
	cudaEventCreate(&start_event);
	cudaEventCreate(&stop_event);

	double elapsed = 0;
	float duration = 0;
	for (int i = 0; i < g_iterations; i++) {

		// Start cuda timing record
		cudaEventRecord(start_event, 0);

		// Call the memcopy API routine
		detail.memcopy_enactor.template Enact<Config>(detail.d_dest, detail.d_src, detail.num_elements, g_max_ctas);

		// End cuda timing record
		cudaEventRecord(stop_event, 0);
		cudaEventSynchronize(stop_event);
		cudaEventElapsedTime(&duration, start_event, stop_event);
		elapsed += (double) duration;
	}

	// Flushes any stdio from the GPU
	cudaThreadSynchronize();

	// Display timing information
	double avg_runtime = elapsed / g_iterations;
	double throughput = ((double) detail.num_elements) / avg_runtime / 1000.0 / 1000.0;
    printf(", %f, %f, %f\n",
		avg_runtime, throughput, 2 * throughput * sizeof(Config::T));
    fflush(stdout);

    // Clean up events
	cudaEventDestroy(start_event);
	cudaEventDestroy(stop_event);
}


/******************************************************************************
 * Kernel configuration sweep types
 ******************************************************************************/

enum Ranges
{
	MIN_LOG_THREADS 			= 5,
	MAX_LOG_THREADS 			= 10 + 1,

	MIN_LOG_LOAD_VEC_SIZE 		= 0,
	MAX_LOG_LOAD_VEC_SIZE 		= 2 + 1,

	MIN_LOG_LOADS_PER_TILE 		= 0,
	MAX_LOG_LOADS_PER_TILE 		= 2 + 1,

	MIN_CACHE_MODIFIER 			= NONE,
	MAX_CACHE_MODIFIER 			= LIMIT,

	MIN_WORK_STEALING 			= 0,
	MAX_WORK_STEALING 			= 1 + 1
};


template <int CUDA_ARCH, typename T>
struct SweepConfig
{
	// Next WORK_STEALING
	template <int LOG_THREADS, int LOG_LOAD_VEC_SIZE, int LOG_LOADS_PER_TILE, int CACHE_MODIFIER, int WORK_STEALING>
	struct Iterate
	{
		static void Invoke(Detail<T> &detail)
		{
			// Invoke this config
			const int CTA_OCCUPANCY = B40C_MIN(B40C_SM_CTAS(CUDA_ARCH), (B40C_SM_THREADS(CUDA_ARCH)) >> LOG_THREADS);
			typedef MemcopyConfig<typename T, CTA_OCCUPANCY, LOG_THREADS, LOG_LOAD_VEC_SIZE, LOG_LOADS_PER_TILE, (CacheModifier) CACHE_MODIFIER, WORK_STEALING> Config;
			TimedMemcopy<Config>(detail);

			// Next WORK_STEALING
			Iterate<LOG_THREADS, LOG_LOAD_VEC_SIZE, LOG_LOADS_PER_TILE, CACHE_MODIFIER, WORK_STEALING + 1>::Invoke(detail);
		}
	};

	// Last WORK_STEALING, next CACHE_MODIFIER
	template <int LOG_THREADS, int LOG_LOAD_VEC_SIZE, int LOG_LOADS_PER_TILE, int CACHE_MODIFIER>
	struct Iterate<LOG_THREADS, LOG_LOAD_VEC_SIZE, LOG_LOADS_PER_TILE, CACHE_MODIFIER, MAX_WORK_STEALING>
	{
		static void Invoke(Detail<T> &detail)
		{
			// Next CACHE_MODIFIER (reset WORK_STEALING)
			Iterate<LOG_THREADS, LOG_LOAD_VEC_SIZE, LOG_LOADS_PER_TILE, CACHE_MODIFIER + 1, MIN_WORK_STEALING>::Invoke(detail);
		}
	};

	// Last CACHE_MODIFIER, next LOG_LOADS_PER_TILE
	template <int LOG_THREADS, int LOG_LOAD_VEC_SIZE, int LOG_LOADS_PER_TILE, int WORK_STEALING>
	struct Iterate<LOG_THREADS, LOG_LOAD_VEC_SIZE, LOG_LOADS_PER_TILE, MAX_CACHE_MODIFIER, WORK_STEALING>
	{
		static void Invoke(Detail<T> &detail)
		{
			// Next LOG_LOADS_PER_TILE (reset CACHE_MODIFIER)
			Iterate<LOG_THREADS, LOG_LOAD_VEC_SIZE, LOG_LOADS_PER_TILE + 1, MIN_CACHE_MODIFIER, MIN_WORK_STEALING>::Invoke(detail);
		}
	};

	// Last LOG_LOADS_PER_TILE, next LOG_LOAD_VEC_SIZE
	template <int LOG_THREADS, int LOG_LOAD_VEC_SIZE, int CACHE_MODIFIER, int WORK_STEALING>
	struct Iterate<LOG_THREADS, LOG_LOAD_VEC_SIZE, MAX_LOG_LOADS_PER_TILE, CACHE_MODIFIER, WORK_STEALING>
	{
		static void Invoke(Detail<T> &detail)
		{
			// Next LOG_LOAD_VEC_SIZE (reset LOG_LOADS_PER_TILE)
			Iterate<LOG_THREADS, LOG_LOAD_VEC_SIZE + 1, MIN_LOG_LOADS_PER_TILE, MIN_CACHE_MODIFIER, MIN_WORK_STEALING>::Invoke(detail);
		}
	};

	// Last LOG_LOAD_VEC_SIZE, next LOG_THREADS
	template <int LOG_THREADS, int LOG_LOADS_PER_TILE, int CACHE_MODIFIER, int WORK_STEALING>
	struct Iterate<LOG_THREADS, MAX_LOG_LOAD_VEC_SIZE, LOG_LOADS_PER_TILE, CACHE_MODIFIER, WORK_STEALING>
	{
		static void Invoke(Detail<T> &detail)
		{
			// Next LOG_THREADS (reset LOG_LOAD_VEC_SIZE)
			Iterate<LOG_THREADS + 1, MIN_LOG_LOAD_VEC_SIZE, MIN_LOG_LOADS_PER_TILE, MIN_CACHE_MODIFIER, MIN_WORK_STEALING>::Invoke(detail);
		}
	};

	// Last LOG_THREADS
	template <int LOG_LOAD_VEC_SIZE, int LOG_LOADS_PER_TILE, int CACHE_MODIFIER, int WORK_STEALING>
	struct Iterate<MAX_LOG_THREADS, LOG_LOAD_VEC_SIZE, LOG_LOADS_PER_TILE, CACHE_MODIFIER, WORK_STEALING>
	{
		static void Invoke(Detail<T> &detail) {}
	};

	// Interface
	static void Invoke(Detail<T> &detail)
	{
		Iterate<MIN_LOG_THREADS, MIN_LOG_LOAD_VEC_SIZE, MIN_LOG_LOADS_PER_TILE, MIN_CACHE_MODIFIER, MIN_WORK_STEALING>::Invoke(detail);
	}
};




/******************************************************************************
 * MemcopyTuner
 ******************************************************************************/

class MemcopyTuner : public Architecture<__B40C_CUDA_ARCH__, MemcopyTuner>
{
	typedef Architecture<__B40C_CUDA_ARCH__, MemcopyTuner> 	BaseArchType;

	// Device properties
	const CudaProperties cuda_props;

public:

	// Constructor
	MemcopyTuner() {}

	// The arch version of the code for the current device that actually have
	// compiled kernels for
	int PtxVersion()
	{
		return cuda_props.kernel_ptx_version;
	}

	// Dispatch call-back with static CUDA_ARCH
	template <int CUDA_ARCH, typename Storage, typename Detail>
	cudaError_t Enact(Storage &problem_storage, Detail &detail)
	{
		// Run the timing tests
		printf("\n");
		printf("sizeof(T), CTA_OCCUPANCY, LOG_THREADS, LOG_LOAD_VEC_SIZE, LOG_LOADS_PER_TILE, "
				"CACHE_MODIFIER, WORK_STEALING, LOG_SCHEDULE_GRANULARITY, "
				"elapsed time (ms), throughput (10^9 items/s), bandwidth (10^9 B/s)\n");
		SweepConfig<CUDA_ARCH, Storage>::Invoke(detail);

		return cudaSuccess;
	}

	/**
	 * Creates an example memcopy problem and then dispatches the problem
	 * to the GPU for the given number of iterations, displaying runtime information.
	 *
	 * @param[in] 		num_elements
	 * 		Size in elements of the vector to copy
	 */
	template<typename T>
	void TestMemcopy(size_t num_elements)
	{
		printf("CodeGen: \t[device_sm_version: %d, kernel_ptx_version: %d]\n",
			cuda_props.device_sm_version, cuda_props.kernel_ptx_version);

		// Allocate the memcopy problem on the host and fill the keys with random bytes

		T *h_data 			= (T*) malloc(num_elements * sizeof(T));
		T *h_reference 		= (T*) malloc(num_elements * sizeof(T));

		if ((h_data == NULL) || (h_reference == NULL)){
			fprintf(stderr, "Host malloc of problem data failed\n");
			exit(1);
		}

		for (size_t i = 0; i < num_elements; ++i) {
//			RandomBits<T>(h_data[i], 0);
			h_data[i] = i;
			h_reference[i] = h_data[i];
		}

		printf("%d iterations, %d elements", g_iterations, num_elements);

		// Allocate device storage and enactor
		Detail<T> detail(num_elements);
		if (B40CPerror(cudaMalloc((void**) &detail.d_src, sizeof(T) * num_elements),
			"TimedMemcopy cudaMalloc d_src failed: ", __FILE__, __LINE__)) exit(1);
		if (B40CPerror(cudaMalloc((void**) &detail.d_dest, sizeof(T) * num_elements),
			"TimedMemcopy cudaMalloc d_dest failed: ", __FILE__, __LINE__)) exit(1);

		// Move a fresh copy of the problem into device storage
		if (B40CPerror(cudaMemcpy(detail.d_src, h_data, sizeof(T) * num_elements, cudaMemcpyHostToDevice),
			"TimedMemcopy cudaMemcpy d_src failed: ", __FILE__, __LINE__)) exit(1);

		T dummy;
		BaseArchType::Enact(dummy, detail);

	    // Copy out data
	    if (B40CPerror(cudaMemcpy(h_data, detail.d_dest, sizeof(T) * num_elements, cudaMemcpyDeviceToHost),
			"TimedMemcopy cudaMemcpy d_dest failed: ", __FILE__, __LINE__)) exit(1);

	    // Free allocated memory
	    if (detail.d_src) cudaFree(detail.d_src);
	    if (detail.d_dest) cudaFree(detail.d_dest);

		// Display copied data
		if (g_verbose) {
			printf("\n\nData:\n");
			for (int i = 0; i < num_elements; i++) {
				PrintValue<T>(h_data[i]);
				printf(", ");
			}
			printf("\n\n");
		}

	    // Verify solution
		CompareResults<T>(h_data, h_reference, num_elements, true);
		printf("\n");
		fflush(stdout);

		// Free our allocated host memory
		if (h_data) free(h_data);
	    if (h_reference) free(h_reference);
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

    int num_elements 					= 1024;

	// Check command line arguments
    if (args.CheckCmdLineFlag("help")) {
		Usage();
		return 0;
	}

    args.GetCmdLineArgumenti("i", g_iterations);
    args.GetCmdLineArgumenti("n", num_elements);
    args.GetCmdLineArgumenti("max-ctas", g_max_ctas);
	g_verbose = args.CheckCmdLineFlag("v");

	MemcopyTuner tuner;

	// Execute test(s)
	tuner.TestMemcopy<unsigned char>(num_elements * 4);
//	tuner.TestMemcopy<unsigned short>(num_elements * 2);
//	tuner.TestMemcopy<unsigned int>(num_elements);
//	tuner.TestMemcopy<unsigned long long>(num_elements / 2);

	return 0;
}



