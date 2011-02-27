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
#include "b40c_numeric_traits.cuh"
#include "b40c_parameter_generation.cuh"

using namespace b40c;
using namespace traits;
using namespace reduction;

/******************************************************************************
 * Defines, constants, globals, and utility types
 ******************************************************************************/

bool g_verbose;
int g_max_ctas = 0;
int g_iterations = 0;


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
 * Timed reduction for applying a specific granularity configuration type
 */
template <typename TuneProblemDetail, typename Config>
void TimedReduction(TuneProblemDetail &detail)
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
		avg_runtime, throughput, throughput * sizeof(typename TuneProblemDetail::T));
    fflush(stdout);

    // Clean up events
	cudaEventDestroy(start_event);
	cudaEventDestroy(stop_event);

    // Copy out data
    if (B40CPerror(cudaMemcpy(detail.h_data, detail.d_dest, sizeof(typename TuneProblemDetail::T), cudaMemcpyDeviceToHost),
		"TimedReduction cudaMemcpy d_dest failed: ", __FILE__, __LINE__)) exit(1);

    // Verify solution
	CompareResults<typename TuneProblemDetail::T>(detail.h_data, detail.h_reference, 1, true);
	printf("\n");
	fflush(stdout);

}


/******************************************************************************
 * Tuning Parameter Enumerations and Ranges
 ******************************************************************************/

/**
 * Enumerated tuning params
 */
enum TuningParam {
	WORK_STEALING = 0,
	UNIFORM_SMEM_ALLOCATION,
	UNIFORM_GRID_SIZE,
	OVERSUBSCRIBED_GRID_SIZE,
	UPSWEEP_LOG_THREADS,
	UPSWEEP_LOG_LOAD_VEC_SIZE,
	UPSWEEP_LOG_LOADS_PER_TILE,

	PARAM_LIMIT,

	// Parameters below here are currently not part of the tuning sweep

	// These can be tuned, but we're currently not compelled to
	READ_MODIFIER,
	WRITE_MODIFIER,
	UPSWEEP_LOG_RAKING_THREADS,

	// Derive these from the others above
	UPSWEEP_CTA_OCCUPANCY,
	UPSWEEP_LOG_SCHEDULE_GRANULARITY,

	// General performance is insensitive to the spine kernel params
	// because it's only a single-CTA: we'll just use reasonable defaults
	SPINE_LOG_THREADS,
	SPINE_LOG_LOAD_VEC_SIZE,
	SPINE_LOG_LOADS_PER_TILE,
	SPINE_LOG_RAKING_THREADS
};


/**
 * Ranges for the tuning params
 */
template <int CUDA_ARCH, typename ProblemDetail, typename ParamList, int PARAM> struct Ranges;

// READ_MODIFIER
template <int CUDA_ARCH, typename ProblemDetail, typename ParamList>
struct Ranges<CUDA_ARCH, ProblemDetail, ParamList, READ_MODIFIER> {
	typedef typename ProblemDetail::T T;
	enum {
		MIN = NONE,
		MAX = ((CUDA_ARCH < 200) || (NumericTraits<T>::REPRESENTATION == NAN)) ? NONE : CS		// No type modifiers for pre-Fermi or non-builtin types
	};
};

// WRITE_MODIFIER
template <int CUDA_ARCH, typename ProblemDetail, typename ParamList>
struct Ranges<CUDA_ARCH, ProblemDetail, ParamList, WRITE_MODIFIER> {
	typedef typename ProblemDetail::T T;
	enum {
		MIN = NONE,
		MAX = ((CUDA_ARCH < 200) || (NumericTraits<T>::REPRESENTATION == NAN)) ? NONE : CS		// No type modifiers for pre-Fermi or non-builtin types
	};
};

// WORK_STEALING
template <int CUDA_ARCH, typename ProblemDetail, typename ParamList>
struct Ranges<CUDA_ARCH, ProblemDetail, ParamList, WORK_STEALING> {
	enum {
		MIN = 0,
		MAX = (CUDA_ARCH < 200) ? 0 : 1			// No workstealing pre-Fermi
	};
};

// UNIFORM_SMEM_ALLOCATION
template <int CUDA_ARCH, typename ProblemDetail, typename ParamList>
struct Ranges<CUDA_ARCH, ProblemDetail, ParamList, UNIFORM_SMEM_ALLOCATION> {
	enum {
		MIN = 0,
		MAX = 1
	};
};

// UNIFORM_GRID_SIZE
template <int CUDA_ARCH, typename ProblemDetail, typename ParamList>
struct Ranges<CUDA_ARCH, ProblemDetail, ParamList, UNIFORM_GRID_SIZE> {
	enum {
		MIN = 0,
		MAX = 1
	};
};

// OVERSUBSCRIBED_GRID_SIZE
template <int CUDA_ARCH, typename ProblemDetail, typename ParamList>
struct Ranges<CUDA_ARCH, ProblemDetail, ParamList, OVERSUBSCRIBED_GRID_SIZE> {
	enum {
		MIN = 0,
		MAX = 1
	};
};

// UPSWEEP_LOG_THREADS
template <int CUDA_ARCH, typename ProblemDetail, typename ParamList>
struct Ranges<CUDA_ARCH, ProblemDetail, ParamList, UPSWEEP_LOG_THREADS> {
	enum {
		MIN = B40C_LOG_WARP_THREADS(CUDA_ARCH),
		MAX = B40C_LOG_CTA_THREADS(CUDA_ARCH)
	};
};

// UPSWEEP_LOG_LOAD_VEC_SIZE
template <int CUDA_ARCH, typename ProblemDetail, typename ParamList>
struct Ranges<CUDA_ARCH, ProblemDetail, ParamList, UPSWEEP_LOG_LOAD_VEC_SIZE> {
	enum {
		MIN = 0,
		MAX = 2
	};
};

// UPSWEEP_LOG_LOADS_PER_TILE
template <int CUDA_ARCH, typename ProblemDetail, typename ParamList>
struct Ranges<CUDA_ARCH, ProblemDetail, ParamList, UPSWEEP_LOG_LOADS_PER_TILE> {
	enum {
		MIN = 0,
		MAX = 2
	};
};

// UPSWEEP_LOG_RAKING_THREADS
template <int CUDA_ARCH, typename ProblemDetail, typename ParamList>
struct Ranges<CUDA_ARCH, ProblemDetail, ParamList, UPSWEEP_LOG_RAKING_THREADS> {
	enum {
		MIN = B40C_LOG_WARP_THREADS(CUDA_ARCH),
		MAX = ParamList::template Access<UPSWEEP_LOG_THREADS>::VALUE
	};
};



/******************************************************************************
 * Tuning Parameter Enumerations and Ranges
 ******************************************************************************/


/**
 * Encapsulation structure for
 * 		- Wrapping problem type and storage
 * 		- Providing call-back for parameter-list generation
 */
template <typename _T, typename _OpType>
struct TuneProblemDetail
{
	typedef _T T;
	typedef _OpType OpType;

	ReductionEnactor<> enactor;
	T *d_dest;
	T *d_src;
	T *h_data;
	T *h_reference;
	size_t num_elements;

	/**
	 * Constructor
	 */
	TuneProblemDetail(size_t num_elements) :
		d_dest(NULL), d_src(NULL), h_data(NULL), h_reference(NULL), num_elements(num_elements) {}

	/**
	 * Callback invoked by parameter-list generation
	 */
	template <int CUDA_ARCH, typename ParamList>
	void Invoke()
	{
		const int C_READ_MODIFIER =
//			ParamList::template Access<READ_MODIFIER>::VALUE;					// These can be tuned, but we're currently not compelled to
			NONE;
		const int C_WRITE_MODIFIER =
//			ParamList::template Access<WRITE_MODIFIER>::VALUE;					// These can be tuned, but we're currently not compelled to
			NONE;
		const int C_WORK_STEALING =
			ParamList::template Access<WORK_STEALING>::VALUE;
		const int C_UNIFORM_SMEM_ALLOCATION =
			ParamList::template Access<UNIFORM_SMEM_ALLOCATION>::VALUE;
		const int C_UNIFORM_GRID_SIZE =
			ParamList::template Access<UNIFORM_GRID_SIZE>::VALUE;
		const int C_OVERSUBSCRIBED_GRID_SIZE =
			ParamList::template Access<OVERSUBSCRIBED_GRID_SIZE>::VALUE;
		const int C_UPSWEEP_LOG_THREADS =
			ParamList::template Access<UPSWEEP_LOG_THREADS>::VALUE;
		const int C_UPSWEEP_LOG_LOAD_VEC_SIZE =
			ParamList::template Access<UPSWEEP_LOG_LOAD_VEC_SIZE>::VALUE;
		const int C_UPSWEEP_LOG_LOADS_PER_TILE =
			ParamList::template Access<UPSWEEP_LOG_LOADS_PER_TILE>::VALUE;
		const int C_UPSWEEP_LOG_RAKING_THREADS =
//			ParamList::template Access<UPSWEEP_LOG_RAKING_THREADS>::VALUE;		// These can be tuned, but we're currently not compelled to
			B40C_LOG_WARP_THREADS(CUDA_ARCH);

		const int C_UPSWEEP_CTA_OCCUPANCY = B40C_MIN(
			B40C_SM_CTAS(CUDA_ARCH),
			(B40C_SM_THREADS(CUDA_ARCH)) >> C_UPSWEEP_LOG_THREADS);
		const int C_UPSWEEP_LOG_SCHEDULE_GRANULARITY =
			C_UPSWEEP_LOG_LOADS_PER_TILE +
			C_UPSWEEP_LOG_LOAD_VEC_SIZE +
			C_UPSWEEP_LOG_THREADS;

		// General performance is insensitive to spine config it's only a single-CTA:
		// simply use reasonable defaults
		const int C_SPINE_LOG_THREADS = 8;
		const int C_SPINE_LOG_LOAD_VEC_SIZE = 0;
		const int C_SPINE_LOG_LOADS_PER_TILE = 1;
		const int C_SPINE_LOG_RAKING_THREADS = B40C_LOG_WARP_THREADS(CUDA_ARCH);
		
		// Establish the problem type
		typedef ReductionProblem<
			typename TuneProblemDetail::T,
			size_t,
			TuneProblemDetail::OpType::Op,
			TuneProblemDetail::OpType::Identity> Problem;

		// Establish the granularity configuration type
		typedef ReductionConfig <Problem,
			(CacheModifier) C_READ_MODIFIER,
			(CacheModifier) C_WRITE_MODIFIER,
			C_WORK_STEALING,
			C_UNIFORM_SMEM_ALLOCATION,
			C_UNIFORM_GRID_SIZE,
			C_OVERSUBSCRIBED_GRID_SIZE,
			C_UPSWEEP_CTA_OCCUPANCY,
			C_UPSWEEP_LOG_THREADS,
			C_UPSWEEP_LOG_LOAD_VEC_SIZE,
			C_UPSWEEP_LOG_LOADS_PER_TILE,
			C_UPSWEEP_LOG_RAKING_THREADS,
			C_UPSWEEP_LOG_SCHEDULE_GRANULARITY,
			C_SPINE_LOG_THREADS, 
			C_SPINE_LOG_LOAD_VEC_SIZE, 
			C_SPINE_LOG_LOADS_PER_TILE, 
			C_SPINE_LOG_RAKING_THREADS> Config;		

		// Invoke this config
		TimedReduction<TuneProblemDetail, Config>(*this);
	}
};



/**
 * Reduction Tuner
 */
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
	template <int CUDA_ARCH, typename Storage, typename TuneProblemDetail>
	cudaError_t Enact(Storage &problem_storage, TuneProblemDetail &detail)
	{
		// Run the timing tests
		ParamListSweep<
			CUDA_ARCH,
			TuneProblemDetail,
			0,
			PARAM_LIMIT,
			Ranges>::template Invoke<void>(detail);
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
		TuneProblemDetail<T, OpType> detail(num_elements);

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

	printf("READ_MODIFIER, WRITE_MODIFIER, WORK_STEALING, UNIFORM_SMEM_ALLOCATION, UNIFORM_GRID_SIZE, OVERSUBSCRIBED_GRID_SIZE, "
		"UPSWEEP_CTA_OCCUPANCY, UPSWEEP_LOG_THREADS, UPSWEEP_LOG_LOAD_VEC_SIZE, UPSWEEP_LOG_LOADS_PER_TILE, UPSWEEP_LOG_RAKING_THREADS, UPSWEEP_LOG_SCHEDULE_GRANULARITY, "
		"SPINE_LOG_THREADS, SPINE_LOG_LOAD_VEC_SIZE, SPINE_LOG_LOADS_PER_TILE, SPINE_LOG_RAKING_THREADS, "
		"elapsed time (ms), throughput (10^9 items/s), bandwidth (10^9 B/s), Correctness\n");

	ReductionTuner tuner;

//	typedef unsigned char T;
//	typedef unsigned short T;
	typedef unsigned int T;
//	typedef unsigned long long T;

	// Execute test(s)
	tuner.TestReduction<T, Sum<T> >(num_elements * sizeof(num_elements) / 4);

	return 0;
}



