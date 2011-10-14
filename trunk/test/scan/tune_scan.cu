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
 * Tuning tool for establishing optimal scan granularity configuration types
 ******************************************************************************/

#include <stdio.h> 

// Scan includes
#include <b40c/util/arch_dispatch.cuh>
#include <b40c/scan/problem_type.cuh>
#include <b40c/scan/policy.cuh>
#include <b40c/scan/enactor.cuh>
#include <b40c/util/cuda_properties.cuh>
#include <b40c/util/numeric_traits.cuh>
#include <b40c/util/parameter_generation.cuh>

// Test utils
#include "b40c_test_util.h"

using namespace b40c;


/******************************************************************************
 * Defines, constants, globals, and utility types
 ******************************************************************************/

#ifndef TUNE_ARCH
	#define TUNE_ARCH (200)
#endif
#ifndef TUNE_SIZE
	#define TUNE_SIZE (4)
#endif

bool 	g_verbose;
int 	g_max_ctas = 0;
int 	g_iterations = 0;
bool 	g_verify;
int 	g_policy_id = 0;;


/******************************************************************************
 * Test wrappers for binary, associative operations
 ******************************************************************************/

template <typename T>
struct Sum
{
	// Binary reduction
	__host__ __device__ __forceinline__ T operator()(const T &a, const T &b)
	{
		return a + b;
	}

	// Identity
	__host__ __device__ __forceinline__ T operator()()
	{
		return 0;
	}

	static const bool IS_COMMUTATIVE = true;
};

template <typename T>
struct Max
{
	// Binary reduction
	__host__ __device__ __forceinline__ T Op(const T &a, const T &b)
	{
		return (a > b) ? a : b;
	}

	// Identity
	__host__ __device__ __forceinline__ T operator()()
	{
		return 0;
	}

	static const bool IS_COMMUTATIVE = true;
};



/******************************************************************************
 * Utility routines
 ******************************************************************************/

/**
 * Displays the commandline usage for this tool
 */
void Usage()
{
	printf("\ntune_scan [--device=<device index>] [--v] [--i=<num-iterations>] "
			"[--max-ctas=<max-thread-blocks>] [--n=<num-words>] [--verify]\n");
	printf("\n");
	printf("\t--v\tDisplays verbose configuration to the console.\n");
	printf("\n");
	printf("\t--verify\tChecks the result.\n");
	printf("\n");
	printf("\t--i\tPerforms the scan operation <num-iterations> times\n");
	printf("\t\t\ton the device. Default = 1\n");
	printf("\n");
	printf("\t--n\tThe number of 32-bit words to comprise the sample problem\n");
	printf("\n");
	printf("\t--max-ctas\tThe number of CTAs to launch\n");
	printf("\n");
}

/******************************************************************************
 * Tuning Parameter Enumerations and Ranges
 ******************************************************************************/

/**
 * Enumerated tuning params
 */
enum TuningParam {

	PARAM_BEGIN,

		UPSWEEP_LOG_THREADS,
		UPSWEEP_LOG_LOAD_VEC_SIZE,
		UPSWEEP_LOG_LOADS_PER_TILE,

		DOWNSWEEP_LOG_THREADS,
		DOWNSWEEP_LOG_LOAD_VEC_SIZE,
		DOWNSWEEP_LOG_LOADS_PER_TILE,
		DOWNSWEEP_LOG_RAKING_THREADS,

	PARAM_END,

	// Parameters below here are currently not part of the tuning sweep
	READ_MODIFIER,
	WRITE_MODIFIER,
	UNIFORM_SMEM_ALLOCATION,
	UNIFORM_GRID_SIZE,
	LOG_SCHEDULE_GRANULARITY,
};


/**
 * Ranges for the tuning params
 */
template <typename ParamList, int PARAM> struct Ranges;

// READ_MODIFIER
template <typename ParamList>
struct Ranges<ParamList, READ_MODIFIER> {
	enum {
		MIN = util::io::ld::NONE,
		MAX = util::io::ld::LIMIT - 1,
	};
};

// WRITE_MODIFIER
template <typename ParamList>
struct Ranges<ParamList, WRITE_MODIFIER> {
	enum {
		MIN = util::io::st::NONE,
		MAX = util::io::st::LIMIT - 1,
	};
};

// UNIFORM_SMEM_ALLOCATION
template <typename ParamList>
struct Ranges<ParamList, UNIFORM_SMEM_ALLOCATION> {
	enum {
		MIN = 0,
		MAX = 1
	};
};

// UNIFORM_GRID_SIZE
template <typename ParamList>
struct Ranges<ParamList, UNIFORM_GRID_SIZE> {
	enum {
		MIN = 0,
		MAX = 1
	};
};

// UPSWEEP_LOG_THREADS
template <typename ParamList>
struct Ranges<ParamList, UPSWEEP_LOG_THREADS> {
	enum {
		MIN = 5,		// 32
		MAX = 10		// 1024
	};
};

// UPSWEEP_LOG_LOAD_VEC_SIZE
template <typename ParamList>
struct Ranges<ParamList, UPSWEEP_LOG_LOAD_VEC_SIZE> {
	enum {
		MIN = 0,
		MAX = 2
	};
};

// UPSWEEP_LOG_LOADS_PER_TILE
template <typename ParamList>
struct Ranges<ParamList, UPSWEEP_LOG_LOADS_PER_TILE> {
	enum {
		MIN = 0,
		MAX = 2
	};
};

// DOWNSWEEP_LOG_THREADS
template <typename ParamList>
struct Ranges<ParamList, DOWNSWEEP_LOG_THREADS> {
	enum {
		MIN = 5,		// 32
		MAX = 10		// 1024
	};
};

// DOWNSWEEP_LOG_LOAD_VEC_SIZE
template <typename ParamList>
struct Ranges<ParamList, DOWNSWEEP_LOG_LOAD_VEC_SIZE> {
	enum {
		MIN = 0,
		MAX = 2
	};
};

// DOWNSWEEP_LOG_LOADS_PER_TILE
template <typename ParamList>
struct Ranges<ParamList, DOWNSWEEP_LOG_LOADS_PER_TILE> {
	enum {
		MIN = 0,
		MAX = 2
	};
};

// DOWNSWEEP_LOG_RAKING_THREADS
template <typename ParamList>
struct Ranges<ParamList, DOWNSWEEP_LOG_RAKING_THREADS> {
	enum {
		MIN = B40C_LOG_WARP_THREADS(TUNE_ARCH),
		MAX = util::Access<ParamList, DOWNSWEEP_LOG_THREADS>::VALUE
	};
};


/******************************************************************************
 * Derived tuning enactor
 ******************************************************************************/

/**
 * Encapsulation structure for
 * 		- Wrapping problem type and storage
 * 		- Providing call-back for parameter-list generation
 */
template <typename T, typename SizeT, typename OpType>
class TuneEnactor : public scan::Enactor
{
public:

	T *d_dest;
	T *d_src;
	T *h_data;
	T *h_reference;
	SizeT num_elements;
	OpType binary_op;

	/**
	 * Constructor
	 */
	TuneEnactor(SizeT num_elements, OpType binary_op) :
		scan::Enactor(),
		d_dest(NULL),
		d_src(NULL),
		h_data(NULL),
		h_reference(NULL),
		binary_op(binary_op) {}


	/**
	 * Applies a specific granularity configuration type
	 */
	template <typename Policy, int VALID>
	struct ApplyPolicy
	{
		template <typename Enactor>
		static void Invoke(Enactor *enactor)
		{
			printf("%d, ", g_policy_id);
			g_policy_id++;

			Policy::Print();
			fflush(stdout);

			// Perform a single iteration to allocate any memory if needed, prime code caches, etc.
			enactor->ENACTOR_DEBUG = g_verbose;
			if (enactor->template Scan<Policy>(
					enactor->d_dest,
					enactor->d_src,
					enactor->num_elements,
					enactor->binary_op,
					enactor->binary_op,
					g_max_ctas))
			{
				exit(1);
			}
			enactor->ENACTOR_DEBUG = false;

			// Perform the timed number of iterations
			GpuTimer timer;
			double elapsed = 0;
			for (int i = 0; i < g_iterations; i++) {

				// Start cuda timing record
				timer.Start();

				// Call the scan API routine
				if (enactor->template Scan<Policy>(
						enactor->d_dest,
						enactor->d_src,
						enactor->num_elements,
						enactor->binary_op,
						enactor->binary_op,
						g_max_ctas))
				{
					exit(1);
				}

				// End cuda timing record
				timer.Stop();
				elapsed += timer.ElapsedMillis();

				// Flushes any stdio from the GPU
				if (util::B40CPerror(cudaThreadSynchronize(), "TimedCopy cudaThreadSynchronize failed: ", __FILE__, __LINE__)) {
					exit(1);
				}
			}

			// Display timing information
			double avg_runtime = elapsed / g_iterations;
			double throughput =  0.0;
			if (avg_runtime > 0.0) throughput = ((double) enactor->num_elements) / avg_runtime / 1000.0 / 1000.0;
			printf(", %f, %f, %f, ",
				avg_runtime, throughput, throughput * sizeof(T) * 3);
			fflush(stdout);

			if (g_verify) {

				// Copy out data
				if (util::B40CPerror(cudaMemcpy(
					enactor->h_data,
					enactor->d_dest,
					sizeof(T) * enactor->num_elements,
					cudaMemcpyDeviceToHost),
						"TimedScan cudaMemcpy d_dest failed: ", __FILE__, __LINE__)) exit(1);

				// Verify solution
				CompareResults(
					enactor->h_data,
					enactor->h_reference,
					enactor->num_elements,
					true);
			}

			printf("\n");
			fflush(stdout);
		}

	};

	template <typename Policy>
	struct ApplyPolicy<Policy, 0>
	{
		template <typename Enactor>
		static void Invoke(Enactor *enactor)
		{
			printf("%d, ", g_policy_id);
			g_policy_id++;
			Policy::Print();
			printf("\n");
			fflush(stdout);
		}
	};

	/**
	 * Callback invoked by parameter-list generation
	 */
	template <typename ParamList>
	void Invoke()
	{
		const bool EXCLUSIVE = true;
		const bool COMMUTATIVE = true;

		// Tuned params
		const int C_UPSWEEP_LOG_THREADS =
			util::Access<ParamList, UPSWEEP_LOG_THREADS>::VALUE;
		const int C_UPSWEEP_LOG_LOAD_VEC_SIZE =
			util::Access<ParamList, UPSWEEP_LOG_LOAD_VEC_SIZE>::VALUE;
		const int C_UPSWEEP_LOG_LOADS_PER_TILE =
			util::Access<ParamList, UPSWEEP_LOG_LOADS_PER_TILE>::VALUE;
		const int C_UPSWEEP_LOG_RAKING_THREADS =
			B40C_LOG_WARP_THREADS(TUNE_ARCH);				// Revisit if we tune non-commutative scan
		const int C_UPSWEEP_MIN_CTA_OCCUPANCY =
			1;
		const int C_UPSWEEP_LOG_SCHEDULE_GRANULARITY =
			C_UPSWEEP_LOG_LOADS_PER_TILE +
			C_UPSWEEP_LOG_LOAD_VEC_SIZE +
			C_UPSWEEP_LOG_THREADS;

		const int C_DOWNSWEEP_LOG_THREADS =
			util::Access<ParamList, DOWNSWEEP_LOG_THREADS>::VALUE;
		const int C_DOWNSWEEP_LOG_LOAD_VEC_SIZE =
			util::Access<ParamList, DOWNSWEEP_LOG_LOAD_VEC_SIZE>::VALUE;
		const int C_DOWNSWEEP_LOG_LOADS_PER_TILE =
			util::Access<ParamList, DOWNSWEEP_LOG_LOADS_PER_TILE>::VALUE;
		const int C_DOWNSWEEP_LOG_RAKING_THREADS =
			util::Access<ParamList, DOWNSWEEP_LOG_RAKING_THREADS>::VALUE;		
		const int C_DOWNSWEEP_MIN_CTA_OCCUPANCY =
			1;
		const int C_DOWNSWEEP_LOG_SCHEDULE_GRANULARITY =
			C_DOWNSWEEP_LOG_LOADS_PER_TILE +
			C_DOWNSWEEP_LOG_LOAD_VEC_SIZE +
			C_DOWNSWEEP_LOG_THREADS;

		// TODO: figure out if we should use min here instead
		const int C_LOG_SCHEDULE_GRANULARITY = B40C_MAX(
			C_UPSWEEP_LOG_SCHEDULE_GRANULARITY,
			C_DOWNSWEEP_LOG_SCHEDULE_GRANULARITY);

		// Non-tuned params

		// General performance is insensitive to spine config it's only a single-CTA:
		// simply use reasonable defaults
		const int C_SPINE_LOG_THREADS = 8;
		const int C_SPINE_LOG_LOAD_VEC_SIZE = 1;
		const int C_SPINE_LOG_LOADS_PER_TILE = 0;
		const int C_SPINE_LOG_RAKING_THREADS = B40C_LOG_WARP_THREADS(TUNE_ARCH);

		const int C_READ_MODIFIER =
//			util::Access<ParamList, READ_MODIFIER>::VALUE;					// These can be tuned, but we're currently not compelled to
			util::io::ld::NONE;
		const int C_WRITE_MODIFIER =
//			util::Access<ParamList, WRITE_MODIFIER>::VALUE;					// These can be tuned, but we're currently not compelled to
			util::io::ld::NONE;
		const int C_UNIFORM_SMEM_ALLOCATION =
//			util::Access<ParamList, UNIFORM_SMEM_ALLOCATION>::VALUE;
			0;
		const int C_UNIFORM_GRID_SIZE =
//			util::Access<ParamList, UNIFORM_GRID_SIZE>::VALUE;
			0;
		const int C_OVERSUBSCRIBED_GRID_SIZE =
			1;

		// Establish the problem type
		typedef scan::ProblemType<
			T,
			SizeT,
			OpType,
			OpType,
			EXCLUSIVE,
			COMMUTATIVE> ProblemType;

		// Establish the granularity configuration type
		typedef scan::Policy <
			ProblemType,
			TUNE_ARCH,

			(util::io::ld::CacheModifier) C_READ_MODIFIER,
			(util::io::st::CacheModifier) C_WRITE_MODIFIER,
			C_UNIFORM_SMEM_ALLOCATION,
			C_UNIFORM_GRID_SIZE,
			C_OVERSUBSCRIBED_GRID_SIZE,
			C_LOG_SCHEDULE_GRANULARITY,

			C_UPSWEEP_MIN_CTA_OCCUPANCY,
			C_UPSWEEP_LOG_THREADS,
			C_UPSWEEP_LOG_LOAD_VEC_SIZE,
			C_UPSWEEP_LOG_LOADS_PER_TILE,
			C_UPSWEEP_LOG_RAKING_THREADS,

			C_SPINE_LOG_THREADS, 
			C_SPINE_LOG_LOAD_VEC_SIZE, 
			C_SPINE_LOG_LOADS_PER_TILE, 
			C_SPINE_LOG_RAKING_THREADS,

			C_DOWNSWEEP_MIN_CTA_OCCUPANCY,
			C_DOWNSWEEP_LOG_THREADS,
			C_DOWNSWEEP_LOG_LOAD_VEC_SIZE,
			C_DOWNSWEEP_LOG_LOADS_PER_TILE,
			C_DOWNSWEEP_LOG_RAKING_THREADS> Policy;

		// Check if this configuration is worth compiling
		const int REG_MULTIPLIER = (sizeof(T) + 4 - 1) / 4;

		const int UPSWEEP_TILE_ELEMENTS_PER_THREAD = 1 << (C_UPSWEEP_LOG_THREADS + C_UPSWEEP_LOG_LOAD_VEC_SIZE + C_UPSWEEP_LOG_LOADS_PER_TILE);
		const int UPSWEEP_REGS_ESTIMATE = (REG_MULTIPLIER * UPSWEEP_TILE_ELEMENTS_PER_THREAD) + 2;
		const int UPSWEEP_EST_REGS_OCCUPANCY = B40C_SM_REGISTERS(TUNE_ARCH) / UPSWEEP_REGS_ESTIMATE;

		const int SPINE_TILE_ELEMENTS_PER_THREAD = 1 << (C_SPINE_LOG_THREADS + C_SPINE_LOG_LOAD_VEC_SIZE + C_SPINE_LOG_LOADS_PER_TILE);
		const int SPINE_REGS_ESTIMATE = (REG_MULTIPLIER * SPINE_TILE_ELEMENTS_PER_THREAD) + 2;
		const int SPINE_EST_REGS_OCCUPANCY = B40C_SM_REGISTERS(TUNE_ARCH) / SPINE_REGS_ESTIMATE;

		const int DOWNSWEEP_TILE_ELEMENTS_PER_THREAD = 1 << (C_DOWNSWEEP_LOG_THREADS + C_DOWNSWEEP_LOG_LOAD_VEC_SIZE + C_DOWNSWEEP_LOG_LOADS_PER_TILE);
		const int DOWNSWEEP_REGS_ESTIMATE = (REG_MULTIPLIER * DOWNSWEEP_TILE_ELEMENTS_PER_THREAD) + 2;
		const int DOWNSWEEP_EST_REGS_OCCUPANCY = B40C_SM_REGISTERS(TUNE_ARCH) / DOWNSWEEP_REGS_ESTIMATE;

		const int VALID =
			(((TUNE_ARCH >= 200) || (C_READ_MODIFIER == util::io::ld::NONE)) &&
			((TUNE_ARCH >= 200) || (C_WRITE_MODIFIER == util::io::st::NONE)) &&
			(C_UPSWEEP_LOG_THREADS <= B40C_LOG_CTA_THREADS(TUNE_ARCH)) &&
			(C_SPINE_LOG_THREADS <= B40C_LOG_CTA_THREADS(TUNE_ARCH)) &&
			(C_DOWNSWEEP_LOG_THREADS <= B40C_LOG_CTA_THREADS(TUNE_ARCH)) &&
			(UPSWEEP_EST_REGS_OCCUPANCY > 0) &&
			(SPINE_EST_REGS_OCCUPANCY > 0) &&
			(DOWNSWEEP_EST_REGS_OCCUPANCY > 0));

		// Invoke this config
		ApplyPolicy<Policy, VALID>::Invoke(this);
	}
};


/**
 * Creates an example scan problem and then dispatches the problem
 * to the GPU for the given number of iterations, displaying runtime information.
 */
template<typename T, typename SizeT, typename OpType>
void TestScan(
	SizeT num_elements,
	OpType binary_op)
{
	// Allocate storage and enactor
	typedef TuneEnactor<T, SizeT, OpType> Detail;
	Detail detail(num_elements, binary_op);

	if (util::B40CPerror(cudaMalloc((void**) &detail.d_src, sizeof(T) * num_elements),
		"TimedScan cudaMalloc d_src failed: ", __FILE__, __LINE__)) exit(1);

	if (util::B40CPerror(cudaMalloc((void**) &detail.d_dest, sizeof(T) * num_elements),
		"TimedScan cudaMalloc d_dest failed: ", __FILE__, __LINE__)) exit(1);

	if ((detail.h_data = (T*) malloc(sizeof(T) * num_elements)) == NULL) {
		fprintf(stderr, "Host malloc of problem data failed\n");
		exit(1);
	}
	if ((detail.h_reference = (T*) malloc(sizeof(T) * num_elements)) == NULL) {
		fprintf(stderr, "Host malloc of problem data failed\n");
		exit(1);
	}

	detail.h_reference[0] = binary_op();

	for (size_t i = 0; i < num_elements; ++i) {
//		util::RandomBits<T>(h_data[i], 0);
		detail.h_data[i] = i;

		detail.h_reference[i] = (i == 0) ?
			binary_op() :
			binary_op(detail.h_reference[i - 1], detail.h_data[i - 1]);
	}

	// Move a fresh copy of the problem into device storage
	if (util::B40CPerror(cudaMemcpy(detail.d_src, detail.h_data, sizeof(T) * num_elements, cudaMemcpyHostToDevice),
		"TimedScan cudaMemcpy d_src failed: ", __FILE__, __LINE__)) exit(1);

	// Run the timing tests
	util::ParamListSweep<
		Detail,
		PARAM_BEGIN + 1,
		PARAM_END,
		Ranges>::template Invoke<util::EmptyTuple>(detail);

	// Free allocated memory
	if (detail.d_src) cudaFree(detail.d_src);
	if (detail.d_dest) cudaFree(detail.d_dest);

	// Free our allocated host memory
	if (detail.h_data) free(detail.h_data);
	if (detail.h_reference) free(detail.h_reference);
}


/******************************************************************************
 * Main
 ******************************************************************************/

int main(int argc, char** argv)
{

	CommandLineArgs args(argc, argv);
	DeviceInit(args);

	// Seed random number generator
	srand(0);				// presently deterministic

	// Use 32-bit integer for array indexing
	typedef int SizeT;
	SizeT num_elements = 1024;

	// Parse command line arguments
    if (args.CheckCmdLineFlag("help")) {
		Usage();
		return 0;
	}
    args.GetCmdLineArgument("i", g_iterations);
    args.GetCmdLineArgument("n", num_elements);
    args.GetCmdLineArgument("max-ctas", g_max_ctas);
    g_verify = args.CheckCmdLineFlag("verify");
	g_verbose = args.CheckCmdLineFlag("v");

	util::CudaProperties cuda_props;

	printf("Test Scan: %d iterations, %lu elements", g_iterations, (unsigned long) num_elements);
	printf("\nCodeGen: \t[device_sm_version: %d, kernel_ptx_version: %d]\n\n",
		cuda_props.device_sm_version, cuda_props.kernel_ptx_version);

	printf(""
		"sizeof(T), "
		"sizeof(SizeT), "
		"CUDA_ARCH, "

		"READ_MODIFIER, "
		"WRITE_MODIFIER, "
		"UNIFORM_SMEM_ALLOCATION, "
		"UNIFORM_GRID_SIZE, "
		"OVERSUBSCRIBED_GRID_SIZE, "
		"LOG_SCHEDULE_GRANULARITY, "

		"UPSWEEP_MIN_CTA_OCCUPANCY, "
		"UPSWEEP_LOG_THREADS, "
		"UPSWEEP_LOG_LOAD_VEC_SIZE, "
		"UPSWEEP_LOG_LOADS_PER_TILE, "
		"UPSWEEP_LOG_RAKING_THREADS, "

		"SPINE_LOG_THREADS, "
		"SPINE_LOG_LOAD_VEC_SIZE, "
		"SPINE_LOG_LOADS_PER_TILE, "
		"SPINE_LOG_RAKING_THREADS, "

		"DOWNSWEEP_MIN_CTA_OCCUPANCY, "
		"DOWNSWEEP_LOG_THREADS, "
		"DOWNSWEEP_LOG_LOAD_VEC_SIZE, "
		"DOWNSWEEP_LOG_LOADS_PER_TILE, "
		"DOWNSWEEP_LOG_RAKING_THREADS, "

		"elapsed time (ms), "
		"throughput (10^9 items/s), "
		"bandwidth (10^9 B/s)");
	if (g_verify) printf(", Correctness");
	printf("\n");


	// Execute test(s)
#if (TUNE_SIZE == 0) || (TUNE_SIZE == 1)
	{
		typedef unsigned char T;
		Sum<T> binary_op;
		TestScan<T>(num_elements * 4, binary_op);
	}
#endif
#if (TUNE_SIZE == 0) || (TUNE_SIZE == 2)
	{
		typedef unsigned short T;
		Sum<T> binary_op;
		TestScan<T>(num_elements * 2, binary_op);
	}
#endif
#if (TUNE_SIZE == 0) || (TUNE_SIZE == 4)
	{
		typedef unsigned int T;
		Sum<T> binary_op;
		TestScan<T>(num_elements, binary_op);
	}
#endif
#if (TUNE_SIZE == 0) || (TUNE_SIZE == 8)
	{
		typedef unsigned long long T;
		Sum<T> binary_op;
		TestScan<T>(num_elements / 2, binary_op);
	}
#endif

	return 0;
}



