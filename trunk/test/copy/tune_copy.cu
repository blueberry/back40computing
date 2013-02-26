/******************************************************************************
 * Copyright (c) 2010-2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2013, NVIDIA CORPORATION.  All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/


/******************************************************************************
 * Tuning tool for establishing optimal copy granularity configuration types
 ******************************************************************************/

#include <stdio.h> 

// Copy includes
#include <b40c/copy/policy.cuh>
#include <b40c/copy/enactor.cuh>
#include <b40c/util/arch_dispatch.cuh>
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

bool g_verbose;
int g_max_ctas = 0;
int g_iterations = 0;
bool g_verify;
int g_policy_id = 0;


/******************************************************************************
 * Utility routines
 ******************************************************************************/

/**
 * Displays the commandline usage for this tool
 */
void Usage()
{
	printf("\ntune_copy [--device=<device index>] [--v] [--i=<num-iterations>] "
			"[--max-ctas=<max-thread-blocks>] [--n=<num-words>] [--verify]\n");
	printf("\n");
	printf("\t--v\tDisplays verbose configuration to the console.\n");
	printf("\n");
	printf("\t--verify\tChecks the result.\n");
	printf("\n");
	printf("\t--i\tPerforms the copy operation <num-iterations> times\n");
	printf("\t\t\ton the device. Default = 1\n");
	printf("\n");
	printf("\t--n\tThe number of 32-bit words to comprise the sample problem\n");
	printf("\n");
	printf("\t--max-ctas\tThe number of CTAs to launch\n");
	printf("\n");
}


/**
 * Enumerated tuning params
 */
enum TuningParam {

	PARAM_BEGIN,

		READ_MODIFIER,
		WRITE_MODIFIER,
		WORK_STEALING,

		LOG_THREADS,
		LOG_LOAD_VEC_SIZE,
		LOG_LOADS_PER_TILE,

	PARAM_END,
};



/**
 * Encapsulation structure for
 * 		- Wrapping problem type and storage
 * 		- Providing call-back for parameter-list generation
 */
template <typename T, typename SizeT>
class TuneEnactor : public copy::Enactor
{
public:

	T *d_dest;
	T *d_src;
	T *h_data;
	T *h_reference;
	SizeT num_elements;

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

	// WORK_STEALING
	template <typename ParamList>
	struct Ranges<ParamList, WORK_STEALING> {
		enum {
			MIN = 0,
			MAX = 1
		};
	};

	// LOG_THREADS
	template <typename ParamList>
	struct Ranges<ParamList, LOG_THREADS> {
		enum {
			MIN = 5,		// 32
			MAX = 10		// 1024
		};
	};

	// LOG_LOAD_VEC_SIZE
	template <typename ParamList>
	struct Ranges<ParamList, LOG_LOAD_VEC_SIZE> {
		enum {
			MIN = 0,
			MAX = 2
		};
	};

	// LOG_LOADS_PER_TILE
	template <typename ParamList>
	struct Ranges<ParamList, LOG_LOADS_PER_TILE> {
		enum {
			MIN = 0,
			MAX = 2
		};
	};

	/**
	 * Constructor
	 */
	TuneEnactor(SizeT num_elements) :
		copy::Enactor(),
		d_dest(NULL),
		d_src(NULL),
		h_data(NULL),
		h_reference(NULL),
		num_elements(num_elements) {}


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
			if (enactor->template Copy<Policy>(
					enactor->d_dest,
					enactor->d_src,
					enactor->num_elements,
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
				if (enactor->template Copy<Policy>(
					enactor->d_dest,
					enactor->d_src,
					enactor->num_elements,
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
				avg_runtime, throughput, throughput * sizeof(T) * 2);
			fflush(stdout);

			if (g_verify) {
				// Copy out data
				if (util::B40CPerror(cudaMemcpy(
						enactor->h_data,
						enactor->d_dest, sizeof(T) *
						enactor->num_elements, cudaMemcpyDeviceToHost),
					"TimedCopy cudaMemcpy d_dest failed: ", __FILE__, __LINE__)) exit(1);

				// Verify solution
				CompareResults<T>(
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
		const int C_READ_MODIFIER =
			util::Access<ParamList, READ_MODIFIER>::VALUE;
		const int C_WRITE_MODIFIER =
			util::Access<ParamList, WRITE_MODIFIER>::VALUE;
		const int C_WORK_STEALING =
			util::Access<ParamList, WORK_STEALING>::VALUE;
		const int C_OVERSUBSCRIBED_GRID_SIZE =
			(C_WORK_STEALING) ? 0 : 1;	// Over-subscribe if we're not work-stealing
		const int C_LOG_THREADS =
			util::Access<ParamList, LOG_THREADS>::VALUE;
		const int C_LOG_LOAD_VEC_SIZE =
			util::Access<ParamList, LOG_LOAD_VEC_SIZE>::VALUE;
		const int C_LOG_LOADS_PER_TILE =
			util::Access<ParamList, LOG_LOADS_PER_TILE>::VALUE;
		const int C_MIN_CTA_OCCUPANCY =
			1;
		const int C_LOG_SCHEDULE_GRANULARITY =
			C_LOG_LOADS_PER_TILE +
			C_LOG_LOAD_VEC_SIZE +
			C_LOG_THREADS;

		// Establish the granularity configuration type
		typedef copy::Policy <
			T,
			SizeT,
			TUNE_ARCH,

			C_LOG_SCHEDULE_GRANULARITY,
			C_MIN_CTA_OCCUPANCY,
			C_LOG_THREADS,
			C_LOG_LOAD_VEC_SIZE,
			C_LOG_LOADS_PER_TILE,
			(util::io::ld::CacheModifier) C_READ_MODIFIER,
			(util::io::st::CacheModifier) C_WRITE_MODIFIER,
			C_WORK_STEALING,
			C_OVERSUBSCRIBED_GRID_SIZE> Policy;

		// Check if this configuration is worth compiling
		const int REG_MULTIPLIER = (sizeof(T) + 4 - 1) / 4;
		const int TILE_ELEMENTS_PER_THREAD = 1 << (C_LOG_THREADS + C_LOG_LOAD_VEC_SIZE + C_LOG_LOADS_PER_TILE);
		const int REGS_ESTIMATE = (REG_MULTIPLIER * TILE_ELEMENTS_PER_THREAD) + 2;
		const int EST_REGS_OCCUPANCY = B40C_SM_REGISTERS(TUNE_ARCH) / REGS_ESTIMATE;

		const int VALID =
			(((TUNE_ARCH >= 200) || (C_READ_MODIFIER == util::io::ld::NONE)) &&
			((TUNE_ARCH >= 200) || (C_WRITE_MODIFIER == util::io::st::NONE)) &&
			(C_LOG_THREADS <= B40C_LOG_CTA_THREADS(TUNE_ARCH)) &&
			(EST_REGS_OCCUPANCY > 0));

		// Invoke this config
		ApplyPolicy<Policy, VALID>::Invoke(this);
	}
};


/**
 * Creates an example scan problem and then dispatches the problem
 * to the GPU for the given number of iterations, displaying runtime information.
 */
template<typename T, typename SizeT>
void TestCopy(SizeT num_elements)
{
	// Allocate storage and enactor
	typedef TuneEnactor<T, SizeT> Detail;
	Detail detail(num_elements);

	if (util::B40CPerror(cudaMalloc((void**) &detail.d_src, sizeof(T) * num_elements),
		"TimedCopy cudaMalloc d_src failed: ", __FILE__, __LINE__)) exit(1);

	if (util::B40CPerror(cudaMalloc((void**) &detail.d_dest, sizeof(T) * num_elements),
		"TimedCopy cudaMalloc d_dest failed: ", __FILE__, __LINE__)) exit(1);

	if ((detail.h_data = (T*) malloc(sizeof(T) * num_elements)) == NULL) {
		fprintf(stderr, "Host malloc of problem data failed\n");
		exit(1);
	}
	if ((detail.h_reference = (T*) malloc(sizeof(T) * num_elements)) == NULL) {
		fprintf(stderr, "Host malloc of problem data failed\n");
		exit(1);
	}

	for (SizeT i = 0; i < num_elements; ++i) {
		// util::RandomBits<T>(detail.h_data[i], 0);
		detail.h_data[i] = i;
		detail.h_reference[i] = detail.h_data[i];
	}


	// Move a fresh copy of the problem into device storage
	if (util::B40CPerror(cudaMemcpy(detail.d_src, detail.h_data, sizeof(T) * num_elements, cudaMemcpyHostToDevice),
		"TimedCopy cudaMemcpy d_src failed: ", __FILE__, __LINE__)) exit(1);

	// Run the timing tests
	util::ParamListSweep<
		PARAM_BEGIN + 1,
		PARAM_END,
		Detail::template Ranges>::template Invoke<util::EmptyTuple>(detail);

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
	// Initialize commandline args and device
	CommandLineArgs args(argc, argv);
	DeviceInit(args);

	// Seed random number generator
	srand(0);				// presently deterministic

	typedef int SizeT;
	SizeT num_elements = 1024;

	// Check command line arguments
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

	printf("Test Copy: %d iterations, %lu 32-bit words (%lu bytes)", g_iterations, (unsigned long) num_elements, (unsigned long) num_elements * 4);
	printf("\nCodeGen: \t[device_sm_version: %d, kernel_ptx_version: %d]\n\n",
		cuda_props.device_sm_version, cuda_props.kernel_ptx_version);

	printf(""
		"sizeof(T), "
		"sizeof(SizeT), "
		"CUDA_ARCH, "
		"LOG_SCHEDULE_GRANULARITY, "
		"MIN_CTA_OCCUPANCY, "
		"LOG_THREADS, "
		"LOG_LOAD_VEC_SIZE, "
		"LOG_LOADS_PER_TILE, "
		"READ_MODIFIER, "
		"WRITE_MODIFIER, "
		"WORK_STEALING, "
		"OVERSUBSCRIBED_GRID_SIZE, "
		"elapsed time (ms), "
		"throughput (10^9 items/s), "
		"bandwidth (10^9 B/s)");
	if (g_verify) printf(", Correctness");
	printf("\n");

	// Execute test(s)
#if (TUNE_SIZE == 0) || (TUNE_SIZE == 1)
	{
		typedef unsigned char T;
		TestCopy<T>(num_elements * 4);
	}
#endif
#if (TUNE_SIZE == 0) || (TUNE_SIZE == 2)
	{
		typedef unsigned short T;
		TestCopy<T>(num_elements * 2);
	}
#endif
#if (TUNE_SIZE == 0) || (TUNE_SIZE == 4)
	{
		typedef unsigned int T;
		TestCopy<T>(num_elements);
	}
#endif
#if (TUNE_SIZE == 0) || (TUNE_SIZE == 8)
	{
		typedef unsigned long long T;
		TestCopy<T>(num_elements / 2);
	}
#endif

	return 0;
}

