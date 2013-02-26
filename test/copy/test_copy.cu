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
 * Simple test driver program for copy.
 ******************************************************************************/

#include <stdio.h> 

// Test utils
#include "b40c_test_util.h"
#include "test_copy.h"

using namespace b40c;

/******************************************************************************
 * Defines, constants, globals
 ******************************************************************************/

bool 	g_verbose;
bool 	g_sample;
bool 	g_from_host;
int 	g_max_ctas 						= 0;
int 	g_iterations  					= 1;
int 	g_num_elements 					= 1024;
int 	g_src_gpu						= -1;
int 	g_dest_gpu						= -1;


/******************************************************************************
 * Utility Routines
 ******************************************************************************/

/**
 * Displays the commandline usage for this tool
 */
void Usage()
{
	printf("\ntest_copy "
		"[--device=<device index>] "
		"[--v] "
		"[--i=<num-iterations>] "
		"[--max-ctas=<max-thread-blocks>] "
		"[--n=<num-bytes>] "
		"[--sample] "
		"[ [--src=<src-gpu> --dest=<dest-gpu>] | --from-host ]\n");
	printf("\n");
	printf("\t--v\tDisplays copied results to the console.\n");
	printf("\n");
	printf("\t--i\tPerforms the copy operation <num-iterations> times\n");
	printf("\t\t\ton the device. Re-copies original input each time. Default = %d\n", g_iterations);
	printf("\n");
	printf("\t--n\tThe number of bytes to comprise the sample problem\n");
	printf("\t\t\tDefault = %lu\n", (unsigned long) g_num_elements);
	printf("\n");
}


/**
 * Creates an example copy problem and then dispatches the problem
 * to the GPU for the given number of iterations, displaying runtime information.
 */
template <typename SizeT>
void TestCopy(SizeT num_elements)
{
	typedef unsigned char T;

	//
	// Allocate the copy problem on the host and fill the keys with random bytes
	//

	T *h_data = (T*) malloc(num_elements * sizeof(T));
	if (!h_data) {
		fprintf(stderr, "Host malloc of problem data failed\n");
		exit(1);
	}

	for (SizeT i = 0; i < num_elements; ++i) {
		// util::RandomBits<T>(h_data[i], 0);
		h_data[i] = i;
	}

	// Allocate device storage (and leave g_dest_gpu as current gpu)
	T *h_src = NULL;
	T *d_src = NULL;
	T *d_dest = NULL;

	bool same_device = (!g_from_host) && (g_src_gpu == g_dest_gpu);

	if (g_from_host) {
		int flags = cudaHostAllocMapped;
		if (util::B40CPerror(cudaHostAlloc((void**) &h_src, sizeof(T) * num_elements, flags),
			"TimedCopy cudaHostAlloc d_src failed", __FILE__, __LINE__)) exit(1);

		// Map into GPU space
		if (util::B40CPerror(cudaHostGetDevicePointer((void **)&d_src, (void *) h_src, 0),
			"TimedCopy cudaHostGetDevicePointer h_src failed", __FILE__, __LINE__)) exit(1);

	} else {
		if (util::B40CPerror(cudaSetDevice(g_src_gpu),
			"MultiGpuBfsEnactor cudaSetDevice failed", __FILE__, __LINE__)) exit(1);
		if (util::B40CPerror(cudaMalloc((void**) &d_src, sizeof(T) * num_elements),
			"TimedCopy cudaMalloc d_src failed: ", __FILE__, __LINE__)) exit(1);
	}

	if (util::B40CPerror(cudaSetDevice(g_dest_gpu),
		"MultiGpuBfsEnactor cudaSetDevice failed", __FILE__, __LINE__)) exit(1);
	if (util::B40CPerror(cudaMalloc((void**) &d_dest, sizeof(T) * num_elements),
		"TimedCopy cudaMalloc d_dest failed: ", __FILE__, __LINE__)) exit(1);

	// Move a fresh copy of the problem into device storage
	if (util::B40CPerror(cudaMemcpy(d_src, h_data, sizeof(T) * num_elements, cudaMemcpyHostToDevice),
		"TimedCopy cudaMemcpy d_src failed: ", __FILE__, __LINE__)) exit(1);

	//
    // Run the timing test(s)
	//

	b40c::GpuTimer gpu_timer;
	double max_exponent 		= log2(double(num_elements)) - 5.0;
	unsigned int max_int 		= (unsigned int) -1;

	if (g_sample) {
		// Sample problem sizes up to num_elements
		printf("ITERATION, ELEMENTS, BYTES, SAMPLES, AVG_MILLIS, BANDWIDTH, STATUS\n");

		for (int i = 0; i < g_iterations; i++) {

			// Sample a problem size
			unsigned int sample;
			b40c::util::RandomBits(sample);
			double scale = double(sample) / max_int;
			SizeT elements = (i < g_iterations / 2) ?
				(SizeT) pow(2.0, (max_exponent * scale) + 5.0) :		// log bias
				elements = scale * num_elements;						// uniform bias

			printf("%d, ", i);

			// One iteration at that problem size
			TimedCopy<copy::UNKNOWN_SIZE>(
				h_data,
				d_src,
				d_dest,
				elements,
				g_max_ctas,
				g_verbose,
				1,
				same_device,
				false);
		}
	} else {
		// Test large and small configs on num_elements
		printf("ELEMENTS, BYTES, SAMPLES, AVG_MILLIS, BANDWIDTH, STATUS\n\n");

		printf("Large-problem configuration:\n");
		TimedCopy<copy::LARGE_SIZE>(
			h_data,
			d_src,
			d_dest,
			num_elements,
			g_max_ctas,
			g_verbose,
			g_iterations,
			same_device);

		printf("\n");

		printf("Small-problem configuration:\n");
		TimedCopy<copy::SMALL_SIZE>(
			h_data,
			d_src,
			d_dest,
			num_elements,
			g_max_ctas,
			g_verbose,
			g_iterations,
			same_device);
	}

    // Free allocated memory
	if (h_data) free(h_data);
    if (h_src) {
		cudaFreeHost(h_src);
	} else {
		cudaFree(d_src);
	}
    if (d_dest) cudaFree(d_dest);
}


/******************************************************************************
 * Main
 ******************************************************************************/

int main(int argc, char** argv)
{

	CommandLineArgs args(argc, argv);
	DeviceInit(args);
	cudaSetDeviceFlags(cudaDeviceMapHost);

	//srand(time(NULL));
	srand(0);				// presently deterministic

	// Check command line arguments
    if (args.CheckCmdLineFlag("help")) {
		Usage();
		return 0;
	}
    args.GetCmdLineArgument("i", g_iterations);
    args.GetCmdLineArgument("n", g_num_elements);
    args.GetCmdLineArgument("src", g_src_gpu);
    args.GetCmdLineArgument("dest", g_dest_gpu);
    args.GetCmdLineArgument("max-ctas", g_max_ctas);
    g_from_host = args.CheckCmdLineFlag("from-host");
    g_sample = args.CheckCmdLineFlag("sample");
	g_verbose = args.CheckCmdLineFlag("v");

	if ((g_src_gpu > -1) && (g_dest_gpu > -1)) {

		printf("Inter-GPU copy.\n");

		// Set device
		if (util::B40CPerror(cudaSetDevice(g_src_gpu),
			"MultiGpuBfsEnactor cudaSetDevice failed", __FILE__, __LINE__)) exit(1);
		printf("Enabling peer access to GPU %d from GPU %d\n", g_src_gpu, g_dest_gpu);
		if (util::B40CPerror(cudaDeviceEnablePeerAccess(g_dest_gpu, 0),
			"MultiGpuBfsEnactor cudaDeviceEnablePeerAccess failed", __FILE__, __LINE__)) exit(1);

		// Set device
		if (util::B40CPerror(cudaSetDevice(g_dest_gpu),
			"MultiGpuBfsEnactor cudaSetDevice failed", __FILE__, __LINE__)) exit(1);
		printf("Enabling peer access to GPU %d from GPU %d\n", g_dest_gpu, g_src_gpu);
		if (util::B40CPerror(cudaDeviceEnablePeerAccess(g_src_gpu, 0),
			"MultiGpuBfsEnactor cudaDeviceEnablePeerAccess failed", __FILE__, __LINE__)) exit(1);

	} else {

		if (g_from_host) {
			printf("From pinned host memory.\n");
		}

		// Put current device as both src and dest
		if (util::B40CPerror(cudaGetDevice(&g_src_gpu),
			"MultiGpuBfsEnactor cudaGetDevice failed", __FILE__, __LINE__)) exit(1);
		if (util::B40CPerror(cudaGetDevice(&g_dest_gpu),
			"MultiGpuBfsEnactor cudaGetDevice failed", __FILE__, __LINE__)) exit(1);
	}
   	TestCopy(g_num_elements);

	return 0;
}



