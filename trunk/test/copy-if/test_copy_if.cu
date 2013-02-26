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
 * Simple test driver program for copy-if.
 ******************************************************************************/

#include <stdio.h> 

// Test utils
#include "b40c_test_util.h"
#include "test_copy_if.h"

#include <b40c/util/multiple_buffering.cuh>

using namespace b40c;

/******************************************************************************
 * Defines, constants, globals
 ******************************************************************************/

bool 	g_verbose 						= false;
bool 	g_sweep							= false;
int 	g_max_ctas 						= 0;
int 	g_iterations  					= 1;


/******************************************************************************
 * Utility Routines
 ******************************************************************************/

/**
 * Displays the commandline usage for this tool
 */
void Usage()
{
	printf("\ntest_copy_if [--device=<device index>] [--max-ctas=<max-thread-blocks>] "
			"[--v] [--i=<num-iterations>] "
			"[--keep<keep-fraction>]  "
			"[--n=<num-elements>]\n");
	printf("\n");
	printf("\t--v\tDisplays copied results to the console.\n");
	printf("\n");
	printf("\t--i\tPerforms the copy-if operation <num-iterations> times\n");
	printf("\t\t\ton the device. Re-copies original input each time. Default = 1\n");
	printf("\n");
	printf("\t--keep\tThe fraction of elements to keep as valid (default = 0.5)");
	printf("\n");
	printf("\t--n\tThe number of elements to comprise the sample problem\n");
	printf("\t\t\tDefault = 512\n");
	printf("\n");
}


/**
 * Tests an example copy-if problem
 */
template<
	typename DoubleBuffer,
	typename SizeT,
	typename EqualityOp>
void TestProblem(
	SizeT num_elements,
	SizeT num_compacted,
	DoubleBuffer &h_problem_storage,
	EqualityOp select_op)
{

	// Execute test(s), optionally sweeping problem size downward
	printf("\nLARGE config:\t");
	double large = TimedConsecutiveRemoval<copy_if::LARGE_SIZE>(
		h_problem_storage, num_elements, num_compacted, select_op, g_max_ctas, g_verbose, g_iterations);

}


/**
 * Creates and tests a keys-only example copy-if problem
 */
template <typename KeyType, typename SizeT>
void TestProblem(SizeT num_elements, float keep_percentage)
{
	util::DoubleBuffer<KeyType> h_problem_storage;

	// Allocate the copy-if problem on the host
	h_problem_storage.d_keys[0] = (KeyType*) malloc(num_elements * sizeof(KeyType));
	h_problem_storage.d_keys[1] = (KeyType*) malloc(num_elements * sizeof(KeyType));

	if (!h_problem_storage.d_keys[0] || !h_problem_storage.d_keys[1]) {
		fprintf(stderr, "Host malloc of problem data failed\n");
		exit(1);
	}

	// Initialize problem
	printf("Initializing input...");
	fflush(stdout);
	for (int i = 0; i < num_elements; i++) {
		util::RandomBits<KeyType>(h_problem_storage.d_keys[0][i]);
	}
	printf(" Done.\n");
	fflush(stdout);

	// Display problem
	if (g_verbose) {
		printf("Input problem: \n");
		for (int i = 0; i < num_elements; i++) {
			printf("%lld, ", (long long) h_problem_storage.d_keys[0][i]);
		}
		printf("\n");
	}

	// Initialize keep conditional
	printf("Keeping %%%f\n", keep_percentage);
	KeyType comparand = keep_percentage * KeyType(-1);
	Select<KeyType> select_op(comparand);
	fflush(stdout);

	// Compute reference solution
	SizeT num_compacted = 0;
	for (SizeT i = 0; i < num_elements; ++i) {
		if (select_op(h_problem_storage.d_keys[0][i])) {
			h_problem_storage.d_keys[1][num_compacted] = h_problem_storage.d_keys[0][i];
			num_compacted++;
		}
	}

	// Test problem
	TestProblem(num_elements, num_compacted, h_problem_storage, select_op);

	// Free our allocated host memory
	if (h_problem_storage.d_keys[0]) free(h_problem_storage.d_keys[0]);
	if (h_problem_storage.d_keys[1]) free(h_problem_storage.d_keys[1]);
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
	//srand(time(NULL));

	// Use 32-bit integer for array indexing
	typedef int SizeT;
	SizeT num_elements 		= 1024;	// default
	float keep_percentage 	= 0.50;	// default

	// Parse command line arguments
    if (args.CheckCmdLineFlag("help")) {
		Usage();
		return 0;
	}
    args.GetCmdLineArgument("keep", keep_percentage);
    args.GetCmdLineArgument("i", g_iterations);
    args.GetCmdLineArgument("n", num_elements);
    args.GetCmdLineArgument("max-ctas", g_max_ctas);
	g_verbose = args.CheckCmdLineFlag("v");
/*
	{
		printf("\n-- UNSIGNED CHAR ----------------------------------------------\n");
		typedef unsigned char T;
		TestProblem<T>(num_elements * 4, keep_percentage);
	}
	{
		printf("\n-- UNSIGNED SHORT ----------------------------------------------\n");
		typedef unsigned short T;
		TestProblem<T>(num_elements * 2, keep_percentage);
	}
*/
	{
		printf("\n-- UNSIGNED INT -----------------------------------------------\n");
		typedef unsigned int T;
		TestProblem<T>(num_elements, keep_percentage);
	}
/*
	{
		printf("\n-- UNSIGNED LONG LONG -----------------------------------------\n");
		typedef unsigned long long T;
		TestProblem<T>(num_elements / 2, keep_percentage);
	}
*/
	return 0;
}



