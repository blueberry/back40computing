/******************************************************************************
 * 
 * Copyright 2010-2012 Duane Merrill
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



