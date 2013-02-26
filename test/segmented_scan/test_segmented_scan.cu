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
 * Simple test driver program for segmented scan.
 ******************************************************************************/

#include <stdio.h> 

// Test utils
#include "b40c_test_util.h"
#include "test_segmented_scan.h"

using namespace b40c;

/******************************************************************************
 * Defines, constants, globals
 ******************************************************************************/

bool 	g_verbose 						= false;
bool 	g_sweep							= false;
int 	g_max_ctas 						= 0;
int 	g_iterations  					= 1;
bool 	g_inclusive						= false;


/******************************************************************************
 * Utility Routines
 ******************************************************************************/

/**
 * Displays the commandline usage for this tool
 */
void Usage()
{
	printf("\ntest_segmented_scan [--device=<device index>] [--v] [--i=<num-iterations>] "
			"[--max-ctas=<max-thread-blocks>] [--n=<num-elements>] [--inclusive] [--sweep]\n");
	printf("\n");
	printf("\t--v\tDisplays copied results to the console.\n");
	printf("\n");
	printf("\t--i\tPerforms the segmented scan operation <num-iterations> times\n");
	printf("\t\t\ton the device. Re-copies original input each time. Default = 1\n");
	printf("\n");
	printf("\t--n\tThe number of elements to comprise the sample problem\n");
	printf("\t\t\tDefault = 512\n");
	printf("\n");
}



/**
 * Creates an example segmented scan problem and then dispatches the problem
 * to the GPU for the given number of iterations, displaying runtime information.
 */
template<
	typename T,
	typename Flag,
	bool EXCLUSIVE,
	typename SizeT,
	typename ReductionOp,
	typename IdentityOp>
void TestSegmentedScan(
	SizeT num_elements,
	ReductionOp scan_op,
	IdentityOp identity_op)
{
    // Allocate the segmented scan problem on the host and fill the keys with random bytes

	T *h_data 			= (T*) malloc(num_elements * sizeof(T));
	T *h_reference 		= (T*) malloc(num_elements * sizeof(T));
	Flag *h_flag_data	= (Flag*) malloc(num_elements * sizeof(Flag));

	if ((h_data == NULL) || (h_reference == NULL) || (h_flag_data == NULL)){
		fprintf(stderr, "Host malloc of problem data failed\n");
		exit(1);
	}

	for (size_t i = 0; i < num_elements; ++i) {
	}

	if (g_verbose) printf("Input problem: \n");
	for (int i = 0; i < num_elements; i++) {
/*
		h_data[i] = 1;
		h_flag_data[i] = (i % 11) == 0;
*/
		util::RandomBits<T>(h_data[i], 0);
		util::RandomBits<Flag>(h_flag_data[i], 2, 1);

		if (g_verbose) {
			printf("(%lld, %lld), ", (long long) h_data[i], (long long) h_flag_data[i]);
		}
	}
	if (g_verbose) printf("\n");


	for (size_t i = 0; i < num_elements; ++i) {
		if (EXCLUSIVE)
		{
			h_reference[i] = ((i == 0) || (h_flag_data[i])) ?
				identity_op() :
				scan_op(h_reference[i - 1], h_data[i - 1]);
		} else {
			h_reference[i] = ((i == 0) || (h_flag_data[i])) ?
				h_data[i] :
				scan_op(h_reference[i - 1], h_data[i]);
		}
	}

	//
    // Run the timing test(s)
	//

	// Execute test(s), optionally sweeping problem size downward
	size_t orig_num_elements = num_elements;
	do {

		printf("\nLARGE config:\t");
		double large = TimedSegmentedScan<EXCLUSIVE, segmented_scan::LARGE_SIZE>(
			h_data,
			h_flag_data,
			h_reference,
			num_elements,
			scan_op,
			identity_op,
			g_max_ctas,
			g_verbose,
			g_iterations);

		printf("\nSMALL config:\t");
		double small = TimedSegmentedScan<EXCLUSIVE, segmented_scan::SMALL_SIZE>(
			h_data,
			h_flag_data,
			h_reference,
			num_elements,
			scan_op,
			identity_op,
			g_max_ctas,
			g_verbose,
			g_iterations);

		if (small > large) {
			printf("%lu-byte elements: Small faster at %lu elements\n",
				(unsigned long) sizeof(T), (unsigned long) num_elements);
		}

		num_elements -= 4096;

	} while (g_sweep && (num_elements < orig_num_elements ));

	// Free our allocated host memory
	if (h_flag_data) free(h_flag_data);
	if (h_data) free(h_data);
    if (h_reference) free(h_reference);
}


/**
 * Creates an example segmented scan problem and then dispatches the problem
 * to the GPU for the given number of iterations, displaying runtime information.
 */
template<
	typename T,
	typename Flag,
	typename SizeT,
	typename ReductionOp,
	typename IdentityOp>
void TestSegmentedScanVariety(
	SizeT num_elements,
	ReductionOp scan_op,
	IdentityOp identity_op)
{
	if (g_inclusive) {
		TestSegmentedScan<T, Flag, false>(num_elements, scan_op, identity_op);
	} else {
		TestSegmentedScan<T, Flag, true>(num_elements, scan_op, identity_op);
	}
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
	SizeT num_elements = 1024;

	// Parse command line arguments
    if (args.CheckCmdLineFlag("help")) {
		Usage();
		return 0;
	}
    g_inclusive = args.CheckCmdLineFlag("inclusive");
    g_sweep = args.CheckCmdLineFlag("sweep");
    args.GetCmdLineArgument("i", g_iterations);
    args.GetCmdLineArgument("n", num_elements);
    args.GetCmdLineArgument("max-ctas", g_max_ctas);
	g_verbose = args.CheckCmdLineFlag("v");

	typedef unsigned char Flag;


	{
		printf("\n-- UNSIGNED CHAR ----------------------------------------------\n");
		typedef unsigned char T;
		Sum<T> op;
		TestSegmentedScanVariety<T, Flag>(num_elements * 4, op, op);
	}
	{
		printf("\n-- UNSIGNED SHORT ----------------------------------------------\n");
		typedef unsigned short T;
		Sum<T> op;
		TestSegmentedScanVariety<T, Flag>(num_elements * 2, op, op);
	}
	{
		printf("\n-- UNSIGNED INT -----------------------------------------------\n");
		typedef unsigned int T;
		Sum<T> op;
		TestSegmentedScanVariety<T, Flag>(num_elements, op, op);
	}
	{
		printf("\n-- UNSIGNED LONG LONG -----------------------------------------\n");
		typedef unsigned long long T;
		Sum<T> op;
		TestSegmentedScanVariety<T, Flag>(num_elements / 2, op, op);
	}

	return 0;
}



