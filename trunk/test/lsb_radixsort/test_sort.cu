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
 * Simple test driver program for sort.
 ******************************************************************************/

#include <stdio.h> 

// Sort includes
#include <b40c/sort_storage.cuh>
#include <b40c/util/basic_utils.cuh>
#include <b40c/util/error_utils.cuh>
#include <b40c/sort/sorting_utils.cuh>
//#include <b40c/sort/granularity_tuned.cuh>
//#include <b40c/sort_enactor_tuned.cuh>

// Test utils
#include "b40c_util.h"
#include "test_sort.h"

using namespace b40c;

/******************************************************************************
 * Defines, constants, globals
 ******************************************************************************/

bool 	g_value_bytes					= 0;
bool 	g_verbose 						= false;
bool 	g_sweep							= false;
int 	g_max_ctas 						= 0;
int 	g_iterations  					= 1;
size_t 	g_num_elements					= 512;


/******************************************************************************
 * Utility Routines
 ******************************************************************************/

/**
 * Displays the commandline usage for this tool
 */
void Usage()
{
	printf("\ntest_sort [--device=<device index>] [--v] [--i=<num-iterations>] "
			"[--max-ctas=<max-thread-blocks>] [--n=<num-elements>]\n");
	printf("\n");
	printf("\t--v\tDisplays copied results to the console.\n");
	printf("\n");
	printf("\t--i\tPerforms the sort operation <num-iterations> times\n");
	printf("\t\t\ton the device. Re-copies original input each time. Default = %d\n", g_iterations);
	printf("\n");
	printf("\t--n\tThe number of elements to comprise the sample problem\n");
	printf("\t\t\tDefault = %lu\n", (unsigned long) g_num_elements);
	printf("\t--value-bytes\tSpecifies the size (in bytes) of values to pair\n");
	printf("\t\t\twith keys.  Default: %d bytes\n", g_value_bytes);
	printf("\n");
}



/**
 * Creates an example sort problem and then dispatches the problem
 * to the GPU for the given number of iterations, displaying runtime information.
 */
template<typename K, typename V, typename SizeT>
void TestSort()
{
	typedef SortStorageMultiCta<K, V, SizeT> Storage;

	// Allocate the sort problem on the host and fill the keys with random bytes

	K *h_keys 			= (K*) malloc(g_num_elements * sizeof(K));
	K *h_ref_keys 		= (K*) malloc(g_num_elements * sizeof(K));

	if ((h_keys == NULL) || (h_ref_keys == NULL)){
		fprintf(stderr, "Host malloc of problem data failed\n");
		exit(1);
	}

	printf("Initializing sorting problem on CPU... ");
	fflush(stdout);
	for (size_t i = 0; i < g_num_elements; ++i) {
		RandomBits<K>(h_keys[i], 0);
//		h_keys[i] = i;
		h_ref_keys[i] = h_keys[i];
	}
	printf("Done.\n");
	fflush(stdout);

	// Sort reference solution
	printf("Sorting reference solution on CPU... ");
	fflush(stdout);
	std::sort(h_ref_keys, h_ref_keys + g_num_elements);	
	printf("Done.\n");
	fflush(stdout);

	// Allocate device storage
	Storage device_storage(g_num_elements);
	if (util::B40CPerror(cudaMalloc((void**) &device_storage.d_keys[0], sizeof(K) * g_num_elements),
		"TimedSort cudaMalloc device_storage.d_keys[0] failed: ", __FILE__, __LINE__)) exit(1);
	if (sort::IsKeysOnly<V>()) {
		if (util::B40CPerror(cudaMalloc((void**) &device_storage.d_values[0], sizeof(V) * g_num_elements),
			"TimedSort cudaMalloc device_storage.d_values[0] failed: ", __FILE__, __LINE__)) exit(1);
	}

//	TimedSort<Storage>(device_storage, h_keys, iterations);

	// Free allocated memory
	if (device_storage.d_keys[0]) cudaFree(device_storage.d_keys[0]);
	if (device_storage.d_keys[1]) cudaFree(device_storage.d_keys[1]);
	if (device_storage.d_values[0]) cudaFree(device_storage.d_values[0]);
	if (device_storage.d_values[1]) cudaFree(device_storage.d_values[1]);
	
	// Free our allocated host memory
	if (h_keys) free(h_keys);
    if (h_ref_keys) free(h_ref_keys);
}


/**
 * Invokes sorting test using value type based upon command line option
 */
template<typename K, typename SizeT>
void TestSort()
{
	switch (g_value_bytes) {
	case 0:
		TestSort<K, sort::KeysOnly, SizeT>();
		break;
	case 1:
//		TestSort<K, unsigned char, SizeT>();
		break;
	case 2:
//		TestSort<K, unsigned short, SizeT>();
		break;
	case 4:
		TestSort<K, unsigned int, SizeT>();
		break;
	case 8:
//		TestSort<K, unsigned long long, SizeT>();
		break;
	default:
		printf("Unsupported number of value bytes\n");
	};
}



/******************************************************************************
 * Main
 ******************************************************************************/

int main(int argc, char** argv)
{
	CommandLineArgs args(argc, argv);
	DeviceInit(args);

	//srand(time(NULL));
	srand(0);				// presently deterministic

    //
	// Check command line arguments
    //

    if (args.CheckCmdLineFlag("help")) {
		Usage();
		return 0;
	}

    args.GetCmdLineArgument("value-bytes", g_value_bytes);
    args.GetCmdLineArgument("i", g_iterations);
    args.GetCmdLineArgument("n", g_num_elements);
    args.GetCmdLineArgument("max-ctas", g_max_ctas);
	g_verbose = args.CheckCmdLineFlag("v");

	typedef unsigned int SizeT;
	
//	typedef unsigned char T;
//	typedef unsigned short T;
	typedef unsigned int T;
//	typedef unsigned long long T;

	TestSort<T, SizeT>();

	return 0;
}



