/******************************************************************************
 * Copyright 2010 Duane Merrill
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
 * Thanks!
 ******************************************************************************/


/******************************************************************************
 * Simple test driver program for BFS graph traversal API.
 *
 * Useful for demonstrating how to integrate BFS traversal into your 
 * application. 
 ******************************************************************************/

#include <stdio.h> 

// Sorting includes
#include <bfs_ce_sg.cu>
#include <bfs_ec_sg.cu>		

#include <test_utils.cu>				// Utilities and correctness-checking
#include <cutil.h>						// Utilities for commandline parsing
#include <b40c_util.h>					// Misc. utils (random-number gen, I/O, etc.)

using namespace b40c;


/******************************************************************************
 * Defines, constants, globals 
 ******************************************************************************/

bool g_verbose;


/******************************************************************************
 * Routines
 ******************************************************************************/

/**
 * Displays the commandline usage for this tool
 */
void Usage() 
{
	printf("\ntest_bfs [--device=<device index>] [--v] [--i=<num-iterations>]\n"); 
	printf("\n");
	printf("\t--v\tDisplays sorted results to the console.\n");
	printf("\n");
	printf("\t--i\tPerforms the BFS traversal <num-iterations> times\n");
	printf("\t\t\ton the device. Re-copies original input each time. Default = 1\n");
	printf("\n");
}


/******************************************************************************
 * Main
 ******************************************************************************/

int main( int argc, char** argv) {

	CUT_DEVICE_INIT(argc, argv);

	//srand(time(NULL));	
	srand(0);				// presently deterministic

	// Check command line arguments
    if (cutCheckCmdLineFlag( argc, (const char**) argv, "help")) {
		Usage();
		return 0;
	}

    // Blah
    
	cudaThreadSynchronize();
}
