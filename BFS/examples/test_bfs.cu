/******************************************************************************
 * Copyright 2010 Duane Merrill
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *	 http://www.apache.org/licenses/LICENSE-2.0
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

#include <iostream>
#include <fstream>
#include <string>

// Sorting includes
#include <bfs_single_grid.cu>

#include <test_utils.cu>				// Utilities and correctness-checking
#include <cutil.h>						// Utilities for commandline parsing
#include <b40c_util.h>					// Misc. utils (random-number gen, I/O, etc.)

using namespace b40c;


/******************************************************************************
 * Defines, constants, globals 
 ******************************************************************************/

bool g_verbose;
bool g_verbose2;


/******************************************************************************
 * Housekeeping Routines
 ******************************************************************************/

/**
 * Displays the commandline usage for this tool
 */
void Usage() 
{
	printf("\ntest_bfs <graph-type> <graph-type-args> [--device=<device index>] "
			"[--v] [--method=<cesg | ecsg>] [--i=<num-iterations>] [--src=< <source idx> | randomize >]\n"
			"\n"
			"graph-types and args:\n"
			"\tgrid2d <width>\n"
			"\t\t2D square grid lattice with width <width>.  Interior vertices \n"
			"\t\thave 4 neighbors.  Default source vertex is the grid-center.\n"
			"\tgrid3d <side-length>\n"
			"\t\t3D square grid lattice with width <width>.  Interior vertices \n"
			"\t\thave 6 neighbors.  Default source vertex is the grid-center.\n"
			"\tdimacs [<file>]\n"
			"\t\tReads a DIMACS-formatted graph from stdin (or from the optionally-\n "
			"\t\tspecified file).  Default source vertex is random.\n" 
			"\trandom <n> <m>\n"			
			"\t\tA random graph generator that adds <m> edges to <n> nodes by randomly \n"
			"\t\tchoosing a pair of nodes for each edge.  There are possibilities of \n"
			"\t\tloops and multiple edges between pairs of nodes. Default source vertex \n"
			"\t\tis random.\n"
			"\n"
			"--v\tVerbose launch and statistical output is displayed to the console.\n"
			"\n"
			"--v2\tSame as --v, but also displays the input graph to the console.\n"
			"\n"
			"--method\tSpecifies the BFS algorithm to use.  Valid choices are:\n"
			"\t\t\tcesg\tContract-expand, single-grid [default]\n"
			"\t\t\tecsg\tExpand-contract, single-grid\n"
			"\n"
			"--i\tPerforms the BFS traversal <num-iterations> times\n"
			"\t\ton the device. Default = 1\n"
			"\n"
			"--src\tBegins BFS from the vertex <source idx>. Default is specific to \n"
			"\t\tgraph-type.  If alternatively specified as \"randomize\", each \n"
			"\t\ttest-iteration will begin with a newly-chosen random source vertex.\n"
			"\n");
}

/**
 * Displays the BFS result (i.e., distance from source)
 */
template<typename IndexType>
void DisplaySolution(IndexType* source_dist, IndexType nodes)
{
	printf("Solution: [");
	for (IndexType i = 0; i < nodes; i++) {
		PrintValue(i);
		printf(":");
		PrintValue(source_dist[i]);
		printf(", ");
	}
	printf("]\n");
}


/**
 * Displays timing and correctness statistics 
 */
template <typename IndexType, typename ValueType>
void DisplayStats(
	char *name,
	IndexType src,
	float elapsed,										// time taken to compute solution
	IndexType *h_source_dist,							// computed answer
	IndexType *reference_source_dist,					// reference answer
	const CsrGraph<IndexType, ValueType> &csr_graph)	// reference host graph
{
	// Compute nodes and edges visited
	IndexType edges_visited = 0;
	IndexType nodes_visited = 0;
	for (IndexType i = 0; i < csr_graph.nodes; i++) {
		if (h_source_dist[i] > -1) {
			nodes_visited++;
			edges_visited += csr_graph.row_offsets[i + 1] - csr_graph.row_offsets[i];
		}
	}
	
	// Display time
	printf("[%s]: ", name);

	// Verify
	if (reference_source_dist != NULL) {
		CompareResults(h_source_dist, reference_source_dist, csr_graph.nodes, true);
	}
	printf("\n\tsrc: ");
	PrintValue(src);
	printf(", nodes visited: ");
	PrintValue(nodes_visited);
	printf(", edges visited: ");
	PrintValue(edges_visited);
	printf("\n\t%6.3f ms, \tMiEdges/s: %6.3f\n", elapsed, (float) edges_visited / (elapsed * 1000.0));
	
}
		

/******************************************************************************
 * BFS Testing Routines
 ******************************************************************************/

template <typename IndexType, typename BfsEnactor, typename ProblemStorage>
float TestGpuBfs(
	BfsEnactor &enactor,
	ProblemStorage &problem_storage,
	IndexType src,
	BfsStrategy strategy,
	IndexType *h_source_dist)						// place to copy results out to
{
	// Create timer
	float elapsed;
	cudaEvent_t start_event, stop_event;
	cudaEventCreate(&start_event);
	cudaEventCreate(&stop_event);

	// (Re)initialize distances
	problem_storage.ResetSourceDist();

	synchronize("Pre-launch check");
	
	// Perform BFS
	cudaEventRecord(start_event, 0);
	enactor.EnactSearch(problem_storage, src, strategy);
	cudaEventRecord(stop_event, 0);
	cudaEventSynchronize(stop_event);
	cudaEventElapsedTime(&elapsed, start_event, stop_event);

	// Copy out results
	cudaMemcpy(
		h_source_dist, 
		problem_storage.d_source_dist, 
		problem_storage.nodes * sizeof(IndexType), 
		cudaMemcpyDeviceToHost);
	
    // Clean up events
	cudaEventDestroy(start_event);
	cudaEventDestroy(stop_event);
	
	return elapsed;
}


/**
 * A simple CPU-based reference BFS ranking implementation.  
 * 
 * Computes the distance of each node from the specified source node. 
 */
template<typename IndexType, typename ValueType>
float SimpleReferenceBfs(
	const CsrGraph<IndexType, ValueType> &csr_graph,
	IndexType *source_dist,
	IndexType src)
{
	// Create timer
	unsigned int timer;
	cutCreateTimer(&timer);

	// (Re)initialize distances
	for (IndexType i = 0; i < csr_graph.nodes; i++) {
		source_dist[i] = -1;
	}
	source_dist[src] = 0;

	// Initialize queue for managing previously-discovered nodes
	std::deque<IndexType> frontier;
	frontier.push_back(src);

	//
	// Perform BFS 
	//
	
	cutStartTimer(timer);
	while (!frontier.empty()) {
		
		// Dequeue node from frontier
		IndexType dequeued_node = frontier.front();  
		IndexType dist = source_dist[dequeued_node];
		frontier.pop_front();

		// Locate adjacency list
		int edges_begin = csr_graph.row_offsets[dequeued_node];
		int edges_end = csr_graph.row_offsets[dequeued_node + 1];

		for (int edge = edges_begin; edge < edges_end; edge++) {

			// Lookup neighbor and enqueue if undiscovered 
			IndexType neighbor = csr_graph.column_indices[edge];
			if (source_dist[neighbor] == -1) {
				source_dist[neighbor] = dist + 1;
				frontier.push_back(neighbor);
			}
		}
	}
	cutStopTimer(timer);
	
	// Cleanup
	float elapsed = cutGetTimerValue(timer);
	cutDeleteTimer(timer);
	
	return elapsed;
}



/******************************************************************************
 * Main
 ******************************************************************************/

int main( int argc, char** argv)  
{
	typedef int IndexType;						// Use int's as the node identifier (we could use long longs for large graphs)					
	typedef int ValueType;						// Use int's as the value type
	
	IndexType src 			= -1;				// Use whatever default for the specified graph-type 
	char* bfs_method_str	= NULL;
	char* src_str			= NULL;
	bool randomized_src 	= false;
	int iterations 			= 1;
	int max_grid_size 		= 0;				// Default: leave it up the enactor

	CUT_DEVICE_INIT(argc, argv);
	srand(0);									// Presently deterministic
	//srand(time(NULL));	

	
	//
	// Check command line arguments
	// 
	
	if (cutCheckCmdLineFlag( argc, (const char**) argv, "help")) {
		Usage();
		return 1;
	}
	cutGetCmdLineArgumentstr( argc, (const char**) argv, "method", &bfs_method_str);
	std::string bfs_method = (bfs_method_str == NULL) ? "" : bfs_method_str;
	cutGetCmdLineArgumentstr( argc, (const char**) argv, "src", &src_str);
	if (src_str != NULL) {
		if (strcmp(src_str, "randomize") == 0) {
			randomized_src = true;
		} else {
			src = atoi(src_str);
		}
	}
	cutGetCmdLineArgumenti( argc, (const char**) argv, "i", &iterations);
	cutGetCmdLineArgumenti( argc, (const char**) argv, "max-ctas", &max_grid_size);
	if (g_verbose2 = cutCheckCmdLineFlag( argc, (const char**) argv, "v2")) {
		g_verbose = true;
	} else {
		g_verbose = cutCheckCmdLineFlag( argc, (const char**) argv, "v");
	}
	BFS_DEBUG = g_verbose;
	int flags = CmdArgReader::getParsedArgc();
	int graph_args = argc - flags - 1;
	
	
	//
	// Obtain CSR search graph
	//

	CsrGraph<IndexType, ValueType> csr_graph;
	
	if (graph_args < 1) {
		Usage();
		return 1;
	}
	std::string graph_type = argv[1];
	if (graph_type == "grid2d") {
		// Two-dimensional regular lattice grid (degree 4)
		if (graph_args < 2) { Usage(); return 1; }
		IndexType width = atoi(argv[2]);
		if (BuildGrid2dGraph(width, src, csr_graph) != 0) {
			return 1;
		}

	} else if (graph_type == "grid3d") {
		// Three-dimensional regular lattice grid (degree 6)
		if (graph_args < 2) { Usage(); return 1; }
		IndexType width = atoi(argv[2]);
		if (BuildGrid3dGraph(width, src, csr_graph) != 0) {
			return 1;
		}

	} else if (graph_type == "dimacs") {
		// DIMACS-formatted graph file
		if (graph_args < 1) { Usage(); return 1; }
		char *dimacs_filename = (graph_args == 2) ? argv[2] : NULL;
		if (BuildDimacsGraph(dimacs_filename, src, csr_graph) != 0) {
			return 1;
		}
		
	} else if (graph_type == "random") {
		// Random graph of n nodes and m edges
		if (graph_args < 3) { Usage(); return 1; }
		IndexType nodes = atol(argv[2]);
		IndexType edges = atol(argv[3]);
		if (BuildRandomGraph(nodes, edges, src, csr_graph) != 0) {
			return 1;
		}

	} else if (graph_type == "rr") {
		// Random-regular-ish graph of n nodes, each with degree d (allows loops and cycles)
		if (graph_args < 3) { Usage(); return 1; }
		IndexType nodes = atol(argv[2]);
		int degree = atol(argv[3]);
		if (BuildRandomRegularishGraph(nodes, degree, src, csr_graph) != 0) {
			return 1;
		}

	} else {
		// Unknown graph type
		fprintf(stderr, "Unspecified graph type\n");
		return 1;
	}
	
	// Optionally display graph
	if (g_verbose2) {
		printf("\n");
		DisplayGraph(csr_graph);
		printf("\n");
	}
	
	
	//
	// Allocate problem storage and enactors
	//
	
	// Allocate host-side source_distance array (for both reference and gpu-computed results)
	IndexType* reference_source_dist 	= (IndexType*) malloc(sizeof(IndexType) * csr_graph.nodes);
	IndexType* h_source_dist 			= (IndexType*) malloc(sizeof(IndexType) * csr_graph.nodes);

	// Allocate a BFS enactor (with maximum frontier-queue size of 4/5 the size of the edge-list)
	SingleGridBfsEnactor<IndexType> bfs_sg_enactor(csr_graph.edges / 5 * 4, max_grid_size);

	// Allocate problem on GPU
	BfsCsrProblem<IndexType> problem_storage(
		csr_graph.nodes, csr_graph.edges, csr_graph.column_indices, csr_graph.row_offsets);

	
	//
	// Perform the specified number of test iterations
	//

	for (int iteration = 0; iteration < iterations; iteration++) {
	
		printf("\n");
		
		// If randomized-src was specified, re-roll the src
		if (randomized_src) src = RandomNode(csr_graph.nodes);
		
		// Compute reference CPU BFS solution
		float elapsed = SimpleReferenceBfs(csr_graph, reference_source_dist, src);
		DisplayStats<IndexType, ValueType>("Simple CPU BFS", src, elapsed, h_source_dist, NULL, csr_graph);

		// Perform contract-expand GPU BFS search
		elapsed = TestGpuBfs(bfs_sg_enactor, problem_storage, src, CONTRACT_EXPAND, h_source_dist);
		DisplayStats<IndexType, ValueType>("Single-grid, contract-expand GPU BFS", 
			src, elapsed, h_source_dist, reference_source_dist, csr_graph);
		
/*		
		// Perform expand-contract GPU BFS search
		elapsed = TestGpuBfs(bfs_sg_enactor, problem_storage, src, EXPAND_CONTRACT, h_source_dist);
		DisplayStats<IndexType, ValueType>("Single-grid, expand-contract GPU BFS", 
			src, elapsed, h_source_dist, reference_source_dist, csr_graph);
*/

		if (g_verbose2) {
			DisplaySolution(reference_source_dist, csr_graph.nodes);
			printf("\n");
		}
		
	}
	
	
	//
	// Cleanup
	//
	
	if (reference_source_dist) free(reference_source_dist);
	if (h_source_dist) free(h_source_dist);
	problem_storage.Free();
	
	cudaThreadSynchronize();
}
