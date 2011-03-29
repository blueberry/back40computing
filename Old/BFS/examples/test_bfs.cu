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
#include <math.h>

#include <iostream>
#include <fstream>
#include <string>

#include <test_utils.cu>				// Utilities and correctness-checking
#include <b40c_util.h>					// Misc. utils (random-number gen, I/O, etc.)

// BFS includes
#include <bfs_single_grid.cu>
#include <bfs_level_grid.cu>

using namespace b40c;


/******************************************************************************
 * Defines, constants, globals 
 ******************************************************************************/

//#define __B40C_ERROR_CHECKING__		 

bool g_verbose;
bool g_verbose2;
bool g_undirected;


/******************************************************************************
 * Housekeeping Routines
 ******************************************************************************/

/**
 * Displays the commandline usage for this tool
 */
void Usage() 
{
	printf("\ntest_bfs <graph type> <graph type args> [--device=<device index>] "
			"[--v] [--instrumented] [--i=<num-iterations>] [--undirected]"
			"[--src=< <source idx> | randomize >] [--queue-size=<queue size>\n"
			"\n"
			"graph types and args:\n"
			"\tgrid2d <width>\n"
			"\t\t2D square grid lattice with width <width>.  Interior vertices \n"
			"\t\thave 4 neighbors.  Default source vertex is the grid-center.\n"
			"\tgrid3d <side-length>\n"
			"\t\t3D square grid lattice with width <width>.  Interior vertices \n"
			"\t\thave 6 neighbors.  Default source vertex is the grid-center.\n"
			"\tdimacs [<file>]\n"
			"\t\tReads a DIMACS-formatted graph of directed edges from stdin (or \n"
			"\t\tfrom the optionally-specified file).  Default source vertex is random.\n" 
			"\trandom <n> <m>\n"			
			"\t\tA random graph generator that adds <m> edges to <n> nodes by randomly \n"
			"\t\tchoosing a pair of nodes for each edge.  There are possibilities of \n"
			"\t\tloops and multiple edges between pairs of nodes. Default source vertex \n"
			"\t\tis random.\n"
			"\trr <n> <d>\n"			
			"\t\tA random graph generator that adds <d> randomly-chosen edges to each\n"
			"\t\tof <n> nodes.  There are possibilities of loops and multiple edges\n"
			"\t\tbetween pairs of nodes. Default source vertex is random.\n"
			"\n"
			"--v\tVerbose launch and statistical output is displayed to the console.\n"
			"\n"
			"--v2\tSame as --v, but also displays the input graph to the console.\n"
			"\n"
			"--instrumented\tKernels keep track of queue-passes, redundant work (i.e., the \n"
			"\t\toverhead of duplicates in the frontier), and average barrier wait (a \n"
			"\t\trelative indicator of load imbalance.)\n"
			"\n"
			"--i\tPerforms <num-iterations> test-iterations of BFS traversals.\n"
			"\t\tDefault = 1\n"
			"\n"
			"--src\tBegins BFS from the vertex <source idx>. Default is specific to \n"
			"\t\tgraph-type.  If alternatively specified as \"randomize\", each \n"
			"\t\ttest-iteration will begin with a newly-chosen random source vertex.\n"
			"\n"
			"--queue-size\tAllocates a frontier queue of <queue size> elements.  Default\n"
			"\t\tis the size of the edge list.\n"
			"\n"
			"--undirected\tEdges are undirected.  Reverse edges are added to DIMACS and\n"
			"\t\trandom graphs, effectively doubling the CSR graph representation size.\n"
			"\t\tGrid2d/grid3d graphs are undirected regardless of this flag, and rr \n"
			"\t\tgraphs are directed regardless of this flag.\n"
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


struct Statistic 
{
	double mean;
	double m2;
	int count;
	
	Statistic() : mean(0.0), m2(0.0), count(0) {}
	
	/**
	 * Updates running statistic, returning bias-corrected sample variance.
	 * Online method as per Knuth.
	 */
	double Update(double sample) {
		count++;
		double delta = sample - mean;
		mean = mean + (delta / count);
		m2 = m2 + (delta * (sample - mean));
		return m2 / (count - 1);					// bias-corrected 
	}
	
};

struct Stats {
	char *name;
	Statistic rate;
	Statistic passes;
	Statistic redundant_work;
	Statistic barrier_wait;
	
	Stats() : name(NULL), rate(), passes(), redundant_work(), barrier_wait() {}
	Stats(char *name) : name(name), rate(), passes(), redundant_work(), barrier_wait() {}
};


/**
 * Displays timing and correctness statistics 
 */
template <typename IndexType, typename ValueType>
void DisplayStats(
	bool queues_nodes,
	Stats &stats,
	IndexType src,
	IndexType *h_source_dist,							// computed answer
	IndexType *reference_source_dist,					// reference answer
	const CsrGraph<IndexType, ValueType> &csr_graph,	// reference host graph
	double elapsed, 
	int passes,
	int total_queued,
	double avg_barrier_wait)
{
	// Compute nodes and edges visited
	int edges_visited = 0;
	int nodes_visited = 0;
	for (IndexType i = 0; i < csr_graph.nodes; i++) {
		if (h_source_dist[i] > -1) {
			nodes_visited++;
			edges_visited += csr_graph.row_offsets[i + 1] - csr_graph.row_offsets[i];
		}
	}
	
	double redundant_work = 0.0;
	if (total_queued > 0)  {
		redundant_work = (queues_nodes) ? 
			((double) total_queued - nodes_visited) / nodes_visited : 
			((double) total_queued - edges_visited) / edges_visited;
	}
	redundant_work *= 100;

	// Display name (and correctness)
	printf("[%s]: ", stats.name);
	if (reference_source_dist != NULL) {
		CompareResults(h_source_dist, reference_source_dist, csr_graph.nodes, true);
	}
	printf("\n");

	if (nodes_visited < 5) {
	
		printf("Fewer than 5 vertices visited.\n");

	} else {
		
		// Display the specific sample statistics
		double m_teps = (double) edges_visited / (elapsed * 1000.0); 
		printf("\telapsed: %.3f ms, rate: %.3f MiEdges/s", elapsed, m_teps);
		if (passes != 0) printf(", passes: %d", passes);
		if (avg_barrier_wait != 0) printf("\n\tavg cta waiting: %.3f ms (%.2f%%), avg g-barrier wait: %.4f ms", 
			avg_barrier_wait, avg_barrier_wait / elapsed * 100, avg_barrier_wait / passes);
		printf("\n\tsrc: %d, nodes visited: %d, edges visited: %d", src, nodes_visited, edges_visited);
		if (redundant_work != 0) printf(", redundant work: %.2f%%", redundant_work);
		printf("\n");

		// Display the aggregate sample statistics
		printf("\tSummary after %d test iterations (bias-corrected):\n", stats.rate.count + 1); 

		double passes_stddev = sqrt(stats.passes.Update((double) passes));
		if (passes != 0) printf(			"\t\t[Passes]:           u: %.1f, s: %.1f, cv: %.4f\n", 
			stats.passes.mean, passes_stddev, passes_stddev / stats.passes.mean);

		double redundant_work_stddev = sqrt(stats.redundant_work.Update(redundant_work));
		if (redundant_work != 0) printf(	"\t\t[redundant work %]: u: %.2f, s: %.2f, cv: %.4f\n", 
			stats.redundant_work.mean, redundant_work_stddev, redundant_work_stddev / stats.redundant_work.mean);

		double barrier_wait_stddev = sqrt(stats.barrier_wait.Update(avg_barrier_wait / elapsed * 100));
		if (avg_barrier_wait != 0) printf(	"\t\t[Waiting %]:        u: %.2f, s: %.2f, cv: %.4f\n", 
			stats.barrier_wait.mean, barrier_wait_stddev, barrier_wait_stddev / stats.barrier_wait.mean);

		double rate_stddev = sqrt(stats.rate.Update(m_teps));
		printf(								"\t\t[Rate MiEdges/s]:   u: %.3f, s: %.3f, cv: %.4f\n", 
			stats.rate.mean, rate_stddev, rate_stddev / stats.rate.mean);
	}
	
	fflush(stdout);

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
/*
	// Create timer
	unsigned int timer;
	cutCreateTimer(&timer);
*/
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
	
//	cutStartTimer(timer);
	while (!frontier.empty()) {
		
		// Dequeue node from frontier
		IndexType dequeued_node = frontier.front();  
		frontier.pop_front();
		IndexType dist = source_dist[dequeued_node];

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
/*
	cutStopTimer(timer);
	
	// Cleanup
	float elapsed = cutGetTimerValue(timer);
	cutDeleteTimer(timer);
	
	return elapsed;
*/
	return 99999.0;
}


/**
 * Runs tests
 */
template <typename IndexType, typename ValueType, bool INSTRUMENT> 
void RunTests(
	const CsrGraph<IndexType, ValueType> &csr_graph,
	IndexType src,
	bool randomized_src,
	int test_iterations,
	int max_grid_size,
	int queue_size) 
{
	// Allocate host-side source_distance array (for both reference and gpu-computed results)
	IndexType* reference_source_dist 	= (IndexType*) malloc(sizeof(IndexType) * csr_graph.nodes);
	IndexType* h_source_dist 			= (IndexType*) malloc(sizeof(IndexType) * csr_graph.nodes);

	// Allocate a BFS enactor (with maximum frontier-queue size the size of the edge-list)
	SingleGridBfsEnactor<IndexType, INSTRUMENT> bfs_sg_enactor(
		(queue_size > 0) ? queue_size : csr_graph.edges * 3 / 2, 
		max_grid_size);

/*
	LevelGridBfsEnactor<IndexType> bfs_sg_enactor(
		(queue_size > 0) ? queue_size : csr_graph.edges * 3 / 2,
		max_grid_size);
*/
	bfs_sg_enactor.BFS_DEBUG = g_verbose;

	// Allocate problem on GPU
	BfsCsrProblem<IndexType> problem_storage(
		csr_graph.nodes, csr_graph.edges, csr_graph.column_indices, csr_graph.row_offsets);

	
	Stats stats[3];
	stats[0] = Stats("Simple CPU BFS");
	stats[1] = Stats("Single-grid, expand-contract GPU BFS");
	stats[2] = Stats("Single-grid, contract-expand GPU BFS"); 
	
	printf("Running %s tests...\n\n", (INSTRUMENT) ? "instrumented" : "non-instrumented");
	
	// Perform the specified number of test iterations
	int test_iteration = 0;
	while (test_iteration < test_iterations) {
	
		// If randomized-src was specified, re-roll the src
		if (randomized_src) src = RandomNode(csr_graph.nodes);
		
		double elapsed = 0.0;
		int total_queued = 0;
		int passes = 0;
		double avg_barrier_wait = 0.0; 

		printf("---------------------------------------------------------------\n");

		// Compute reference CPU BFS solution
		elapsed = SimpleReferenceBfs(csr_graph, reference_source_dist, src);
		DisplayStats<IndexType, ValueType>(
			true, stats[0], src, reference_source_dist, NULL, csr_graph, 
			elapsed, passes, total_queued, avg_barrier_wait);
		printf("\n");

		// Perform expand-contract GPU BFS search
		elapsed = TestGpuBfs(bfs_sg_enactor, problem_storage, src, EXPAND_CONTRACT, h_source_dist);
		bfs_sg_enactor.GetStatistics(total_queued, passes, avg_barrier_wait);	
		DisplayStats<IndexType, ValueType>(
			true, stats[1], src, h_source_dist, reference_source_dist, csr_graph,
			elapsed, passes, total_queued, avg_barrier_wait);
		printf("\n");

		// Perform contract-expand GPU BFS search
		elapsed = TestGpuBfs(bfs_sg_enactor, problem_storage, src, CONTRACT_EXPAND, h_source_dist);
		bfs_sg_enactor.GetStatistics(total_queued, passes, avg_barrier_wait);		
		DisplayStats<IndexType, ValueType>(
			false, stats[2], src, h_source_dist, reference_source_dist, csr_graph,
			elapsed, passes, total_queued, avg_barrier_wait);
		printf("\n");

		if (g_verbose2) {
			DisplaySolution(reference_source_dist, csr_graph.nodes);
			printf("\n");
		}
		
		if (randomized_src) {
			test_iteration = stats[0].rate.count;
		} else {
			test_iteration++;
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


/******************************************************************************
 * Main
 ******************************************************************************/

int main( int argc, char** argv)  
{
	typedef int IndexType;						// Use int's as the node identifier (we could use long longs for large graphs)					
	typedef int ValueType;						// Use int's as the value type
	
	IndexType src 			= -1;				// Use whatever default for the specified graph-type 
	char* src_str			= NULL;
	bool randomized_src		= false;
	bool instrumented;
	int test_iterations 	= 1;
	int max_grid_size 		= 0;				// Default: leave it up the enactor
	int queue_size			= -1;				// Default: the size of the edge list

	CommandLineArgs args(argc, argv);
	DeviceInit(args);

	srand(0);									// Presently deterministic
	//srand(time(NULL));	

	
	//
	// Check command line arguments
	// 
	
	if (args.CheckCmdLineFlag("help")) {
		Usage();
		return 1;
	}
	instrumented = args.CheckCmdLineFlag("instrumented");
	args.GetCmdLineArgument("src", src_str);
	if (src_str != NULL) {
		if (strcmp(src_str, "randomize") == 0) {
			randomized_src = true;
		} else {
			src = atoi(src_str);
		}
	}
	g_undirected = args.CheckCmdLineFlag("undirected");
	args.GetCmdLineArgument("i", test_iterations);
	args.GetCmdLineArgument("max-ctas", max_grid_size);
	args.GetCmdLineArgument("queue-size", queue_size);
	if (g_verbose2 = args.CheckCmdLineFlag("v2")) {
		g_verbose = true;
	} else {
		g_verbose = args.CheckCmdLineFlag("v");
	}
	int flags = args.ParsedArgc();
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
		if (BuildDimacsGraph(dimacs_filename, src, csr_graph, g_undirected) != 0) {
			return 1;
		}
		
	} else if (graph_type == "random") {
		// Random graph of n nodes and m edges
		if (graph_args < 3) { Usage(); return 1; }
		IndexType nodes = atol(argv[2]);
		IndexType edges = atol(argv[3]);
		if (BuildRandomGraph(nodes, edges, src, csr_graph, g_undirected) != 0) {
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

	if (instrumented) {
		// Run instrumented kernel for runtime statistics
		RunTests<IndexType, ValueType, true>(
			csr_graph, src, randomized_src, test_iterations, max_grid_size, queue_size);
	} else {
		// Run regular kernel 
		RunTests<IndexType, ValueType, false>(
			csr_graph, src, randomized_src, test_iterations, max_grid_size, queue_size);
	}
}
