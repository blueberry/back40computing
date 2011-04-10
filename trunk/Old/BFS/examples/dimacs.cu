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

#pragma once

#include <math.h>
#include <time.h>
#include <stdio.h>

#include <test_utils.cu>



/******************************************************************************
 * DIMACS Graph Construction Routines
 ******************************************************************************/

/**
 * Reads a DIMACS graph from an input-stream into a CSR sparse format 
 */
template<bool LOAD_VALUES, typename VertexId, typename Value, typename SizeT>
int ReadDimacsStream(
	FILE *f_in,
	CsrGraph<VertexId, Value, SizeT> &csr_graph,
	bool undirected)
{
	typedef CooEdgeTuple<VertexId, Value> EdgeTupleType;
	
	SizeT edges_read = 0;
	SizeT nodes = 0;
	SizeT edges = 0;
	SizeT directed_edges = 0;
	EdgeTupleType *coo = NULL;		// read in COO format
	
	time_t mark0 = time(NULL);
	printf("  Parsing DIMACS COO format ");
	fflush(stdout);

	char line[1024];
	char problem_type[1024];

	while(true) {

		if (fscanf(f_in, "%[^\n]\n", line) <= 0) {
			break;
		}

		switch (line[0]) {
		case 'p':
		{
			// Problem description (nodes is nodes, edges is edges)
			long long ll_nodes, ll_edges;
			sscanf(line, "p %s %lld %lld", problem_type, &ll_nodes, &ll_edges);
			nodes = ll_nodes;
			edges = ll_edges;

			directed_edges = (undirected) ? edges * 2 : edges;
			printf(" (%lld nodes, %lld %s edges)... ",
				(unsigned long long) ll_nodes, (unsigned long long) ll_edges,
				(undirected) ? "undirected" : "directed");
			fflush(stdout);
			
			// Allocate coo graph
			coo = (EdgeTupleType*) malloc(sizeof(EdgeTupleType) * directed_edges);

			break;
		}
		case 'a':
		{
			// Edge description (v -> w) with value val
			if (!coo) {
				fprintf(stderr, "Error parsing DIMACS graph: invalid format\n");
				return -1;
			}			
			if (edges_read >= edges) {
				fprintf(stderr, "Error parsing DIMACS graph: encountered more than %d edges\n", edges);
				if (coo) free(coo);
				return -1;
			}

			long long ll_row, ll_col, ll_val;
			sscanf(line, "a %lld %lld %lld", &ll_row, &ll_col, &ll_val);

			coo[edges_read].row = ll_row - 1;	// zero-based array
			coo[edges_read].col = ll_col - 1;	// zero-based array
			coo[edges_read].val = ll_val;

			if (undirected) {
				// Reverse edge
				coo[edges + edges_read].row = coo[edges_read].col;
				coo[edges + edges_read].col = coo[edges_read].row;
				coo[edges + edges_read].val = coo[edges_read].val;
			}

			edges_read++;
			break;
		}
		
		default:
			// read remainder of line
			break;
		}
	}
	
	if (coo == NULL) {
		fprintf(stderr, "No graph found\n");
		return -1;
	}

	if (edges_read != edges) {
		fprintf(stderr, "Error parsing DIMACS graph: only %d/%d edges read\n", edges_read, edges);
		if (coo) free(coo);
		return -1;
	}
	
	time_t mark1 = time(NULL);
	printf("Done parsing (%ds).\n", (int) (mark1 - mark0));
	fflush(stdout);
	
	// Convert sorted COO to CSR
	csr_graph.template FromCoo<LOAD_VALUES>(coo, nodes, directed_edges);
	free(coo);

	fflush(stdout);
	
	return 0;
}


/**
 * Loads a DIMACS-formatted CSR graph from the specified file.  If 
 * dimacs_filename == NULL, then it is loaded from stdin.
 * 
 * If src == -1, it is assigned a random node.  Otherwise it is verified 
 * to be in range of the constructed graph.
 */
template<bool LOAD_VALUES, typename VertexId, typename Value, typename SizeT>
int BuildDimacsGraph(
	char *dimacs_filename, 
	VertexId &src,
	CsrGraph<VertexId, Value, SizeT> &csr_graph,
	bool undirected)
{ 
	if (dimacs_filename == NULL) {

		// Read from stdin
		printf("Reading from stdin:\n");
		if (ReadDimacsStream<LOAD_VALUES>(stdin, csr_graph, undirected) != 0) {
			return -1;
		}

	} else {
	
		// Read from file
		FILE *f_in = fopen(dimacs_filename, "r");
		if (f_in) {
			printf("Reading from %s:\n", dimacs_filename);
			if (ReadDimacsStream<LOAD_VALUES>(f_in, csr_graph, undirected) != 0) {
				fclose(f_in);
				return -1;
			}
			fclose(f_in);
		} else {
			perror("Unable to open file");
			return -1;
		}
	}
	
	// If unspecified, assign default source.  Otherwise verify source range.
	if (src == -1) {
		// Random source
		src = RandomNode(csr_graph.nodes);
	} else if ((src < 0 ) || (src > csr_graph.nodes)) {
		fprintf(stderr, "Invalid src: %d", src);
		csr_graph.Free();
		return -1;
	}
	
	return 0;
}

