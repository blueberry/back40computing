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

#include <time.h>
#include <stdio.h>

#include <string>
#include <sstream>
#include <iostream>

#include <fstream>
#include <deque>
#include <algorithm>

#include <b40c_util.h>					// Misc. utils (random-number gen, I/O, etc.)


/******************************************************************************
 * General utility routines
 ******************************************************************************/

/**
 * Returns a random node-ID in the range of [0, num_nodes) 
 */
template<typename IndexType>
IndexType RandomNode(IndexType num_nodes) {
	IndexType node_id;
	b40c::RandomBits(node_id);
	if (node_id < 0) node_id *= -1;
	return node_id % num_nodes;
}


/******************************************************************************
 * Simple COO sparse graph datastructure
 ******************************************************************************/

/**
 * COO sparse format edge.  (A COO graph is just a list/array/vector of these.)
 */
template<typename IndexType, typename ValueType>
struct CooEdgeTuple {
	IndexType row;
	IndexType col;
	ValueType val;
};

/**
 * Comparator for sorting COO sparse format edges
 */
template<typename IndexType, typename ValueType>
bool DimacsTupleCompare (
	CooEdgeTuple<IndexType, ValueType> elem1, 
	CooEdgeTuple<IndexType, ValueType> elem2)
{
	if (elem1.row < elem2.row) {
		return true;
	} else if ((elem1.row == elem2.row) && (elem1.col < elem2.col)) {
		return true;
	} 
	
	return false;
}


/******************************************************************************
 * Simple CSR sparse graph datastructure
 ******************************************************************************/

/**
 * CSR sparse format graph
 */
template<typename IndexType, typename ValueType>
struct CsrGraph {
	IndexType nodes;
	IndexType edges;
	
	IndexType* row_offsets;
	IndexType* column_indices;
	ValueType* values;
	
	/**
	 * Empty constructor
	 */
	CsrGraph() {
		nodes = 0;
		edges = 0;
		row_offsets = NULL;
		column_indices = NULL;
		values = NULL;
	}

	/**
	 * Build CSR graph from sorted COO graph
	 */
	void FromCoo(CooEdgeTuple<IndexType, ValueType> *coo, IndexType coo_nodes, IndexType coo_edges) {
		
		nodes = coo_nodes;
		edges = coo_edges;
		row_offsets = (IndexType*) malloc(sizeof(IndexType) * (nodes + 1));
		column_indices = (IndexType*) malloc(sizeof(IndexType) * edges);
		values = (ValueType*) malloc(sizeof(ValueType) * edges);
		
		IndexType prev_row = -1;
		for (int edge = 0; edge < edges; edge++) {
			
			int current_row = coo[edge].row;
			
			// Fill in rows up to and including the current row
			for (int row = prev_row + 1; row <= current_row; row++) {
				row_offsets[row] = edge;
			}
			prev_row = current_row;
			
			column_indices[edge] = coo[edge].col;
			values[edge] = coo[edge].val;
		}

		// Fill out any trailing edgeless nodes (and the end-of-list element)
		for (int row = prev_row + 1; row <= nodes; row++) {
			row_offsets[row] = edges;
		}
		
	}
	
	void Free() {
		if (row_offsets) { free(row_offsets); row_offsets = NULL; }
		if (column_indices) { free(column_indices); column_indices = NULL; }
		if (values) { free (values); values = NULL; }
		nodes = 0;
		edges = 0;
	}
	
	~CsrGraph() {
		Free();
	}
};


/**
 * Display CSR graph to console
 */
template<typename IndexType, typename ValueType>
void DisplayGraph(const CsrGraph<IndexType, ValueType> &csr_graph) 
{
	printf("Input Graph:\n");
	for (IndexType node = 0; node < csr_graph.nodes; node++) {
		PrintValue(node);
		printf(": ");
		for (IndexType edge = csr_graph.row_offsets[node]; edge < csr_graph.row_offsets[node + 1]; edge++) {
			PrintValue(csr_graph.column_indices[edge]);
			printf(", ");
		}
		printf("\n");
	}
	
}



/******************************************************************************
 * Gridded Graph Construction Routines
 ******************************************************************************/

/**
 * Builds a square 2D grid CSR graph.  Interior nodes have degree 4.  
 * 
 * If src == -1, then it is assigned to the grid-center.  Otherwise it is 
 * verified to be in range of the constructed graph.
 * 
 * Returns 0 on success, 1 on failure.
 */
template<typename IndexType, typename ValueType>
int BuildGrid2dGraph(
	IndexType width,
	IndexType &src,
	CsrGraph<IndexType, ValueType> &csr_graph)
{ 
	if (width < 0) {
		fprintf(stderr, "Invalid width: %d", width);
		return -1;
	}
	
	IndexType interior_nodes = (width - 2) * (width - 2);
	IndexType edge_nodes = (width - 2) * 4;
	IndexType corner_nodes = 4;
	
	csr_graph.edges 			= (interior_nodes * 4) + (edge_nodes * 3) + (corner_nodes * 2);
	csr_graph.nodes 			= width * width;
	csr_graph.row_offsets 		= (IndexType*) malloc(sizeof(IndexType) * (csr_graph.nodes + 1));
	csr_graph.column_indices 	= (IndexType*) malloc(sizeof(IndexType) * csr_graph.edges);
	csr_graph.values 			= (ValueType*) malloc(sizeof(ValueType) * csr_graph.edges);

	IndexType total = 0;
	for (IndexType j = 0; j < width; j++) {
		for (IndexType k = 0; k < width; k++) {
			
			IndexType me = (j * width) + k;
			csr_graph.row_offsets[me] = total; 

			IndexType neighbor = (j * width) + (k - 1);
			if (k - 1 >= 0) {
				csr_graph.column_indices[total] = neighbor; 
				csr_graph.values[me] = 1;
				total++;
			}

			neighbor = (j * width) + (k + 1);
			if (k + 1 < width) {
				csr_graph.column_indices[total] = neighbor; 
				csr_graph.values[me] = 1;
				total++;
			}
			
			neighbor = ((j - 1) * width) + k;
			if (j - 1 >= 0) {
				csr_graph.column_indices[total] = neighbor; 
				csr_graph.values[me] = 1;
				total++;
			}
			
			neighbor = ((j + 1) * width) + k;
			if (j + 1 < width) {
				csr_graph.column_indices[total] = neighbor; 
				csr_graph.values[me] = 1;
				total++;
			}
		}
	}
	csr_graph.row_offsets[csr_graph.nodes] = total; 	// last offset is always num_entries

	// If unspecified, assign default source.  Otherwise verify source range.
	if (src == -1) {
		IndexType half = width / 2;
		src = half * (width + 1);
	} else if ((src < 0 ) || (src > csr_graph.nodes)) {
		fprintf(stderr, "Invalid src: %d", src);
		csr_graph.Free();
		return -1;
	}
	
	return 0;
}


/**
 * Builds a square 3D grid CSR graph.  Interior nodes have degree 6.  
 * 
 * If src == -1, then it is assigned to the grid-center.  Otherwise it is 
 * verified to be in range of the constructed graph.
 * 
 * Returns 0 on success, 1 on failure.
 */
template<typename IndexType, typename ValueType>
int BuildGrid3dGraph(
	IndexType width,
	IndexType &src,
	CsrGraph<IndexType, ValueType> &csr_graph)
{ 
	if (width < 0) {
		fprintf(stderr, "Invalid width: %d", width);
		return -1;
	}
	
	IndexType interior_nodes = (width - 2) * (width - 2) * (width - 2);
	IndexType face_nodes = (width - 2) * (width - 2) * 6;
	IndexType edge_nodes = (width - 2) * 12;
	IndexType corner_nodes = 8;
	
	csr_graph.edges 			= (interior_nodes * 6) + (face_nodes * 5) + (edge_nodes * 4) + (corner_nodes * 3);
	csr_graph.nodes 			= width * width * width;
	csr_graph.row_offsets 		= (IndexType*) malloc(sizeof(IndexType) * (csr_graph.nodes + 1));
	csr_graph.column_indices 	= (IndexType*) malloc(sizeof(IndexType) * csr_graph.edges);
	csr_graph.values 			= (ValueType*) malloc(sizeof(ValueType) * csr_graph.edges);
			
	IndexType total = 0;
	for (IndexType i = 0; i < width; i++) {
		for (IndexType j = 0; j < width; j++) {
			for (IndexType k = 0; k < width; k++) {
				
				IndexType me = (i * width * width) + (j * width) + k;
				csr_graph.row_offsets[me] = total; 

				IndexType neighbor = (i * width * width) + (j * width) + (k - 1);
				if (k - 1 >= 0) {
					csr_graph.column_indices[total] = neighbor; 
					csr_graph.values[me] = 1;
					total++;
				}
				
				neighbor = (i * width * width) + (j * width) + (k + 1);
				if (k + 1 < width) {
					csr_graph.column_indices[total] = neighbor; 
					csr_graph.values[me] = 1;
					total++;
				}
				
				neighbor = (i * width * width) + ((j - 1) * width) + k;
				if (j - 1 >= 0) {
					csr_graph.column_indices[total] = neighbor; 
					csr_graph.values[me] = 1;
					total++;
				}

				neighbor = (i * width * width) + ((j + 1) * width) + k;
				if (j + 1 < width) {
					csr_graph.column_indices[total] = neighbor; 
					csr_graph.values[me] = 1;
					total++;
				}

				neighbor = ((i - 1) * width * width) + (j * width) + k;
				if (i - 1 >= 0) {
					csr_graph.column_indices[total] = neighbor; 
					csr_graph.values[me] = 1;
					total++;
				}

				neighbor = ((i + 1) * width * width) + (j * width) + k;
				if (i + 1 < width) {
					csr_graph.column_indices[total] = neighbor; 
					csr_graph.values[me] = 1;
					total++;
				}
			}
		}
	}
	csr_graph.row_offsets[csr_graph.nodes] = total; 	// last offset is always num_entries

	// If unspecified, assign default source.  Otherwise verify source range.
	if (src == -1) {
		IndexType half = width / 2;
		src = half * ((width * width) + width + 1);
	} else if ((src < 0 ) || (src > csr_graph.nodes)) {
		fprintf(stderr, "Invalid src: %d", src);
		csr_graph.Free();
		return -1;
	}
	
	return 0;
}


/******************************************************************************
 * DIMACS Graph Construction Routines
 ******************************************************************************/

/**
 * Reads a DIMACS graph from an input-stream into a CSR sparse format 
 */
template<typename IndexType, typename ValueType>
int ReadDimacsStream(
	FILE *f_in,
	CsrGraph<IndexType, ValueType> &csr_graph,
	bool undirected)
{
	typedef CooEdgeTuple<IndexType, ValueType> EdgeTupleType;
	
	IndexType nread = 0;
	IndexType nodes = 0;
	IndexType edges = 0;
	IndexType directed_edges = 0;
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
			sscanf(line, "p %s %lld %lld", &problem_type, &ll_nodes, &ll_edges);
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
			if (nread >= edges) {
				fprintf(stderr, "Error parsing DIMACS graph: encountered more than %d edges\n", edges);
				if (coo) free(coo);
				return -1;
			}

			long long ll_row, ll_col, ll_val;
			sscanf(line, "a %d %d %d", &ll_row, &ll_col, &ll_val);

			coo[nread].row = ll_row - 1;	// zero-based array
			coo[nread].col = ll_col - 1;	// zero-based array
			coo[nread].val = ll_val;

			if (undirected) {
				// Reverse edge
				coo[edges + nread].row = coo[nread].col;
				coo[edges + nread].col = coo[nread].row;
				coo[edges + nread].val = coo[nread].val;
			}

			nread++;
			break;
		}
		
		default:
			// read remainder of line
			break;
		}
	}
	
	if (nread != edges) {
		fprintf(stderr, "Error parsing DIMACS graph: only %d/%d edges read\n", nread, edges);
		if (coo) free(coo);
		return -1;
	}
	
	if (coo == NULL) {
		fprintf(stderr, "No graph found\n");
		return -1;
	}
	
	time_t mark1 = time(NULL);
	printf("Done parsing (%ds).\n  Sorting COO format... ", mark1 - mark0);
	fflush(stdout);
	
	// Sort COO by row, then by col
	std::sort(coo, coo + directed_edges, DimacsTupleCompare<IndexType, ValueType>);

	time_t mark2 = time(NULL);
	printf("Done sorting (%ds).\n  Converting to CSR format... ", mark2 - mark1);
	fflush(stdout);
	
	// Convert sorted COO to CSR
	csr_graph.FromCoo(coo, nodes, directed_edges);
	free(coo);

	time_t mark3 = time(NULL);
	printf("Done converting (%ds).\n", mark3 - mark2);

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
template<typename IndexType, typename ValueType>
int BuildDimacsGraph(
	char *dimacs_filename, 
	IndexType &src,
	CsrGraph<IndexType, ValueType> &csr_graph,
	bool undirected)
{ 
	if (dimacs_filename == NULL) {

		// Read from stdin
		printf("Reading from stdin:\n");
		if (ReadDimacsStream(stdin, csr_graph, undirected) != 0) {
			return -1;
		}

	} else {
	
		// Read from file
		FILE *f_in = fopen(dimacs_filename, "r");
		if (f_in) {
			printf("Reading from %s:\n", dimacs_filename);
			if (ReadDimacsStream(f_in, csr_graph, undirected) != 0) {
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

/******************************************************************************
 * Random Graph Construction Routines
 ******************************************************************************/


/**
 * Builds a random CSR graph by adding edges edges to nodes nodes by randomly choosing
 * a pair of nodes for each edge.  There are possibilities of loops and multiple 
 * edges between pairs of nodes.    
 * 
 * If src == -1, it is assigned a random node.  Otherwise it is verified 
 * to be in range of the constructed graph.
 * 
 * Returns 0 on success, 1 on failure.
 */
template<typename IndexType, typename ValueType>
int BuildRandomGraph(
	IndexType nodes,
	IndexType edges,
	IndexType &src,
	CsrGraph<IndexType, ValueType> &csr_graph,
	bool undirected)
{ 
	typedef CooEdgeTuple<IndexType, ValueType> EdgeTupleType;

	if ((nodes < 0) || (edges < 0)) {
		fprintf(stderr, "Invalid graph size: nodes=%d, edges=%d", nodes, edges);
		return -1;
	}

	time_t mark0 = time(NULL);
	printf("  Selecting %llu %s random edges in COO format... ", 
		(unsigned long long) edges, (undirected) ? "undirected" : "directed");
	fflush(stdout);

	// Construct COO graph
	IndexType directed_edges = (undirected) ? edges * 2 : edges;
	EdgeTupleType *coo = (EdgeTupleType*) malloc(sizeof(EdgeTupleType) * directed_edges);
	for (int i = 0; i < edges; i++) {
		coo[i].row = RandomNode(nodes);
		coo[i].col = RandomNode(nodes);
		coo[i].val = 1;
		if (undirected) {
			// Reverse edge
			coo[edges + i].row = coo[i].col;
			coo[edges + i].col = coo[i].row;
			coo[edges + i].val = 1;
		}
	}

	time_t mark1 = time(NULL);
	printf("Done selecting (%ds).\n  Sorting COO format... ", mark1 - mark0);
	fflush(stdout);
	
	// Sort COO by row, then by col
	std::sort(coo, coo + directed_edges, DimacsTupleCompare<IndexType, ValueType>);

	time_t mark2 = time(NULL);
	printf("Done sorting (%ds).\n  Converting to CSR format... ", mark2 - mark1);
	fflush(stdout);

	// Convert sorted COO to CSR
	csr_graph.FromCoo(coo, nodes, directed_edges);
	free(coo);

	time_t mark3 = time(NULL);
	printf("Done converting (%ds).\n", mark3 - mark2);
	
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

/**
 * A random graph where each node has a guaranteed degree of random neighbors.
 * Does not meet definition of random-regular: loops, and multi-edges are 
 * possible, and in-degree is not guaranteed to be the same as out degree.   
 */
template<typename IndexType, typename ValueType>
int BuildRandomRegularishGraph(
	IndexType nodes,
	int degree,
	IndexType &src,
	CsrGraph<IndexType, ValueType> &csr_graph)
{
	IndexType edges = nodes * degree;
	
	csr_graph.edges 			= edges;
	csr_graph.nodes 			= nodes;
	csr_graph.row_offsets 		= (IndexType*) malloc(sizeof(IndexType) * (csr_graph.nodes + 1));
	csr_graph.column_indices 	= (IndexType*) malloc(sizeof(IndexType) * csr_graph.edges);
	csr_graph.values 			= (ValueType*) malloc(sizeof(ValueType) * csr_graph.edges);

	IndexType total = 0;
    for (IndexType node = 0; node < nodes; node++) {
    	
    	csr_graph.row_offsets[node] = total;
    	
    	for (IndexType edge = 0; edge < degree; edge++) {
    		
    		IndexType neighbor = RandomNode(csr_graph.nodes);
    		csr_graph.column_indices[total] = neighbor;
    		csr_graph.values[node] = 1;
    		
    		total++;
    	}
    }
    
    csr_graph.row_offsets[nodes] = total; 	// last offset is always num_entries

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

