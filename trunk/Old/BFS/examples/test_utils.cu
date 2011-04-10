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

#include <algorithm>

#include <test/b40c_test_util.h>					// Misc. utils (random-number gen, I/O, etc.)


/******************************************************************************
 * General utility routines
 ******************************************************************************/

/**
 * Returns a random node-ID in the range of [0, num_nodes) 
 */
template<typename SizeT>
SizeT RandomNode(SizeT num_nodes) {
	SizeT node_id;
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
template<typename VertexId, typename Value>
struct CooEdgeTuple {
	VertexId row;
	VertexId col;
	Value val;
};

/**
 * Comparator for sorting COO sparse format edges
 */
template<typename VertexId, typename Value>
bool DimacsTupleCompare (
	CooEdgeTuple<VertexId, Value> elem1,
	CooEdgeTuple<VertexId, Value> elem2)
{
	if (elem1.row < elem2.row) {
		// Sort edges by source node (to make rows)
		return true;
/*
	} else if ((elem1.row == elem2.row) && (elem1.col < elem2.col)) {
		// Sort edgelists as well for coherence
		return true;
*/
	} 
	
	return false;
}


/******************************************************************************
 * Simple CSR sparse graph datastructure
 ******************************************************************************/

/**
 * CSR sparse format graph
 */
template<typename VertexId, typename Value, typename SizeT>
struct CsrGraph
{
	SizeT nodes;
	SizeT edges;
	
	SizeT 		*row_offsets;
	VertexId	*column_indices;
	Value		*values;
	
	/**
	 * Empty constructor
	 */
	CsrGraph()
	{
		nodes = 0;
		edges = 0;
		row_offsets = NULL;
		column_indices = NULL;
		values = NULL;
	}

	/**
	 * Build CSR graph from sorted COO graph
	 */
	template <bool LOAD_VALUES>
	void FromCoo(
		CooEdgeTuple<VertexId, Value> *coo,
		SizeT coo_nodes,
		SizeT coo_edges)
	{
		printf("  Converting to CSR format... ");
		time_t mark1 = time(NULL);
		fflush(stdout);
		
		nodes 				= coo_nodes;
		edges 				= coo_edges;
		row_offsets 		= (SizeT*) malloc(sizeof(SizeT) * (nodes + 1));
		column_indices 		= (VertexId*) malloc(sizeof(VertexId) * edges);
		values 				= (LOAD_VALUES) ? (Value*) malloc(sizeof(Value) * edges) : NULL;
		
		// Sort COO by row, then by col
		std::stable_sort(coo, coo + coo_edges, DimacsTupleCompare<VertexId, Value>);

		VertexId prev_row = -1;
		for (SizeT edge = 0; edge < edges; edge++) {
			
			VertexId current_row = coo[edge].row;
			
			// Fill in rows up to and including the current row
			for (VertexId row = prev_row + 1; row <= current_row; row++) {
				row_offsets[row] = edge;
			}
			prev_row = current_row;
			
			column_indices[edge] = coo[edge].col;
			if (LOAD_VALUES) {
				values[edge] = coo[edge].val;
			}
		}

		// Fill out any trailing edgeless nodes (and the end-of-list element)
		for (VertexId row = prev_row + 1; row <= nodes; row++) {
			row_offsets[row] = edges;
		}

		time_t mark2 = time(NULL);
		printf("Done converting (%ds).\n", (int) (mark2 - mark1));
		fflush(stdout);
	}

	/**
	 * Print log-histogram
	 */
	void PrintHistogram()
	{
		fflush(stdout);

		// Initialize
		int log_counts[32];
		for (int i = 0; i < 32; i++) {
			log_counts[i] = 0;
		}

		// Scan
		int max_log_length = -1;
		for (VertexId i = 0; i < nodes; i++) {

			SizeT length = row_offsets[i + 1] - row_offsets[i];

			int log_length = -1;
			while (length > 0) {
				length >>= 1;
				log_length++;
			}
			if (log_length > max_log_length) {
				max_log_length = log_length;
			}

			log_counts[log_length + 1]++;
		}
		printf("\nDegree Histogram (%lld vertices, %lld directed edges):\n", (long long) nodes, (long long) edges);
		for (int i = -1; i < max_log_length + 1; i++) {
			printf("\tDegree 2^%i: %d (%.2f%%)\n", i, log_counts[i + 1], (float) log_counts[i + 1] * 100.0 / nodes);
		}
		printf("\n");
		fflush(stdout);
	}

	/**
	 * Display CSR graph to console
	 */
	void DisplayGraph()
	{
		printf("Input Graph:\n");
		for (VertexId node = 0; node < nodes; node++) {
			PrintValue(node);
			printf(": ");
			for (SizeT edge = row_offsets[node]; edge < row_offsets[node + 1]; edge++) {
				PrintValue(column_indices[edge]);
				printf(", ");
			}
			printf("\n");
		}

	}

	void Free()
	{
		if (row_offsets) { free(row_offsets); row_offsets = NULL; }
		if (column_indices) { free(column_indices); column_indices = NULL; }
		if (values) { free (values); values = NULL; }
		nodes = 0;
		edges = 0;
	}
	
	~CsrGraph()
	{
		Free();
	}
};


