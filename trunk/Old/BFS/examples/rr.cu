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

#include <string>
#include <sstream>
#include <iostream>

#include <fstream>
#include <deque>
#include <algorithm>

#include <b40c_util.h>					// Misc. utils (random-number gen, I/O, etc.)
#include <test_utils.cu>


/******************************************************************************
 * Random Graph Construction Routines
 ******************************************************************************/

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

