/******************************************************************************
 * 
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
 * 
 ******************************************************************************/

/******************************************************************************
 * Common types and kernel routines for B40C BFS kernels  
 ******************************************************************************/

#pragma once

namespace b40c {
namespace bfs {


/******************************************************************************
 * BFS Algorithm and Granularity Configuration 
 ******************************************************************************/

/**
 * Enumeration of parallel BFS algorithm strategies
 */
enum BfsStrategy {
	
	/**
	 * Contract-then-Expand
	 * 
	 * At each iteration, the frontier queue is comprised of "unvisited-edges", 
	 * i.e. the concatenation of the adjacency lists belonging to the nodes 
	 * that were discovered from the previous iteration. (These unvisited-edges 
	 * are simply the incident node IDs.)  The algorithm discards the edges 
	 * leading to previously-visited nodes, and then expands the edge-lists of 
	 * the remaining (newly-discovered) nodes into the frontier queue for the 
	 * next iteration. As the frontier is streamed through the SMs for each BFS 
	 * iteration, the kernel operates by:
	 * 
	 *  (1) Streaming in tiles of the frontier queue and contracting the unvisited-
	 *      edges by:
	 *        (i)  Removing incident nodes that were discovered by previous 
	 *             iterations
	 *        (ii) A heuristic for removing duplicate incident nodes††. 
	 *         
	 *      The remaining incident nodes are marked as being discovered at this 
	 *      iteration. 
	 *       
	 *  (2) Expanding the newly discovered nodes into their adjacency lists and
	 *      enqueuing these adjacency lists into the outgoing frontier for 
	 *      processing by the next iteration. 
	 */
	CONTRACT_EXPAND,
	
	/**
	 * Expand-then-Contract 
	 * 
	 * At each iteration, the frontier queue is comprised of "discovered nodes" 
	 * from the previous iteration.  The algorithm expands these nodes into 
	 * their edge-lists.  The edges leading to previously-visited nodes are 
	 * discarded.  Then the remaining (newly-discovered) nodes are enqueued 
	 * into the frontier queue for the next iteration. As the frontier is 
	 * streamed through the SMs for each BFS iteration, the kernel operates by:
	 * 
	 *  (1) Streaming in tiles of the frontier queue and expanding those nodes 
	 *      into their adjacency lists into shared-memory scratch space.
	 *         
	 *  (2) Contracting these "unvisited edges" in shared-memory scratch by:
	 *        (i)  Removing incident nodes that were discovered by previous 
	 *             iterations
	 *        (ii) A heuristic for removing duplicate incident nodes††. 
	 *         
	 *      The remaining incident nodes are marked as being discovered at this 
	 *      iteration, and enqueued into the outgoing frontier for processing by 
	 *      the next iteration.
	 */
	EXPAND_CONTRACT,
	
	/**
	 * Footnotes:
	 * 
	 *   †† Frontier duplicates exist when a node is neighbor to multiple nodes 
	 *      discovered by the previous iteration.  Although the operation of the 
	 *      algorithm is correct regardless of the number of times a node is 
	 *      discovered within a given iteration, duplicate-removal can drastically 
	 *      reduce the overall work performed.  When the same node is discovered
	 *      concurrently within a given iteration, its entire adjacency list will 
	 *      be duplicated in the next iteration's frontier.  Duplicate-removal is 
	 *      particularly effective for lattice-like graphs: nodes are often 
	 *      discoverable at a given iteration via multiple indicent edges.  
	 */

	CULL,
	NONE
};


} // namespace bfs
} // b40c namespace


