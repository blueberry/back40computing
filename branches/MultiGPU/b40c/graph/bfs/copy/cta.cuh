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
 * Thanks!
 * 
 ******************************************************************************/

/******************************************************************************
 * Tile-processing functionality for BFS copy kernels
 ******************************************************************************/

#pragma once

#include <b40c/util/io/modified_load.cuh>
#include <b40c/util/io/modified_store.cuh>
#include <b40c/util/io/load_tile.cuh>
#include <b40c/util/io/store_tile.cuh>

namespace b40c {
namespace graph {
namespace bfs {
namespace copy {


/**
 * Derivation of KernelPolicy that encapsulates tile-processing routines
 */
template <typename KernelPolicy>
struct Cta
{
	//---------------------------------------------------------------------
	// Typedefs
	//---------------------------------------------------------------------

	typedef typename KernelPolicy::VertexId 		VertexId;
	typedef typename KernelPolicy::SizeT 			SizeT;

	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Input and output device pointers
	VertexId* d_in;
	VertexId* d_out;
	VertexId* d_parent_in;
	VertexId* d_parent_out;

	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------

	/**
	 * Constructor
	 */
	__device__ __forceinline__ Cta(
		VertexId 				*d_in,
		VertexId 				*d_out,
		VertexId 				*d_parent_in,
		VertexId 				*d_parent_out) :
			d_in(d_in),
			d_out(d_out),
			d_parent_in(d_parent_in),
			d_parent_out(d_parent_out)
	{}


	/**
	 * Process a single tile
	 *
	 * Each thread copies only the strided values it loads.
	 */
	__device__ __forceinline__ void ProcessTile(
		SizeT cta_offset,
		SizeT guarded_elements = KernelPolicy::TILE_ELEMENTS)
	{
		// Tile of elements
		VertexId data[KernelPolicy::LOADS_PER_TILE][KernelPolicy::LOAD_VEC_SIZE];

		// Load tile of vertices
		util::io::LoadTile<
			KernelPolicy::LOG_LOADS_PER_TILE,
			KernelPolicy::LOG_LOAD_VEC_SIZE,
			KernelPolicy::THREADS,
			KernelPolicy::READ_MODIFIER>::LoadValid(
				data, d_in + cta_offset, guarded_elements);

		if (KernelPolicy::LOADS_PER_TILE > 1) __syncthreads();

		// Store tile of vertices
		util::io::StoreTile<
			KernelPolicy::LOG_LOADS_PER_TILE,
			KernelPolicy::LOG_LOAD_VEC_SIZE,
			KernelPolicy::THREADS,
			KernelPolicy::WRITE_MODIFIER>::Store(
				data, d_out + cta_offset, guarded_elements);

		if (KernelPolicy::MARK_PARENTS) {

			// Load tile of parents
			util::io::LoadTile<
				KernelPolicy::LOG_LOADS_PER_TILE,
				KernelPolicy::LOG_LOAD_VEC_SIZE,
				KernelPolicy::THREADS,
				KernelPolicy::READ_MODIFIER>::LoadValid(
					data, d_parent_in + cta_offset, guarded_elements);

			if (KernelPolicy::LOADS_PER_TILE > 1) __syncthreads();

			// Store tile of parents
			util::io::StoreTile<
				KernelPolicy::LOG_LOADS_PER_TILE,
				KernelPolicy::LOG_LOAD_VEC_SIZE,
				KernelPolicy::THREADS,
				KernelPolicy::WRITE_MODIFIER>::Store(
					data, d_parent_out + cta_offset, guarded_elements);
		}
	}
};

} // namespace copy
} // namespace bfs
} // namespace graph
} // namespace b40c

