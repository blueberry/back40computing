/******************************************************************************
 * 
 * Copyright 2010-2012 Duane Merrill
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
 * CTA-processing abstraction for reduction kernels
 ******************************************************************************/

#pragma once

#include <b40c/util/io/modified_load.cuh>
#include <b40c/util/io/modified_store.cuh>
#include <b40c/util/io/load_tile.cuh>

#include <b40c/util/reduction/serial_reduce.cuh>
#include <b40c/util/reduction/tree_reduce.cuh>
#include <b40c/util/reduction/cta_reduction.cuh>

namespace b40c {
namespace reduction {



/**
 * Templated texture reference for global input
 */
template <typename TexVec>
struct InputTex
{
	static texture<TexVec, cudaTextureType1D, cudaReadModeElementType> d_in_ref;
};
template <typename TexVec>
typename texture<TexVec, cudaTextureType1D, cudaReadModeElementType> InputTex<TexVec>::d_in_ref;



/**
 * Reduction CTA abstraction
 */
template <typename KernelPolicy>
struct Cta
{
	//---------------------------------------------------------------------
	// Typedefs
	//---------------------------------------------------------------------

	typedef typename KernelPolicy::T 			T;					// Data type to reduce
	typedef typename KernelPolicy::TexVec		TexVec;				// Texture vector type
	typedef typename KernelPolicy::TexRef		TexRef;				// Texture reference type
	typedef typename KernelPolicy::SizeT 		SizeT;				// Counting type
	typedef typename KernelPolicy::ReductionOp	ReductionOp;		// Reduction operator type

	// Tile reader type
	typedef util::io::TileReader<
		KernelPolicy::THREADS,
		KernelPolicy::READ_MODIFIER> TileReader;

	// CTA reduction type
	typedef util::reduction::CtaReduction<
		KernelPolicy::THREADS,
		T> CtaReduction;

	// Shared memory layout
	struct SmemStorage
	{
		typename CtaReduction::SmemStorage reduction_storage;
	};


	//---------------------------------------------------------------------
	// Constants
	//---------------------------------------------------------------------

	enum {
		CUDA_ARCH				= __B40C_CUDA_ARCH__,
		THREAD_OCCUPANCY		= B40C_SM_THREADS(CUDA_ARCH) >> KernelPolicy::LOG_THREADS,
		SMEM_OCCUPANCY			= B40C_SMEM_BYTES(CUDA_ARCH) / sizeof(SmemStorage),
		MAX_CTA_OCCUPANCY  		= B40C_MIN(B40C_SM_CTAS(CUDA_ARCH), B40C_MIN(THREAD_OCCUPANCY, SMEM_OCCUPANCY)),
		VALID					= (MAX_CTA_OCCUPANCY > 0),
	};


	//---------------------------------------------------------------------
	// Thread fields
	//---------------------------------------------------------------------

	T* 					d_in;				// Input device pointer
	T* 					d_out;				// Output device pointer
	T 					accumulator;		// The value we will accumulate (in each thread)
	ReductionOp			reduction_op;		// Reduction operator
	CtaReduction 		reducer;			// Collective reducer
	TexRef 				d_in_ref;			// Input texture reference


	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------

	/**
	 * Constructor
	 */
	__device__ __forceinline__ Cta(
		SmemStorage &smem_storage,
		T* d_in,
		T* d_out,
		ReductionOp reduction_op) :
			// Initializers
			d_in(d_in),
			d_out(d_out),
			reduction_op(reduction_op),
			reducer(smem_storage.reduction_storage),
			d_in_ref(InputTex<TexVec>::d_in_ref)
	{
	}


	/**
	 * Process a single, full tile
	 *
	 * Each thread reduces only the strided values it loads.
	 */
	__device__ __forceinline__ void ProcessFullTile(
		SizeT cta_offset,
		bool first_tile)
	{
		// Tile of elements
		T data[KernelPolicy::LOADS_PER_TILE][KernelPolicy::LOAD_VEC_SIZE];

		// Load tile
		TileReader::LoadUnguarded(data, d_in_ref, d_in, cta_offset);

		// Reduce the data we loaded for this tile
		T tile_partial = util::reduction::SerialReduce(data, reduction_op);

		// Reduce into accumulator
		accumulator = (first_tile) ?
			tile_partial :
			reduction_op(accumulator, tile_partial);
	}


	/**
	 * Process a single, partial tile
	 *
	 * Each thread reduces only the strided values it loads.
	 */
	__device__ __forceinline__ void ProcessPartialTile(
		SizeT cta_offset,
		SizeT out_of_bounds,
		bool first_tile)
	{
		cta_offset += threadIdx.x;

		// First tile processed loads into the accumulator directly
		if ((first_tile) && (cta_offset < out_of_bounds)) {

			util::io::ModifiedLoad<KernelPolicy::READ_MODIFIER>::Ld(accumulator, d_in + cta_offset);
			cta_offset += KernelPolicy::THREADS;
		}

		// Process loads singly
		while (cta_offset < out_of_bounds) {

			T datum;
			util::io::ModifiedLoad<KernelPolicy::READ_MODIFIER>::Ld(datum, d_in + cta_offset);
			accumulator = reduction_op(accumulator, datum);
			cta_offset += KernelPolicy::THREADS;
		}

	}


	/**
	 * Guarded collective reduction across all threads, stores final reduction
	 * to output. Used to collectively reduce each thread's aggregate after striding through
	 * the input.
	 *
	 * Only threads with ranks less than num_elements are assumed to have valid
	 * accumulator data.
	 */
	__device__ __forceinline__ void OutputToSpine(int num_elements)
	{
		// Collective CTA reduction of thread accumulators
		accumulator = reducer.Reduce(
			accumulator,
			reduction_op,
			num_elements);

		// Write output
		if (threadIdx.x == 0) {
			util::io::ModifiedStore<KernelPolicy::WRITE_MODIFIER>::St(
				accumulator, d_out + blockIdx.x);
		}
	}


	/**
	 * Process work range of tiles
	 */
	__device__ __forceinline__ void ProcessWorkRange(
		util::CtaWorkLimits<SizeT> &work_limits)
	{
		// Make sure we get a local copy of the cta's offset (work_limits may be in smem)
		SizeT cta_offset = work_limits.offset;

		if (cta_offset < work_limits.guarded_offset) {

			// Process at least one full tile of tile_elements
			ProcessFullTile(cta_offset, true);
			cta_offset += KernelPolicy::TILE_ELEMENTS;

			// Process more full tiles (not first tile)
			while (cta_offset < work_limits.guarded_offset) {
				ProcessFullTile(cta_offset, false);
				cta_offset += KernelPolicy::TILE_ELEMENTS;
			}

			// Clean up last partial tile with guarded-io (not first tile)
			if (work_limits.guarded_elements) {
				ProcessPartialTile(
					cta_offset,
					work_limits.out_of_bounds,
					false);
			}

			// Collectively reduce accumulator from each thread into output
			// destination (all thread have valid reduction partials)
			OutputToSpine(KernelPolicy::TILE_ELEMENTS);

		} else {

			// Clean up last partial tile with guarded-io (first tile)
			ProcessPartialTile(
				cta_offset,
				work_limits.out_of_bounds,
				true);

			// Collectively reduce accumulator from each thread into output
			// destination (not every thread may have a valid reduction partial)
			OutputToSpine(work_limits.elements);
		}
	}

};


} // namespace reduction
} // namespace b40c

