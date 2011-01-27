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
 * AUTHORS' REQUEST: 
 * 
 * 		If you use|reference|benchmark this code, please cite our Technical 
 * 		Report (http://www.cs.virginia.edu/~dgm4d/papers/RadixSortTR.pdf):
 * 
 *		@TechReport{ Merrill:Sorting:2010,
 *        	author = "Duane Merrill and Andrew Grimshaw",
 *        	title = "Revisiting Sorting for GPGPU Stream Architectures",
 *        	year = "2010",
 *        	institution = "University of Virginia, Department of Computer Science",
 *        	address = "Charlottesville, VA, USA",
 *        	number = "CS2010-03"
 *		}
 * 
 * For more information, see our Google Code project site: 
 * http://code.google.com/p/back40computing/
 * 
 * Thanks!
 * 
 ******************************************************************************/


/******************************************************************************
 * Top-level histogram/spine scanning kernel. The second kernel in a 
 * radix-sorting digit-place pass. 
 ******************************************************************************/

#pragma once

#include "radixsort_kernel_common.cu"

namespace b40c {

namespace lsb_radix_sort {


/******************************************************************************
 * Granularity Configuration
 ******************************************************************************/

/**
 * Spine-scan granularity configuration.  This C++ type encapsulates our 
 * kernel-tuning parameters (they are reflected via the static fields).
 *  
 * The kernels are specialized for problem-type, SM-version, etc. by declaring 
 * them with different performance-tuned parameterizations of this type.  By 
 * incorporating this type into the kernel code itself, we guide the compiler in 
 * expanding/unrolling the kernel code for specific architectures and problem 
 * types.    
 */
template <
	typename _SpineType,
	int _CTA_OCCUPANCY,
	int _LOG_THREADS,
	int _LOG_LOAD_VEC_SIZE,
	int _LOG_LOADS_PER_TILE,
	int _LOG_RAKING_THREADS,
	CacheModifier _CACHE_MODIFIER>

struct SpineScanConfig
{
	typedef _SpineType							SpineType;
	static const int CTA_OCCUPANCY  			= _CTA_OCCUPANCY;
	static const int LOG_THREADS 				= _LOG_THREADS;
	static const int LOG_LOAD_VEC_SIZE  		= _LOG_LOAD_VEC_SIZE;
	static const int LOG_LOADS_PER_TILE 		= _LOG_LOADS_PER_TILE;
	static const int LOG_RAKING_THREADS			= _LOG_RAKING_THREADS;
	static const CacheModifier CACHE_MODIFIER 	= _CACHE_MODIFIER;
};



/******************************************************************************
 * Kernel Configuration  
 ******************************************************************************/

/**
 * A detailed upsweep configuration type that specializes kernel code for a specific 
 * sorting pass.  It encapsulates granularity details derived from the inherited 
 * UpsweepConfigType 
 */
template <typename SpineScanConfigType>
struct SpineScanKernelConfig : SpineScanConfigType
{
	static const int THREADS							= 1 << SpineScanConfigType::LOG_THREADS;
	
	static const int LOG_TILE_ELEMENTS					= SpineScanConfigType::LOG_THREADS + 
															SpineScanConfigType::LOG_LOADS_PER_TILE +
															SpineScanConfigType::LOG_LOAD_VEC_SIZE;
	static const int TILE_ELEMENTS						= 1 << LOG_TILE_ELEMENTS;
	
	// We reduce/scan the elements of a loaded vector in registers, and then place that  
	// partial reduction into smem rows for further reduction/scanning
	
	static const int LOG_SMEM_PARTIALS					= SpineScanConfigType::LOG_THREADS + SpineScanConfigType::LOG_LOADS_PER_TILE;				
	static const int SMEM_PARTIALS			 			= 1 << LOG_SMEM_PARTIALS;
	
	static const int LOG_SMEM_PARTIALS_PER_SEG 			= LOG_SMEM_PARTIALS - SpineScanConfigType::LOG_RAKING_THREADS;	
	static const int SMEM_PARTIALS_PER_SEG 				= 1 << LOG_SMEM_PARTIALS_PER_SEG;

	static const int LOG_SMEM_PARTIALS_PER_BANK_ARRAY	= B40C_LOG_MEM_BANKS(__B40C_CUDA_ARCH__) + 
															B40C_LOG_BANK_STRIDE_BYTES(__B40C_CUDA_ARCH__) - 
															LogBytes<typename SpineScanConfigType::SpineType>::LOG_BYTES;

	static const int PADDING_PARTIALS					= 1 << B40C_MAX(0, B40C_LOG_BANK_STRIDE_BYTES(__B40C_CUDA_ARCH__) - LogBytes<typename SpineScanConfigType::SpineType>::LOG_BYTES); 

	static const int LOG_SMEM_PARTIALS_PER_ROW			= B40C_MAX(LOG_SMEM_PARTIALS_PER_SEG, LOG_SMEM_PARTIALS_PER_BANK_ARRAY);
	static const int SMEM_PARTIALS_PER_ROW				= 1 << LOG_SMEM_PARTIALS_PER_ROW;

	static const int PADDED_SMEM_PARTIALS_PER_ROW		= SMEM_PARTIALS_PER_ROW + SMEM_PARTIALS_PER_ROW;
	
	static const int LOG_SEGS_PER_ROW 					= LOG_SMEM_PARTIALS_PER_ROW - LOG_SMEM_PARTIALS_PER_SEG;	
	static const int SEGS_PER_ROW						= 1 << LOG_SEGS_PER_ROW;

	static const int LOG_SMEM_ROWS						= LOG_SMEM_PARTIALS - LOG_SMEM_PARTIALS_PER_ROW;
	static const int SMEM_ROWS 							= 1 << LOG_SMEM_ROWS;
	
	static const int SHARED_BYTES						= SMEM_ROWS * PADDED_SMEM_PARTIALS_PER_ROW * sizeof(typename SpineScanConfigType::SpineType);
	static const int SHARED_INT4S						= (SHARED_BYTES + sizeof(int4) - 1) / sizeof(int4);
	
};
	
	
	


/******************************************************************************
 * Spine-scan kernel subroutines
 ******************************************************************************/



/**
 * Scans a cycle of RADIXSORT_TILE_ELEMENTS elements
 */
/*
template<CacheModifier CACHE_MODIFIER, int SMEM_PARTIALS_PER_SEG>
__device__ __forceinline__ void SrtsScanTile(
	int *smem_offset,
	int *smem_segment,
	int warpscan[2][B40C_WARP_THREADS],
	int4 *in, 
	int4 *out,
	int &carry)
{
	int4 datum; 

	// read input data
	ModifiedLoad<int4, CACHE_MODIFIER>::Ld(datum, in, threadIdx.x);

	smem_offset[0] = datum.x + datum.y + datum.z + datum.w;

	__syncthreads();

	if (threadIdx.x < B40C_WARP_THREADS) {

		int partial_reduction = SerialReduce<int, SMEM_PARTIALS_PER_SEG>(smem_segment);

		int seed = WarpScan<B40C_WARP_THREADS, false>(warpscan, partial_reduction, 0);
		seed += carry;		
		
		SerialScan<int, SMEM_PARTIALS_PER_SEG>(smem_segment, seed);

		carry += warpscan[1][B40C_WARP_THREADS - 1];	
	}

	__syncthreads();

	int part0 = smem_offset[0];
	int part1;

	part1 = datum.x + part0;
	datum.x = part0;
	part0 = part1 + datum.y;
	datum.y = part1;

	part1 = datum.z + part0;
	datum.z = part0;
	part0 = part1 + datum.w;
	datum.w = part1;
	
	out[threadIdx.x] = datum;
}
*/


/**
 * Host stub to calm the linker for arch-specializations that we didn't 
 * end up compiling PTX for.
 */
template <typename KernelConfig> 
__host__ void __wrapper__device_stub_LsbSpineScanKernel(
	typename KernelConfig::SpineType *&, 
	int &) {}


/**
 * Kernel entry point
 */
template <typename KernelConfig>
__launch_bounds__ (KernelConfig::THREADS, KernelConfig::CTA_OCCUPANCY)
__global__ void LsbSpineScanKernel(
	typename KernelConfig::SpineType *d_spine,
	int spine_elements)
{
	typedef typename KernelConfig::SpineType SpineType;
	
/*	
	// Shared memory pool
	__shared__ int4 smem_pool[KernelConfig::SHARED_INT4S];
	
	
	__shared__ int smem[SMEM_ROWS][SMEM_PARTIALS_PER_ROW + 1];
	__shared__ int warpscan[2][B40C_WARP_THREADS];

	int *smem_segment = 0;
	int carry = 0;

	int row = threadIdx.x >> LOG_SMEM_PARTIALS_PER_ROW;		
	int col = threadIdx.x & (SMEM_PARTIALS_PER_ROW - 1);			
	int *smem_offset = &smem[row][col];

	if (blockIdx.x > 0) {
		return;
	}
	
	if (threadIdx.x < B40C_WARP_THREADS) {
		
		// two segs per row, odd segs are offset by 8
		row = threadIdx.x >> LOG_SEGS_PER_ROW;
		col = (threadIdx.x & (SEGS_PER_ROW - 1)) << LOG_SMEM_PARTIALS_PER_SEG;
		smem_segment = &smem[row][col];
	
		if (threadIdx.x < B40C_WARP_THREADS) {
			warpscan[0][threadIdx.x] = 0;
		}
	}

	// scan the spine in blocks of cycle_elements
	int block_offset = 0;
	while (block_offset < normal_block_elements) {
		
		SrtsScanTile<NONE, SMEM_PARTIALS_PER_SEG>(	
			smem_offset, 
			smem_segment, 
			warpscan,
			reinterpret_cast<int4 *>(&d_ispine[block_offset]), 
			reinterpret_cast<int4 *>(&d_ospine[block_offset]), 
			carry);

		block_offset += B40C_RADIXSORT_SPINE_TILE_ELEMENTS;
	}
*/	
} 


} // namespace lsb_radix_sort

} // namespace b40c

