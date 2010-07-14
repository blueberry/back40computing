/**
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
 * 
 * 
 * 
 * AUTHORS' REQUEST: 
 * 
 * 		If you use|reference|benchmark this code, please cite our Technical 
 * 		Report (http://www.cs.virginia.edu/~dgm4d/papers/RadixSortTR.pdf):
 * 
 * 		Duane Merrill and Andrew Grimshaw, "Revisiting Sorting for GPGPU 
 * 		Stream Architectures," University of Virginia, Department of 
 * 		Computer Science, Charlottesville, VA, USA, Technical Report 
 * 		CS2010-03, 2010.
 * 
 * For more information, see our Google Code project site: 
 * http://code.google.com/p/back40computing/
 * 
 * Thanks!
 */


#ifndef _SRTS_RADIX_SORT_SPINE_KERNEL_H_
#define _SRTS_RADIX_SORT_SPINE_KERNEL_H_

#include <kernel/srts_radixsort_kernel_common.cu>


//------------------------------------------------------------------------------
// SRTS Spine Configuration
//------------------------------------------------------------------------------

// 128 threads
#define SRTS_LOG_SPINE_THREADS						7							
#define SRTS_SPINE_THREADS							(1 << SRTS_LOG_SPINE_THREADS)	

// 512 elements
#define SRTS_LOG_SPINE_CYCLE_ELEMENTS				9
#define SRTS_SPINE_CYCLE_ELEMENTS					(1 << SRTS_LOG_SPINE_CYCLE_ELEMENTS)



//------------------------------------------------------------------------------
// SrtsScanSpine
//------------------------------------------------------------------------------

template<
	unsigned int SMEM_ROWS,
	unsigned int RAKING_THREADS,
	unsigned int PARTIALS_PER_ROW,
	unsigned int PARTIALS_PER_SEG>
__device__ inline void SrtsScan512(
	unsigned int smem[SMEM_ROWS][PARTIALS_PER_ROW + 1],
	unsigned int *smem_offset,
	unsigned int *smem_segment,
	unsigned int warpscan[2][WARP_THREADS],
	uint4 *in, 
	uint4 *out,
	unsigned int &carry)
{
	uint4 datum; 

	// read input data
	datum = in[threadIdx.x];

	smem_offset[0] = datum.x + datum.y + datum.z + datum.w;

	__syncthreads();

	if (threadIdx.x < WARP_THREADS) {

		unsigned int partial_reduction = SerialReduce<PARTIALS_PER_SEG>(smem_segment);

		unsigned int seed = WarpScan<WARP_THREADS, false>(warpscan, partial_reduction, 0);
		seed += carry;		
		carry += warpscan[1][WARP_THREADS - 1];	
		
		SerialScan<PARTIALS_PER_SEG>(smem_segment, seed);
	}

	__syncthreads();

	unsigned int part0 = smem_offset[0];
	unsigned int part1;

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


__global__ void SrtsScanSpine(
	unsigned int *d_ispine,
	unsigned int *d_ospine,
	unsigned int normal_block_elements)
{
	const unsigned int LOG_RAKING_THREADS 		= LOG_WARP_THREADS;				
	const unsigned int RAKING_THREADS 			= 1 << LOG_RAKING_THREADS;		
	
	const unsigned int LOG_PARTIALS				= SRTS_LOG_THREADS;				
	const unsigned int PARTIALS			 		= 1 << LOG_PARTIALS;
	
	const unsigned int LOG_PARTIALS_PER_SEG 	= LOG_PARTIALS - LOG_RAKING_THREADS;	
	const unsigned int PARTIALS_PER_SEG 		= 1 << LOG_PARTIALS_PER_SEG;

	const unsigned int LOG_PARTIALS_PER_ROW		= (LOG_PARTIALS_PER_SEG < LOG_MEM_BANKS(__CUDA_ARCH__)) ? LOG_MEM_BANKS(__CUDA_ARCH__) : LOG_PARTIALS_PER_SEG;		// floor of 32 elts per row
	const unsigned int PARTIALS_PER_ROW			= 1 << LOG_PARTIALS_PER_ROW;
	
	const unsigned int LOG_SEGS_PER_ROW 		= LOG_PARTIALS_PER_ROW - LOG_PARTIALS_PER_SEG;	
	const unsigned int SEGS_PER_ROW				= 1 << LOG_SEGS_PER_ROW;

	const unsigned int SMEM_ROWS 				= PARTIALS / PARTIALS_PER_ROW;
	
	__shared__ unsigned int smem[SMEM_ROWS][PARTIALS_PER_ROW + 1];
	__shared__ unsigned int warpscan[2][WARP_THREADS];

	unsigned int *smem_segment;
	unsigned int carry;

	unsigned int row = threadIdx.x >> LOG_PARTIALS_PER_ROW;		
	unsigned int col = threadIdx.x & (PARTIALS_PER_ROW - 1);			
	unsigned int *smem_offset = &smem[row][col];

	if (blockIdx.x > 0) {
		return;
	}
	
	if (threadIdx.x < RAKING_THREADS) {
		
		// two segs per row, odd segs are offset by 8
		row = threadIdx.x >> LOG_SEGS_PER_ROW;
		col = (threadIdx.x & (SEGS_PER_ROW - 1)) << LOG_PARTIALS_PER_SEG;
		smem_segment = &smem[row][col];
	
		if (threadIdx.x < WARP_THREADS) {
			carry = 0;
			warpscan[0][threadIdx.x] = 0;
		}
	}

	// scan the spine in blocks of cycle_elements
	unsigned int block_offset = 0;
	while (block_offset < normal_block_elements) {
		
		SrtsScan512<SMEM_ROWS, RAKING_THREADS, PARTIALS_PER_ROW, PARTIALS_PER_SEG>(	
			smem, smem_offset, smem_segment, warpscan,
			(uint4 *) &d_ispine[block_offset], 
			(uint4 *) &d_ospine[block_offset], 
			carry);

		block_offset += SRTS_SPINE_CYCLE_ELEMENTS;
	}
} 


#endif



