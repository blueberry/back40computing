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
 * Radix sorting upsweep digit-reduction/counting kernel.  The first kernel in 
 * a radix-sorting digit-place pass.
 ******************************************************************************/

#pragma once

#include "radixsort_kernel_common.cu"

namespace b40c {
namespace lsb_radix_sort {
namespace upsweep {


/******************************************************************************
 * Granularity Configuration
 ******************************************************************************/

/**
 * Upsweep granularity configuration.  This C++ type encapsulates our 
 * kernel-tuning parameters (they are reflected via the static fields).
 *  
 * The kernels are specialized for problem-type, SM-version, etc. by declaring 
 * them with different performance-tuned parameterizations of this type.  By 
 * incorporating this type into the kernel code itself, we guide the compiler in 
 * expanding/unrolling the kernel code for specific architectures and problem 
 * types.    
 */
template <
	typename _KeyType,
	typename _IndexType,
	int _RADIX_BITS,
	int _LOG_SUBTILE_ELEMENTS,
	int _CTA_OCCUPANCY,
	int _LOG_THREADS,
	int _LOG_LOAD_VEC_SIZE,
	int _LOG_LOADS_PER_TILE,
	CacheModifier _CACHE_MODIFIER>

struct UpsweepConfig
{
	typedef _KeyType							KeyType;
	typedef _IndexType							IndexType;
	static const int RADIX_BITS					= _RADIX_BITS;
	static const int LOG_SUBTILE_ELEMENTS		= _LOG_SUBTILE_ELEMENTS;
	static const int CTA_OCCUPANCY  			= _CTA_OCCUPANCY;
	static const int LOG_THREADS 				= _LOG_THREADS;
	static const int LOG_LOAD_VEC_SIZE  		= _LOG_LOAD_VEC_SIZE;
	static const int LOG_LOADS_PER_TILE 		= _LOG_LOADS_PER_TILE;
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
template <
	typename 		UpsweepConfigType,
	typename 		PreprocessFunctorType,
	int 			_CURRENT_PASS,
	int 			_CURRENT_BIT>

struct UpsweepKernelConfig : UpsweepConfigType
{
	typedef PreprocessFunctorType					PreprocessFunctor;
	
	enum {		// N.B.: We use an enum type here b/c of a NVCC-win compiler bug involving ternary expressions in static-const fields

		RADIX_DIGITS 						= 1 << UpsweepConfigType::RADIX_BITS,
		CURRENT_PASS						= _CURRENT_PASS,
		CURRENT_BIT							= _CURRENT_BIT,

		THREADS								= 1 << UpsweepConfigType::LOG_THREADS,

		LOG_WARPS							= UpsweepConfigType::LOG_THREADS - B40C_LOG_WARP_THREADS(__B40C_CUDA_ARCH__),
		WARPS								= 1 << LOG_WARPS,

		LOAD_VEC_SIZE						= 1 << UpsweepConfigType::LOG_LOAD_VEC_SIZE,
		LOADS_PER_TILE						= 1 << UpsweepConfigType::LOG_LOADS_PER_TILE,

		LOG_TILE_ELEMENTS_PER_THREAD		= UpsweepConfigType::LOG_LOAD_VEC_SIZE + UpsweepConfigType::LOG_LOADS_PER_TILE,
		TILE_ELEMENTS_PER_THREAD			= 1 << LOG_TILE_ELEMENTS_PER_THREAD,

		LOG_TILE_ELEMENTS 					= LOG_TILE_ELEMENTS_PER_THREAD + UpsweepConfigType::LOG_THREADS,
		TILE_ELEMENTS						= 1 << LOG_TILE_ELEMENTS,

		// A lane is a row of 32-bit words, one words per thread, each words a
		// composite of four 8-bit digit counters, i.e., we need one lane for every
		// four radix digits.

		LOG_COMPOSITE_LANES 				= (UpsweepConfigType::RADIX_BITS >= 2) ?
												UpsweepConfigType::RADIX_BITS - 2 :
												0,	// Always at least one lane
		COMPOSITE_LANES 					= 1 << LOG_COMPOSITE_LANES,
	
		LOG_COMPOSITES_PER_LANE				= UpsweepConfigType::LOG_THREADS,				// Every thread contributes one partial for each lane
		COMPOSITES_PER_LANE 				= 1 << LOG_COMPOSITES_PER_LANE,
	
		// To prevent digit-counter overflow, we must partially-aggregate the
		// 8-bit composite counters back into IndexType-bit registers periodically.  The lanes
		// are divided up amongst the warps for aggregation.  Each lane is
		// therefore equivalent to four rows of IndexType-bit digit-counts, each the width of a warp.
	
		LANES_PER_WARP 						= B40C_MAX(1, COMPOSITE_LANES / WARPS),
	
		COMPOSITES_PER_LANE_PER_THREAD 		= COMPOSITES_PER_LANE / B40C_WARP_THREADS(__B40C_CUDA_ARCH__),					// Number of partials per thread to aggregate
	
		AGGREGATED_ROWS						= RADIX_DIGITS,
		AGGREGATED_PARTIALS_PER_ROW 		= B40C_WARP_THREADS(__B40C_CUDA_ARCH__),
		PADDED_AGGREGATED_PARTIALS_PER_ROW 	= AGGREGATED_PARTIALS_PER_ROW + 1,

		// Required size of the re-purposable shared memory region
		COMPOSITE_COUNTER_BYTES				= COMPOSITE_LANES * COMPOSITES_PER_LANE * sizeof(int),
		PARTIALS_BYTES						= AGGREGATED_ROWS * PADDED_AGGREGATED_PARTIALS_PER_ROW * sizeof(typename UpsweepConfigType::IndexType),
	
		SMEM_BYTES 							= B40C_MAX(COMPOSITE_COUNTER_BYTES, PARTIALS_BYTES)
	};
};


/******************************************************************************
 * Reduction kernel subroutines
 ******************************************************************************/

// Reset 8-bit composite counters back to zero
template <typename Config>
__device__ __forceinline__ void ResetCompositeLanes(int *composite_column)
{
	#pragma unroll
	for (int LANE = 0; LANE < Config::COMPOSITE_LANES; LANE++) {
		composite_column[LANE * Config::COMPOSITES_PER_LANE] = 0;
	}
}


// Partially-reduce 8-bit composite counters back into full 32-bit registers
template <typename Config> 
struct ReduceCompositeLanes
{
	typedef typename Config::IndexType IndexType; 
	
	// Next composite counter
	template <int LANE, int TOTAL_LANES, int COMPOSITE, int TOTAL_COMPOSITES>
	struct Iterate {
		static __device__ __forceinline__ void Invoke(
			IndexType local_counts[Config::LANES_PER_WARP][4],
			int *smem_pool,
			int warp_id,
			int warp_idx)
		{
			int composite_offset = 
				((warp_id << Config::LOG_COMPOSITES_PER_LANE) + warp_idx) + 	// base lane offset + base composite offset
				(LANE * Config::COMPOSITES_PER_LANE * Config::WARPS) + 			// stride to current lane
				(COMPOSITE * B40C_WARP_THREADS(__B40C_CUDA_ARCH__)); 			// stride to current composite
			
			unsigned char* composite_counters = (unsigned char *) &smem_pool[composite_offset];
			local_counts[LANE][0] += composite_counters[0];
			local_counts[LANE][1] += composite_counters[1];
			local_counts[LANE][2] += composite_counters[2];
			local_counts[LANE][3] += composite_counters[3];

			Iterate<LANE, TOTAL_LANES, COMPOSITE + 1, TOTAL_COMPOSITES>::Invoke(local_counts, smem_pool, warp_id, warp_idx);
		}
	};

	// Next lane
	template <int LANE, int TOTAL_LANES, int TOTAL_COMPOSITES>
	struct Iterate<LANE, TOTAL_LANES, TOTAL_COMPOSITES, TOTAL_COMPOSITES> {
		static __device__ __forceinline__ void Invoke(
			IndexType local_counts[Config::LANES_PER_WARP][4],
			int *smem_pool,
			int warp_id,
			int warp_idx)
		{
			Iterate<LANE + 1, TOTAL_LANES, 0, TOTAL_COMPOSITES>::Invoke(local_counts, smem_pool, warp_id, warp_idx);
		}
	};
	
	// Terminate
	template <int TOTAL_LANES, int TOTAL_COMPOSITES>
	struct Iterate<TOTAL_LANES, TOTAL_LANES, 0, TOTAL_COMPOSITES> {
		static __device__ __forceinline__ void Invoke(
			IndexType local_counts[Config::LANES_PER_WARP][4],
			int *smem_pool,
			int warp_id,
			int warp_idx) {}
	};
	
	// Interface
	static __device__ __forceinline__ void Invoke(
		IndexType local_counts[Config::LANES_PER_WARP][4],
		int *smem_pool,
		int warp_id,
		int warp_idx) 
	{
		// Accomodate radices where we have more warps than lanes
		if (warp_id < Config::COMPOSITE_LANES) {
			Iterate<0, Config::LANES_PER_WARP, 0, Config::COMPOSITES_PER_LANE_PER_THREAD>::Invoke(
				local_counts, smem_pool, warp_id, warp_idx);
		}
	}
};


// Bucket a key
template <typename Config>
__device__ __forceinline__ void Bucket(
	typename Config::KeyType key, 
	int *composite_column)
{
	typedef typename Config::KeyType KeyType;

	// Pre-process key with bit-twiddling functor if necessary
	Config::PreprocessFunctor::Transform(key, true);

	// Extract lane containing corresponding composite counter 
	int lane;
	ExtractKeyBits<
		KeyType, 
		Config::CURRENT_BIT + 2, 
		Config::RADIX_BITS - 2>::Extract(lane, key);

	if (__B40C_CUDA_ARCH__ >= 200) {	
	
		// GF100+ has special bit-extraction instructions (instead of shift+mask)
		int quad_byte;
		if (Config::RADIX_BITS < 2) { 
			ExtractKeyBits<KeyType, Config::CURRENT_BIT, 1>::Extract(quad_byte, key);
		} else {
			ExtractKeyBits<KeyType, Config::CURRENT_BIT, 2>::Extract(quad_byte, key);
		}
		
		// Increment sub-field in composite counter 
		unsigned char *encoded_col = (unsigned char *) &composite_column[FastMul(lane, Config::COMPOSITES_PER_LANE)];
		encoded_col[quad_byte]++;

	} else {

		// GT200 can save an instruction because it can source an operand 
		// directly from smem
		const int BYTE_ENCODE_SHIFT 		= 0x3;
		const KeyType QUAD_MASK 			= (Config::RADIX_BITS < 2) ? 0x1 : 0x3;

		int quad_shift = MagnitudeShift<KeyType, BYTE_ENCODE_SHIFT - Config::CURRENT_BIT>(
			key & (QUAD_MASK << Config::CURRENT_BIT));
		
		// Increment sub-field in composite counter 
		composite_column[FastMul(lane, Config::COMPOSITES_PER_LANE)] += (1 << quad_shift);
	}
}


// Bucket a tile of keys
template <typename Config> 
struct BucketTileKeys
{
	typedef typename Config::KeyType KeyType;
	
	// Iterate over vec-elements
	template <int LOAD, int TOTAL_LOADS, int VEC_ELEMENT, int TOTAL_VEC_ELEMENTS>
	struct Iterate {
		static __device__ __forceinline__ void Invoke(
			KeyType keys[Config::LOADS_PER_TILE][Config::LOAD_VEC_SIZE], 
			int *composite_column) 
		{
			Bucket<Config>(keys[LOAD][VEC_ELEMENT], composite_column);
			Iterate<LOAD, TOTAL_LOADS, VEC_ELEMENT + 1, TOTAL_VEC_ELEMENTS>::Invoke(keys, composite_column);
		}
	};

	// Iterate over loads
	template <int LOAD, int TOTAL_LOADS, int TOTAL_VEC_ELEMENTS>
	struct Iterate<LOAD, TOTAL_LOADS, TOTAL_VEC_ELEMENTS, TOTAL_VEC_ELEMENTS> {
		static __device__ __forceinline__ void Invoke(
			KeyType keys[Config::LOADS_PER_TILE][Config::LOAD_VEC_SIZE], 
			int *composite_column) 
		{
			Iterate<LOAD + 1, TOTAL_LOADS, 0, TOTAL_VEC_ELEMENTS>::Invoke(keys, composite_column);
		}
	};
	
	// Terminate
	template <int TOTAL_LOADS, int TOTAL_VEC_ELEMENTS>
	struct Iterate<TOTAL_LOADS, TOTAL_LOADS, 0, TOTAL_VEC_ELEMENTS> {
		static __device__ __forceinline__ void Invoke(
			KeyType keys[Config::LOADS_PER_TILE][Config::LOAD_VEC_SIZE], 
			int *composite_column) {} 
	};

	// Interface
	static __device__ __forceinline__ void Invoke(
		KeyType keys[Config::LOADS_PER_TILE][Config::LOAD_VEC_SIZE], 
		int *composite_column) 
	{
		Iterate<0, Config::LOADS_PER_TILE, 0, Config::LOAD_VEC_SIZE>::Invoke(keys, composite_column);
	} 
};




// Process one tile of keys
template <typename Config>
__device__ __forceinline__ void ProcessTile(
	typename Config::KeyType *d_in_keys, 
	typename Config::IndexType cta_offset, 
	int *composite_column) 
{
	typedef typename Config::KeyType KeyType;
	typedef typename Config::IndexType IndexType;

	// Load tile of keys
	KeyType keys[Config::LOADS_PER_TILE][Config::LOAD_VEC_SIZE];

	LoadTile<
		KeyType, 
		IndexType, 
		Config::LOG_LOADS_PER_TILE, 
		Config::LOG_LOAD_VEC_SIZE, 
		Config::THREADS, 
		Config::CACHE_MODIFIER, 
		true>::Invoke(keys, d_in_keys, cta_offset);

//	if (__B40C_CUDA_ARCH__ >= 200) __syncthreads();
	if (Config::LOADS_PER_TILE > 1) __syncthreads();		// Prevents bucketing from being hoisted up into loads 

	// Bucket tile of keys
	BucketTileKeys<Config>::Invoke(keys, composite_column);
}


// Unroll tiles
template <typename Config, int UNROLL_COUNT> 
struct UnrollTiles
{
	typedef typename Config::KeyType KeyType;
	typedef typename Config::IndexType IndexType;
	
	// Iterate over counts
	template <int COUNT, int __dummy = 0>
	struct Iterate {
		static __device__ __forceinline__ void Invoke(
			KeyType *d_in_keys, 
			IndexType cta_offset, 
			int *composite_column)
		{
			ProcessTile<Config>(d_in_keys, cta_offset, composite_column);
			Iterate<COUNT + 1>::Invoke(d_in_keys, cta_offset + Config::TILE_ELEMENTS, composite_column);
		}
	};

	// Terminate
	template <int __dummy>
	struct Iterate<UNROLL_COUNT, __dummy> {
		static __device__ __forceinline__ void Invoke(
			KeyType *d_in_keys, 
			IndexType cta_offset, 
			int *composite_column) {}
	};
};


// Unroll key processing in batches of UNROLL_COUNT tiles, optionally with 
// occasional partial-reduction to avoid overflow.
template <typename Config, int UNROLL_COUNT, bool REDUCE_AFTERWARD>
__device__ __forceinline__ void UnrollTileBatches(
	typename Config::KeyType *d_in_keys,
	typename Config::IndexType &cta_offset,
	int* 	composite_column,
	int*	smem_pool,
	typename Config::IndexType out_of_bounds,
	typename Config::IndexType local_counts[Config::LANES_PER_WARP][4],
	int 	warp_id,
	int 	warp_idx) 
{
	const int UNROLLED_ELEMENTS = UNROLL_COUNT * Config::TILE_ELEMENTS;
	
	while (cta_offset + UNROLLED_ELEMENTS < out_of_bounds) {
	
		UnrollTiles<Config, UNROLL_COUNT>::template Iterate<0>::Invoke(
			d_in_keys, cta_offset, composite_column);

		cta_offset += UNROLLED_ELEMENTS;

		// Optionally aggregate back into local_count registers to prevent overflow
		if (REDUCE_AFTERWARD) {
			
			__syncthreads();
	
			ReduceCompositeLanes<Config>::Invoke(
				local_counts, smem_pool, warp_id, warp_idx);
	
			__syncthreads();
			
			// Reset composite counters
			ResetCompositeLanes<Config>(composite_column);
		}
	} 
}


// Reduces all full-tiles (unguarded loads)
template <typename Config, bool AGGRESSIVELY_UNROLL> struct FullTiles;


// Reduces all full-tiles with modest unrolling (for lower static instruction count)
template <typename Config>
struct FullTiles<Config, false>
{
	typedef typename Config::KeyType KeyType;
	typedef typename Config::IndexType IndexType;
	
	__device__ __forceinline__ static void Reduce(
		KeyType 	*d_in_keys,
		IndexType 	&cta_offset,
		int			*composite_column,
		int			*smem_pool,
		IndexType  	out_of_bounds,
		IndexType  	local_counts[Config::LANES_PER_WARP][4],
		int  		warp_id,
		int  		warp_idx) 
	{
		// Loop over tile-batches of 32 keys per thread, aggregating after each batch
		UnrollTileBatches<Config, 32 / Config::TILE_ELEMENTS_PER_THREAD, true>(
			d_in_keys, cta_offset, composite_column, smem_pool, out_of_bounds, local_counts, warp_id, warp_idx); 

		// Loop over tiles one at a time
		UnrollTileBatches<Config, 1, false>(
			d_in_keys, cta_offset, composite_column, smem_pool, out_of_bounds, local_counts, warp_id, warp_idx); 
	}
};


// Reduces all full-tiles with aggressive unrolling (for lower dynamic instruction count)
template <typename Config>
struct FullTiles<Config, true>
{
	typedef typename Config::KeyType KeyType;
	typedef typename Config::IndexType IndexType;
	
	__device__ __forceinline__ static void Reduce(
		KeyType 	*d_in_keys,
		IndexType 	&cta_offset,
		int			*composite_column,
		int			*smem_pool,
		IndexType  	out_of_bounds,
		IndexType  	local_counts[Config::LANES_PER_WARP][4],
		int  		warp_id,
		int  		warp_idx) 
	{
		// Loop over tile-batches of 128 keys per thread, aggregating after each batch
		UnrollTileBatches<Config, 128 / Config::TILE_ELEMENTS_PER_THREAD, true>(
			d_in_keys, cta_offset, composite_column, smem_pool, out_of_bounds, local_counts, warp_id, warp_idx); 

		// Loop over tile-batches of 8 keys per thread
		UnrollTileBatches<Config, 8 / Config::TILE_ELEMENTS_PER_THREAD, false>(
			d_in_keys, cta_offset, composite_column, smem_pool, out_of_bounds, local_counts, warp_id, warp_idx); 
		
		// Loop over tiles one at a time
		UnrollTileBatches<Config, 1, false>(
			d_in_keys, cta_offset, composite_column, smem_pool, out_of_bounds, local_counts, warp_id, warp_idx); 
	}
};


template <
	typename Config, 
	bool AGGRESSIVE_UNROLLING>
__device__ __forceinline__ void ReductionPass(
	typename Config::KeyType 	*d_in_keys,
	typename Config::IndexType 	*d_spine,
	int*	smem_pool,
	int* 	composite_column,
	typename Config::IndexType 	cta_offset,
	typename Config::IndexType 	out_of_bounds)
{
	typedef typename Config::KeyType KeyType; 
	typedef typename Config::IndexType IndexType;
	
	int warp_id = threadIdx.x >> B40C_LOG_WARP_THREADS(__B40C_CUDA_ARCH__);
	int warp_idx = threadIdx.x & (B40C_WARP_THREADS(__B40C_CUDA_ARCH__) - 1);
	
	// Each thread is responsible for aggregating an unencoded segment of a fours-group
	IndexType local_counts[Config::LANES_PER_WARP][4];								
	
	// Initialize local counts
	#pragma unroll 
	for (int LANE = 0; LANE < Config::LANES_PER_WARP; LANE++) {
		local_counts[LANE][0] = 0;
		local_counts[LANE][1] = 0;
		local_counts[LANE][2] = 0;
		local_counts[LANE][3] = 0;
	}
	
	// Reset encoded counters
	ResetCompositeLanes<Config>(composite_column);
	
	// Process loads in bulk (if applicable), making sure to leave
	// enough headroom for one more (partial) tile before rollover  
	FullTiles<Config, AGGRESSIVE_UNROLLING>::Reduce(
		d_in_keys,
		cta_offset,
		composite_column,
		smem_pool,
		out_of_bounds,
		local_counts, 
		warp_id,
		warp_idx);

	// Process (potentially-partial) loads singly
	while (cta_offset + threadIdx.x < out_of_bounds) {
		KeyType key;
		ModifiedLoad<KeyType, Config::CACHE_MODIFIER>::Ld(key, d_in_keys, cta_offset + threadIdx.x);
		Bucket<Config>(key, composite_column);
		cta_offset += Config::THREADS;
	}
	
	__syncthreads();
	
	// Aggregate back into local_count registers 
	ReduceCompositeLanes<Config>::Invoke(local_counts, smem_pool, warp_id, warp_idx);

	__syncthreads();

	
	//
	// Final raking reduction of aggregated local_counts within each warp  
	//
	
	IndexType *raking_pool = reinterpret_cast<IndexType*>(smem_pool);

	// Iterate over lanes per warp, placing the four counts from each into smem
	int base_row_offset =		// My thread's (first) reduction counter placement offset 
		FastMul(warp_id, Config::PADDED_AGGREGATED_PARTIALS_PER_ROW * Config::WARPS) + warp_idx;	

	if (warp_id < Config::COMPOSITE_LANES) {

		// We have at least one lane to place
		#pragma unroll
		for (int i = 0; i < Config::LANES_PER_WARP; i++) {
	
			// Four counts per composite lane
			raking_pool[base_row_offset + (Config::PADDED_AGGREGATED_PARTIALS_PER_ROW * 0)] = local_counts[i][0];
			raking_pool[base_row_offset + (Config::PADDED_AGGREGATED_PARTIALS_PER_ROW * 1)] = local_counts[i][1];
			raking_pool[base_row_offset + (Config::PADDED_AGGREGATED_PARTIALS_PER_ROW * 2)] = local_counts[i][2];
			raking_pool[base_row_offset + (Config::PADDED_AGGREGATED_PARTIALS_PER_ROW * 3)] = local_counts[i][3];
			
			base_row_offset += Config::PADDED_AGGREGATED_PARTIALS_PER_ROW * Config::WARPS;
		}
	}

	__syncthreads();

	// Rake-reduce and write out the digit_count reductions 
	if (threadIdx.x < Config::RADIX_DIGITS) {

		int base_row_offset = FastMul(threadIdx.x, Config::PADDED_AGGREGATED_PARTIALS_PER_ROW);

		IndexType digit_count = SerialReduce<IndexType, Config::AGGREGATED_PARTIALS_PER_ROW>::Invoke(
			raking_pool + base_row_offset);

		int spine_digit_offset = FastMul(gridDim.x, threadIdx.x) + blockIdx.x;
		d_spine[spine_digit_offset] = digit_count;
	}
}





/**
 * Kernel entry point
 */
template <typename KernelConfig>
__launch_bounds__ (KernelConfig::THREADS, KernelConfig::CTA_OCCUPANCY)
__global__ 
void LsbRakingReductionKernel(
	int *d_selectors,
	typename KernelConfig::IndexType 	*d_spine,
	typename KernelConfig::KeyType 		*d_in_keys,
	typename KernelConfig::KeyType 		*d_out_keys,
	CtaDecomposition<typename KernelConfig::IndexType> work_decomposition)
{
	typedef typename KernelConfig::IndexType IndexType;
	
	// Shared memory pool
	__shared__ unsigned char smem_pool[KernelConfig::SMEM_BYTES];
	
	// The smem column in composite lanes for each thread 
	int *composite_column = reinterpret_cast<int*>(smem_pool) + threadIdx.x;	// first element of column

	// Determine where to read our input
	if ((KernelConfig::CURRENT_PASS != 0) && (d_selectors[KernelConfig::CURRENT_PASS & 0x1])) {
		d_in_keys = d_out_keys;
	}
	
	// Determine our threadblock's work range
	
	IndexType cta_offset;			// Offset at which this CTA begins processing
	IndexType cta_elements;			// Total number of elements for this CTA to process
	IndexType guarded_offset; 		// Offset of final, partially-full tile (requires guarded loads)
	IndexType guarded_elements;		// Number of elements in partially-full tile

	work_decomposition.GetCtaWorkLimits<KernelConfig::LOG_TILE_ELEMENTS, KernelConfig::LOG_SUBTILE_ELEMENTS>(
		cta_offset, cta_elements, guarded_offset, guarded_elements);
		
	// Perform reduction pass over work range with agressive unrolling
	ReductionPass<KernelConfig, true>(
		d_in_keys,
		d_spine,
		reinterpret_cast<int*>(smem_pool),
		composite_column,
		cta_offset,
		cta_offset + cta_elements);
} 


/**
 * Host stub to calm the linker for arch-specializations that we didn't
 * end up compiling PTX for.
 */
template <typename KernelConfig>
__host__ void __wrapper__device_stub_LsbRakingReductionKernel(
	int 								*&,
	typename KernelConfig::IndexType 	*&,
	typename KernelConfig::KeyType 		*&,
	typename KernelConfig::KeyType 		*&,
	CtaDecomposition<typename KernelConfig::IndexType> &)
{}
 


} // namespace upsweep
} // namespace lsb_radix_sort
} // namespace b40c

