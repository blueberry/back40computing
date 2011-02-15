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
 * Default (i.e., large-problem) "granularity tuning types" for LSB
 * radix sorting
 ******************************************************************************/

#pragma once

#include "b40c_cuda_properties.cuh"
#include "b40c_kernel_data_movement.cuh"
#include "radixsort_granularity.cuh"

namespace b40c {
namespace lsb_radix_sort {
namespace large_problem_tuning {

/**
 * Enumeration of architecture-families that we have tuned for
 */
enum Family
{
	SM20 	= 200,
	SM13	= 130,
	SM10	= 100
};


/**
 * Classifies a given CUDA_ARCH into an architecture-family
 */
template <int CUDA_ARCH>
struct FamilyClassifier
{
	static const Family FAMILY =	(CUDA_ARCH < SM13) ? 	SM10 :
									(CUDA_ARCH < SM20) ? 	SM13 :
															SM20;
};


/**
 * Granularity parameterization type
 *
 * We can tune this type per SM-architecture, per problem type.  Parameters
 * for separate kernels are largely performance-independent.
 */
template <
	int CUDA_ARCH,
	typename KeyType,
	typename ValueType,
	typename IndexType>
struct TunedConfig :
	TunedConfig<FamilyClassifier<CUDA_ARCH>::FAMILY, KeyType, ValueType, IndexType> {};



//-----------------------------------------------------------------------------
// SM2.0 default granularity parameterization type
//-----------------------------------------------------------------------------

template <typename KeyType, typename ValueType, typename IndexType>
struct TunedConfig<SM20, KeyType, ValueType, IndexType>
	: GranularityConfig<
		KeyType,				// KeyType
		ValueType,				// ValueType
		IndexType,				// IndexType

		// Common
		4,						// RADIX_BITS: 				4-bit radix digits
		10,						// LOG_SUBTILE_ELEMENTS: 	1024 subtile elements
		NONE,					// CACHE_MODIFIER: 			Default (CA: cache all levels)
		true,					// EARLY_EXIT: 				Terminate downsweep if homogeneous digits
		false,					// UNIFORM_SMEM_ALLOCATION:	No dynamic smem padding added
		true, 					// UNIFORM_GRID_SIZE: 		Use "do-nothing" spine-scan CTAs maintain constant grid size across all kernels
											
		// Upsweep Kernel
		8,						// UPSWEEP_CTA_OCCUPANCY: 		8 CTAs/SM
		7,						// UPSWEEP_LOG_THREADS: 		128 threads/CTA
		0,						// UPSWEEP_LOG_LOAD_VEC_SIZE: 	vec-1 loads
		2,						// UPSWEEP_LOG_LOADS_PER_TILE: 	4 loads/tile
	
		// Spine-scan Kernel
		1,										// SPINE_CTA_OCCUPANCY: 		Only 1 CTA/SM really needed
		7,										// SPINE_LOG_THREADS: 			128 threads/CTA
		2,										// SPINE_LOG_LOAD_VEC_SIZE:		vec-4 loads
		0,										// SPINE_LOG_LOADS_PER_TILE:	1 loads/tile
		B40C_LOG_WARP_THREADS(CUDA_ARCH) + 0,	// SPINE_LOG_RAKING_THREADS:	1 warp
	
		// Downsweep Kernel
		8,						// DOWNSWEEP_CTA_OCCUPANCY: 		8 CTAs/SM
		6,						// DOWNSWEEP_LOG_THREADS: 			64 threads/CTA
		2,						// DOWNSWEEP_LOG_LOAD_VEC_SIZE: 	vec-4 loads
		1,						// DOWNSWEEP_LOG_LOADS_PER_CYCLE: 	2 loads/cycle
		(B40C_MAX(sizeof(KeyType), sizeof(ValueType)) <= 4) ?
			1 : 					// Normal keys|values: 	DOWNSWEEP_LOG_CYCLES_PER_TILE: 2 cycles/tile
			0, 						// Large keys|values: 	DOWNSWEEP_LOG_CYCLES_PER_TILE: 1 cycle/tile
		B40C_LOG_WARP_THREADS(CUDA_ARCH) + 1		// DOWNSWEEP_LOG_RAKING_THREADS: 2 warps
	> 
{}; 



//-----------------------------------------------------------------------------
// SM1.3 default granularity parameterization type
//-----------------------------------------------------------------------------

template <typename KeyType, typename ValueType, typename IndexType>
struct TunedConfig<SM13, KeyType, ValueType, IndexType>
	: GranularityConfig<
		KeyType,				// KeyType
		ValueType,				// ValueType
		IndexType,				// IndexType

		// Common
		4,						// RADIX_BITS: 				4-bit radix digits
		9,						// LOG_SUBTILE_ELEMENTS: 	512 subtile elements
		NONE,					// CACHE_MODIFIER: 			Default (CA: cache all levels)
		true,					// EARLY_EXIT: 				Terminate downsweep if homogeneous digits
		true,					// UNIFORM_SMEM_ALLOCATION:	Use dynamic smem padding to maintain constant grid smem allocations across all kernels
		true, 					// UNIFORM_GRID_SIZE: 		Use "do-nothing" spine-scan CTAs maintain constant grid size across all kernels

		// Upsweep Kernel
		5,						// UPSWEEP_CTA_OCCUPANCY:		8 CTAs/SM
		7,						// UPSWEEP_LOG_THREADS: 		128 threads/CTA
		1,						// UPSWEEP_LOG_LOAD_VEC_SIZE: 	vec-2 loads
		0,						// UPSWEEP_LOG_LOADS_PER_TILE: 	1 loads/tile

		// Spine-scan Kernel
		1,										// SPINE_CTA_OCCUPANCY: 		Only 1 CTA/SM really needed
		7,										// SPINE_LOG_THREADS: 			128 threads/CTA
		2,										// SPINE_LOG_LOAD_VEC_SIZE: 	vec-4 loads
		0,										// SPINE_LOG_LOADS_PER_TILE: 	1 loads/tile
		B40C_LOG_WARP_THREADS(CUDA_ARCH) + 0,	// SPINE_LOG_RAKING_THREADS:	1 warp

		// Downsweep Kernel
		5,						// DOWNSWEEP_CTA_OCCUPANCY: 		8 CTAs/SM
		6,						// DOWNSWEEP_LOG_THREADS: 			64 threads/CTA
		2,						// DOWNSWEEP_LOG_LOAD_VEC_SIZE: 	vec-4 loads
		1,						// DOWNSWEEP_LOG_LOADS_PER_CYCLE: 	2 loads/cycle
		(B40C_MAX(sizeof(KeyType), sizeof(ValueType)) <= 4) ?
			0 : 					// Normal keys|values: 	DOWNSWEEP_LOG_CYCLES_PER_TILE: 1 cycles/tile
			0, 						// Large keys|values: 	DOWNSWEEP_LOG_CYCLES_PER_TILE: 1 cycle/tile
		B40C_LOG_WARP_THREADS(CUDA_ARCH) + 0		// DOWNSWEEP_LOG_RAKING_THREADS: 1 warps
	>
{};



//-----------------------------------------------------------------------------
// SM1.0 default granularity parameterization type
//-----------------------------------------------------------------------------

template <typename KeyType, typename ValueType, typename IndexType>
struct TunedConfig<SM10, KeyType, ValueType, IndexType>
	: GranularityConfig<
		KeyType,				// KeyType
		ValueType,				// ValueType
		IndexType,				// IndexType

		// Common
		4,						// RADIX_BITS: 				4-bit radix digits
		9,						// LOG_SUBTILE_ELEMENTS: 	512 subtile elements
		NONE,					// CACHE_MODIFIER: 			Default (CA: cache all levels)
		true,					// EARLY_EXIT: 				Terminate downsweep if homogeneous digits
		false,					// UNIFORM_SMEM_ALLOCATION:	No dynamic smem padding added
		true, 					// UNIFORM_GRID_SIZE: 		Use "do-nothing" spine-scan CTAs maintain constant grid size across all kernels

		// Upsweep Kernel
		3,						// UPSWEEP_CTA_OCCUPANCY:		8 CTAs/SM
		7,						// UPSWEEP_LOG_THREADS: 		128 threads/CTA
		0,						// UPSWEEP_LOG_LOAD_VEC_SIZE: 	vec-1 loads
		0,						// UPSWEEP_LOG_LOADS_PER_TILE: 	1 loads/tile

		// Spine-scan Kernel
		1,										// SPINE_CTA_OCCUPANCY: 		Only 1 CTA/SM really needed
		7,										// SPINE_LOG_THREADS: 			128 threads/CTA
		2,										// SPINE_LOG_LOAD_VEC_SIZE: 	vec-4 loads
		0,										// SPINE_LOG_LOADS_PER_TILE: 	1 loads/tile
		B40C_LOG_WARP_THREADS(CUDA_ARCH) + 0,	// SPINE_LOG_RAKING_THREADS:	1 warp

		// Downsweep Kernel
		2,						// DOWNSWEEP_CTA_OCCUPANCY: 		8 CTAs/SM
		7,						// DOWNSWEEP_LOG_THREADS: 			64 threads/CTA
		1,						// DOWNSWEEP_LOG_LOAD_VEC_SIZE: 	vec-4 loads
		1,						// DOWNSWEEP_LOG_LOADS_PER_CYCLE: 	2 loads/cycle
		(B40C_MAX(sizeof(KeyType), sizeof(ValueType)) <= 4) ?
			1 : 					// Normal keys|values: 	DOWNSWEEP_LOG_CYCLES_PER_TILE: 2 cycles/tile
			1, 						// Large keys|values: 	DOWNSWEEP_LOG_CYCLES_PER_TILE: 2 cycle/tile
		B40C_LOG_WARP_THREADS(CUDA_ARCH) + 2		// DOWNSWEEP_LOG_RAKING_THREADS: 4 warps
	>
{};




}// namespace large_problem_tuning
}// namespace lsb_radix_sort
}// namespace b40c

