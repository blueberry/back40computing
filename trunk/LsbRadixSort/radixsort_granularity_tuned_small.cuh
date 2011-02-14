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
 *  
 ******************************************************************************/

#pragma once

#include "b40c_cuda_properties.cuh"
#include "b40c_kernel_data_movement.cuh"
#include "radixsort_granularity.cuh"

namespace b40c {
namespace lsb_radix_sort {

/**
 * Small-problem granularity parameterization type.
 * 
 * We can tune this type per SM-architecture, per problem type.  Parameters
 * for separate kernels are largely performance-independent.
 */
template <
	int SM_ARCH, 
	typename KeyType, 
	typename ValueType,
	typename IndexType>
struct SmallGranularityConfig : 
	GranularityConfig<

		//---------------------------------------------------------------------
		// Common
		//---------------------------------------------------------------------
		
		// KeyType
		KeyType,
		
		// ValueType
		ValueType,
		
		// IndexType
		IndexType,
		
		// RADIX_BITS
		(SM_ARCH >= 200) ? 					4 :		// 4-bit radix digits on GF100+
		(SM_ARCH >= 120) ? 					4 :		// 4-bit radix digits on GT200
											4,		// 4-bit radix digits on G80/90
		
		// LOG_SUBTILE_ELEMENTS
		(SM_ARCH >= 200) ? 					9 :		// 512 elements on GF100+
		(SM_ARCH >= 120) ? 					9 :		// 512 elements on GT200
											9,		// 512 elements on G80/90
		
		// CACHE_MODIFIER
		NONE,										// Default (CA: cache all levels)
		
		// EARLY_EXIT								// Default (no early termination)
		false,
		
		// UNIFORM_SMEM_ALLOCATION
		(SM_ARCH >= 200) ? 					false :		// No on GF100+
		(SM_ARCH >= 120) ? 					true :		// Yes on GT200
											false,		// No on G80/90
											
		// UNIFORM_GRID_SIZE
		(SM_ARCH >= 200) ? 					true :		// No on GF100+
		(SM_ARCH >= 120) ? 					true :		// No on GT200
											true,		// No on G80/90
											
		//---------------------------------------------------------------------
		// Upsweep Kernel
		//---------------------------------------------------------------------
		
		// UPSWEEP_CTA_OCCUPANCY
		(SM_ARCH >= 200) ? 					8 :		// 8 CTAs/SM on GF100+
		(SM_ARCH >= 120) ? 					5 :		// 5 CTAs/SM on GT200
											3,		// 3 CTAs/SM on G80/90
										
		// UPSWEEP_LOG_THREADS
		(SM_ARCH >= 200) ? 					7 :		// 128 threads/CTA on GF100+
		(SM_ARCH >= 120) ? 					7 :		// 128 threads/CTA on GT200
											7,		// 128 threads/CTA on G80/90
		
		// UPSWEEP_LOG_LOAD_VEC_SIZE
		(SM_ARCH >= 200) ? 					1 :		// vec-2 loads on GF100+
		(SM_ARCH >= 120) ? 					0 :		// vec-1 loads on GT200
											0,		// vec-1 loads on G80/90
		
		// UPSWEEP_LOG_LOADS_PER_TILE
		(SM_ARCH >= 200) ? 					0 :		// 1 load/tile on GF100+
		(SM_ARCH >= 120) ? 					0 :		// 1 load/tile on GT200
											0,		// 1 load/tile on G80/90
		
		//---------------------------------------------------------------------
		// Spine-scan Kernel
		//---------------------------------------------------------------------
		
		// SPINE_CTA_OCCUPANCY
		1,											// 1 CTA/SM on all architectures
		
		// SPINE_LOG_THREADS
		(SM_ARCH >= 200) ? 					8 :		// 256 threads/CTA on GF100+
		(SM_ARCH >= 120) ? 					7 :		// 128 threads/CTA on GT200
											7,		// 128 threads/CTA on G80/90
		
		// SPINE_LOG_LOAD_VEC_SIZE
		(SM_ARCH >= 200) ? 					2 :		// vec-4 loads on GF100+
		(SM_ARCH >= 120) ? 					2 :		// vec-4 loads on GT200
											2,		// vec-4 loads on G80/90
		
		// SPINE_LOG_LOADS_PER_TILE
		(SM_ARCH >= 200) ? 					0 :		// 1 loads/tile on GF100+
		(SM_ARCH >= 120) ? 					0 :		// 1 loads/tile on GT200
											0,		// 1 loads/tile on G80/90
		
		// SPINE_LOG_RAKING_THREADS
		(SM_ARCH >= 200) ? 					B40C_LOG_WARP_THREADS(SM_ARCH) + 0 :	// 1 warp on GF100+
		(SM_ARCH >= 120) ? 					B40C_LOG_WARP_THREADS(SM_ARCH) + 0 :	// 1 warp on GT200
											B40C_LOG_WARP_THREADS(SM_ARCH) + 0,		// 1 warp on G80/90
		
		
		//---------------------------------------------------------------------
		// Downsweep Kernel
		//---------------------------------------------------------------------
		
		// DOWNSWEEP_CTA_OCCUPANCY
		(SM_ARCH >= 200) ? 					4 :		// 4 CTAs/SM on GF100+
		(SM_ARCH >= 120) ? 					5 :		// 5 CTAs/SM on GT200
											2,		// 2 CTAs/SM on G80/90
		
		// DOWNSWEEP_LOG_THREADS
		(SM_ARCH >= 200) ? 					8 :		// 256 threads/CTA on GF100+
		(SM_ARCH >= 120) ? 					7 :		// 128 threads/CTA on GT200
											7,		// 128 threads/CTA on G80/90
		
		// DOWNSWEEP_LOG_LOAD_VEC_SIZE
		(SM_ARCH >= 200) ? 					0 :		// vec-1 loads on GF100+
		(SM_ARCH >= 120) ? 					1 :		// vec-2 loads on GT200
											1,		// vec-2 loads on G80/90
		
		// DOWNSWEEP_LOG_LOADS_PER_CYCLE
		(SM_ARCH >= 200) ? 					1 :		// 2 loads/cycle on GF100+
		(SM_ARCH >= 120) ? 					0 :		// 1 load/cycle on GT200
											1,		// 2 loads/cycle on G80/90
		
		// DOWNSWEEP_LOG_CYCLES_PER_TILE
		(SM_ARCH >= 200) ?
			(B40C_MAX(sizeof(KeyType), sizeof(ValueType)) > 4 ?
												0 : 	// 1 cycle/tile on GF100+ for large (64bit+) keys|values
												0) :	// 1 cycles/tile on GF100+
		(SM_ARCH >= 120) ?
			(B40C_MAX(sizeof(KeyType), sizeof(ValueType)) > 4 ?
												0 : 	// 1 cycle/tile on GT200 for large (64bit+) keys|values
												1) :	// 2 cycles/tile on GT200
											1,			// 2 cycles/tile on G80/90
										
		// DOWNSWEEP_LOG_RAKING_THREADS
		(SM_ARCH >= 200) ? 					B40C_LOG_WARP_THREADS(SM_ARCH) + 2 :	// 4 warps on GF100+
		(SM_ARCH >= 120) ? 					B40C_LOG_WARP_THREADS(SM_ARCH) + 0 :	// 1 warp on GT200
											B40C_LOG_WARP_THREADS(SM_ARCH) + 2		// 4 warps on G80/90
	> 
{};


}// namespace lsb_radix_sort
}// namespace b40c

