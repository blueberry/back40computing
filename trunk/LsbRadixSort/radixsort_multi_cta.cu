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
 * 
 * 
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
 ******************************************************************************/



/******************************************************************************
 * Radix Sorting API
 ******************************************************************************/

#pragma once

#include "radixsort_base.cu"

namespace b40c {


/**
 * Base class for multi-CTA sorting enactors
 */
template <typename K, typename V, typename Storage>
class MultiCtaRadixSortingEnactor : public BaseRadixSortingEnactor<K, V, Storage >
{
private:
	
	typedef BaseRadixSortingEnactor<K, V, Storage> Base; 
	
protected:
	
	// Maximum number of threadblocks this enactor will launch
	int max_grid_size;

	// Fixed "tile size" of keys by which threadblocks iterate over 
	int tile_elements;
	
	// Temporary device storage needed for scanning digit histograms produced
	// by separate CTAs
	int *d_spine;
	
protected:
	
	/**
	 * Constructor.
	 */
	MultiCtaRadixSortingEnactor(
		int max_grid_size,
		int tile_elements,
		int max_radix_bits,
		const CudaProperties &props = CudaProperties()) : 
			Base::BaseRadixSortingEnactor(props),  
			max_grid_size(max_grid_size), 
			tile_elements(tile_elements),
			d_spine(NULL)
	{
		// Allocate the spine
		int spine_elements = max_grid_size * (1 << max_radix_bits);
		int spine_cycles = (spine_elements + B40C_RADIXSORT_SPINE_CYCLE_ELEMENTS - 1) / 
				B40C_RADIXSORT_SPINE_CYCLE_ELEMENTS;
		spine_elements = spine_cycles * B40C_RADIXSORT_SPINE_CYCLE_ELEMENTS;
		cudaMalloc((void**) &d_spine, spine_elements * sizeof(int));
	}


	/**
	 * Computes the work-decomposition amongst CTAs for the give problem size 
	 * and grid size
	 */
	void GetWorkDecomposition(
		int num_elements, 
		int grid_size,
		CtaDecomposition &work_decomposition) 
	{
		int total_tiles 		= num_elements / tile_elements;
		int tiles_per_block 	= total_tiles / grid_size;						
		int extra_tiles 		= total_tiles - (tiles_per_block * grid_size);

		work_decomposition.num_big_blocks 				= extra_tiles;
		work_decomposition.big_block_elements 			= (tiles_per_block + 1) * tile_elements;
		work_decomposition.normal_block_elements 		= tiles_per_block * tile_elements;
		work_decomposition.extra_elements_last_block 	= num_elements - (total_tiles * tile_elements);
		work_decomposition.num_elements 				= num_elements;
	}
	

public:

    /**
     * Destructor
     */
    virtual ~MultiCtaRadixSortingEnactor() 
    {
   		if (d_spine) cudaFree(d_spine);
    }
    
};






}// namespace b40c

