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
 * Multi-CTA LSB Sorting Base Class & Storage API 
 ******************************************************************************/

#pragma once

#include "radixsort_base.cu"
#include "radixsort_reduction_kernel.cu"
#include "radixsort_spine_kernel.cu"
#include "radixsort_scanscatter_kernel.cu"

namespace b40c {

using namespace lsb_radix_sort;


/******************************************************************************
 *  Storage-wrapper / Problem-descriptor API 
 ******************************************************************************/

/**
 * Base storage management structure for multi-CTA-sorting device vectors.
 * 
 * Everything is public in these structures.  It’s just a simple transparent 
 * mechanism for encapsulating a specific sorting problem for delivery to  
 * a sorting enactor. The caller is free to assign and alias the storage as 
 * they see fit and to change the num_elements field for arbitrary extents. This 
 * provides maximum flexibility for re-using device allocations for subsequent 
 * sorting operations.  As such, it is the caller's responsibility to free any 
 * non-NULL storage arrays when no longer needed.
 * 
 * Multi-CTA sorting is performed out-of-core, meaning that sorting passes
 * must have two equally sized arrays: one for reading in from, the other for 
 * writing out to.  As such, this structure maintains a pair of device vectors 
 * for keys (and for values), and a "selector" member to index which vector 
 * contains valid data (i.e., the data to-be-sorted, or the valid-sorted data 
 * after a sorting operation). 
 * 
 * E.g., consider a MultiCtaSortStorage "device_storage".  The valid data 
 * should always be accessible by: 
 * 
 * 		device_storage.d_keys[device_storage.selector];
 * 
 * The non-selected array(s) can be allocated lazily upon first sorting by the 
 * sorting enactor if left NULL, or a-priori by the caller.  (If user-allocated, 
 * they should be large enough to accomodate num_elements.)    
 * 
 * NOTE: After a sorting operation has completed, the selecter member will
 * index the key (and value) pointers that contain the final sorted results.
 * (E.g., an odd number of sorting passes may leave the results in d_keys[1] if 
 * the input started in d_keys[0].)
 * 
 */
template <typename KeyType, typename ValueType, typename _IndexType> 
class MultiCtaSortStorageBase
{
public:	
	
	// Integer type suitable for indexing storage elements 
	// (e.g., int, long long, etc.)
	typedef _IndexType IndexType;

	// Pair of device vector pointers for keys
	KeyType* d_keys[2];
	
	// Pair of device vector pointers for values
	ValueType* d_values[2];

	// Number of elements for sorting in the above vectors 
	IndexType num_elements;
	
	// Selector into the pair of device vector pointers indicating valid 
	// sorting elements (i.e., where the results are)
	int selector;

protected:	
	
	// Constructor
	MultiCtaSortStorageBase() 
	{
		num_elements = 0;
		selector = 0;
		d_keys[0] = NULL;
		d_keys[1] = NULL;
		d_values[0] = NULL;
		d_values[1] = NULL;
	}
};


/**
 * Standard sorting storage wrapper and problem descriptor.  
 *
 * For use in sorting up to 2^31 elements.   
 */
template <typename KeyType, typename ValueType = KeysOnly> 
struct MultiCtaSortStorage : 
	public MultiCtaSortStorageBase<KeyType, ValueType, int>
{
public:
	// Typedef for base class
	typedef MultiCtaSortStorageBase<KeyType, ValueType, int> Base;
		
	// Constructor
	MultiCtaSortStorage() : Base::MultiCtaSortStorageBase() {}
	
	// Constructor
	MultiCtaSortStorage(int num_elements) : Base::MultiCtaSortStorageBase() 
	{
		this->num_elements = num_elements;
	}

	// Constructor
	MultiCtaSortStorage(
		int num_elements, 
		KeyType* keys, 
		ValueType* values = NULL) : Base::MultiCtaSortStorageBase()
	{
		this->num_elements = num_elements;
		this->d_keys[0] = keys;
		this->d_values[0] = values;
	}
};


/**
 * 64-bit sorting storage wrapper and problem descriptor.  
 *
 * For use in sorting up to 2^63 elements.  May exhibit lower performance
 * than the (32-bit) MultiCtaSortStorage wrapper due to increased 
 * register pressure and memory workloads.
 */
template <typename KeyType, typename ValueType = KeysOnly> 
struct MultiCtaSortStorage64 : 
	public MultiCtaSortStorageBase<KeyType, ValueType, long long>
{
public:
	// Typedef for base class
	typedef MultiCtaSortStorageBase<KeyType, ValueType, long long> Base;
		
	// Constructor
	MultiCtaSortStorage64() : Base::MultiCtaSortStorageBase() {}
	
	// Constructor
	MultiCtaSortStorage64(long long num_elements) : Base::MultiCtaSortStorageBase() 
	{
		this->num_elements = num_elements;
	}

	// Constructor
	MultiCtaSortStorage64(
		long long num_elements, 
		KeyType* keys, 
		ValueType* values = NULL) : Base::MultiCtaSortStorageBase()
	{
		this->num_elements = num_elements;
		this->d_keys[0] = keys;
		this->d_values[0] = values;
	}
};

		
/******************************************************************************
 * Multi-CTA Sorting Enactor Base Class
 ******************************************************************************/

/**
 * Base class for multi-CTA sorting enactors
 */
template <
	typename KeyType,
	typename ConvertedKeyType,
	typename ValueType>

class MultiCtaLsbSortEnactor : 
	public BaseLsbSortEnactor<KeyType, ValueType>
{
protected:
	
	//---------------------------------------------------------------------
	// Utility Types
	//---------------------------------------------------------------------

	// Typedef for base class
	typedef BaseLsbSortEnactor<KeyType, ValueType> Base; 

	// Wrapper for managing the state for a specific sorting operation 
	template <typename Storage>
	struct SortingCtaDecomposition : CtaDecomposition<typename Storage::IndexType>
	{
		int sweep_grid_size;
		int spine_elements;
		Storage problem_storage;
		
		// Constructor
		SortingCtaDecomposition(
			const Storage &problem_storage,
			int subtile_elements,
			int sweep_grid_size,
			int spine_elements) :
				
				CtaDecomposition<typename Storage::IndexType>(
					problem_storage.num_elements, 
					subtile_elements, 
					sweep_grid_size),
				sweep_grid_size(sweep_grid_size),
				spine_elements(spine_elements),
				problem_storage(problem_storage) {};
	};


public:
		
	//---------------------------------------------------------------------
	// Granularity Parameterization
	//---------------------------------------------------------------------
		
	/**
	 * Config configuration that is common across all three kernels in a 
	 * sorting pass (upsweep, spinescan, and downsweep).  This C++ type encapsulates 
	 * all three sets of kernel-tuning parameters (they are reflected via the static 
	 * fields). By deriving from the three granularity types, we assure operational 
	 * consistency over an entire sorting pass. 
	 */
	template <
		// Common
		typename StorageType,
		int RADIX_BITS,
		int LOG_SUBTILE_ELEMENTS,
		CacheModifier CACHE_MODIFIER,
		
		// Upsweep
		int UPSWEEP_CTA_OCCUPANCY,
		int UPSWEEP_LOG_THREADS,
		int UPSWEEP_LOG_LOAD_VEC_SIZE,
		int UPSWEEP_LOG_LOADS_PER_TILE,
		
		// Spine-scan
		int SPINE_CTA_OCCUPANCY,
		int SPINE_LOG_THREADS,
		int SPINE_LOG_LOAD_VEC_SIZE,
		int SPINE_LOG_LOADS_PER_TILE,
		int SPINE_LOG_RAKING_THREADS,

		// Downsweep
		int DOWNSWEEP_CTA_OCCUPANCY,
		int DOWNSWEEP_LOG_THREADS,
		int DOWNSWEEP_LOG_LOAD_VEC_SIZE,
		int DOWNSWEEP_LOG_LOADS_PER_CYCLE,
		int DOWNSWEEP_LOG_CYCLES_PER_TILE,
		int DOWNSWEEP_LOG_RAKING_THREADS>

	struct MultiCtaConfig
	{
		typedef StorageType 						Storage;
		
		typedef SortingCtaDecomposition<Storage>	Decomposition;
		
		typedef UpsweepConfig<
			ConvertedKeyType, 
			typename StorageType::IndexType,
			RADIX_BITS, 
			LOG_SUBTILE_ELEMENTS,
			UPSWEEP_CTA_OCCUPANCY,  
			UPSWEEP_LOG_THREADS,
			UPSWEEP_LOG_LOAD_VEC_SIZE,  	
			UPSWEEP_LOG_LOADS_PER_TILE,
			CACHE_MODIFIER> 						Upsweep;
		
		typedef SpineScanConfig<
			typename StorageType::IndexType,
			SPINE_CTA_OCCUPANCY,
			SPINE_LOG_THREADS,
			SPINE_LOG_LOAD_VEC_SIZE,
			SPINE_LOG_LOADS_PER_TILE,
			SPINE_LOG_RAKING_THREADS,
			CACHE_MODIFIER> 						SpineScan;
		
		typedef DownsweepConfig<
			ConvertedKeyType,
			ValueType,
			typename StorageType::IndexType,
			RADIX_BITS,
			LOG_SUBTILE_ELEMENTS,
			DOWNSWEEP_CTA_OCCUPANCY,
			DOWNSWEEP_LOG_THREADS,
			DOWNSWEEP_LOG_LOAD_VEC_SIZE,
			DOWNSWEEP_LOG_LOADS_PER_CYCLE,
			DOWNSWEEP_LOG_CYCLES_PER_TILE,
			DOWNSWEEP_LOG_RAKING_THREADS,
			CACHE_MODIFIER> 						Downsweep;
	};
		
protected:
	
	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------
	
	// Default size (in bytes) of spine storage.  (600 CTAs x 16 digits each (4 radix bits) x 4-byte-digits)
	static const int DEFAULT_SPINE_BYTES = 600 * (1 << 4) * sizeof(int);	
	
	// Temporary device storage needed for scanning digit histograms produced
	// by separate CTAs
	void *d_spine;
	
	// Number of bytes backed by d_spine 
	int spine_bytes;
	

	//-----------------------------------------------------------------------------
	// Utility Routines
	//-----------------------------------------------------------------------------
	
	/**
	 * Utility function: Determines the number of spine elements needed 
	 * for a given grid size, rounded up to the nearest spine tile
	 */
	static int SpineElements(int sweep_grid_size, int radix_bits, int spine_tile_elements) 
	{
		int spine_elements = sweep_grid_size * (1 << radix_bits);
		int spine_tiles = (spine_elements + spine_tile_elements - 1) / spine_tile_elements;
		return spine_tiles * spine_tile_elements;
	}

	
	//-----------------------------------------------------------------------------
	// Construction
	//-----------------------------------------------------------------------------
	
	/**
	 * Constructor.
	 */
	MultiCtaLsbSortEnactor(const CudaProperties &props = CudaProperties()) : 
		Base::BaseLsbSortEnactor(props),  
		d_spine(NULL),
		spine_bytes(DEFAULT_SPINE_BYTES)
	{
		// Allocate the spine for the maximum number of spine elements we might have
		cudaMalloc((void**) &d_spine, spine_bytes);
	    dbg_perror_exit("MultiCtaLsbSortEnactor:: cudaMalloc d_spine failed: ", __FILE__, __LINE__);
	}


	//-----------------------------------------------------------------------------
	// Sorting Operation 
	//-----------------------------------------------------------------------------
	
    /**
     * Pre-sorting logic.
     */
	template <typename StorageType>
    cudaError_t PreSort(StorageType &problem_storage, int problem_spine_elements) 
    {
    	// Allocate device memory for temporary storage (if necessary)
    	if (problem_storage.d_keys[0] == NULL) {
    		cudaMalloc((void**) &problem_storage.d_keys[0], problem_storage.num_elements * sizeof(KeyType));
    	    dbg_perror_exit("MultiCtaLsbSortEnactor:: cudaMalloc problem_storage.d_keys[0] failed: ", __FILE__, __LINE__);
    	}
    	if (problem_storage.d_keys[1] == NULL) {
    		cudaMalloc((void**) &problem_storage.d_keys[1], problem_storage.num_elements * sizeof(KeyType));
    	    dbg_perror_exit("MultiCtaLsbSortEnactor:: cudaMalloc problem_storage.d_keys[1] failed: ", __FILE__, __LINE__);
    	}
    	if (!Base::KeysOnly()) {
    		if (problem_storage.d_values[0] == NULL) {
    			cudaMalloc((void**) &problem_storage.d_values[0], problem_storage.num_elements * sizeof(ValueType));
    		    dbg_perror_exit("MultiCtaLsbSortEnactor:: cudaMalloc problem_storage.d_values[0] failed: ", __FILE__, __LINE__);
    		}
    		if (problem_storage.d_values[1] == NULL) {
    			cudaMalloc((void**) &problem_storage.d_values[1], problem_storage.num_elements * sizeof(ValueType));
    		    dbg_perror_exit("MultiCtaLsbSortEnactor:: cudaMalloc problem_storage.d_values[1] failed: ", __FILE__, __LINE__);
    		}
    	}
    	
    	// Make sure our spine is big enough
    	int problem_spine_bytes = problem_spine_elements * sizeof(typename StorageType::IndexType);
    	
    	if (problem_spine_bytes > spine_bytes) {
   			cudaFree(d_spine);
    	    dbg_perror_exit("MultiCtaLsbSortEnactor:: cudaFree d_spine failed: ", __FILE__, __LINE__);
    	    
    	    spine_bytes = problem_spine_bytes;
    	    
    		cudaMalloc((void**) &d_spine, spine_bytes);
    	    dbg_perror_exit("MultiCtaLsbSortEnactor:: cudaMalloc d_spine failed: ", __FILE__, __LINE__);
    	}

    	return cudaSuccess;
    }
    
	
    /**
     * Post-sorting logic.
     */
	template <typename Storage> 
    cudaError_t PostSort(Storage &problem_storage, int passes) 
    {
    	return cudaSuccess;
    }
	
    
public:

	//-----------------------------------------------------------------------------
	// Construction
	//-----------------------------------------------------------------------------
    
    /**
     * Destructor
     */
    virtual ~MultiCtaLsbSortEnactor() 
    {
   		if (d_spine) {
   			cudaFree(d_spine);
    	    dbg_perror_exit("MultiCtaLsbSortEnactor:: cudaFree d_spine failed: ", __FILE__, __LINE__);
   		}
    }
    
};


}// namespace b40c

