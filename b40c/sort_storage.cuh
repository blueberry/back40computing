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
 *  Storage-wrapper / problem-descriptor for sorting operations  
 ******************************************************************************/

#include <b40c/radix_sort/sorting_utils.cuh>

#pragma once

namespace b40c {

/**
 * Base storage management structure for multi-CTA-sorting device vectors.
 * 
 * Everything is public in these structures.  They're just a simple transparent 
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
 * E.g., consider a SortStorageMultiCta "device_storage".  The valid data 
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
template <
	typename _KeyType, 
	typename _ValueType 	= radix_sort::KeysOnly, 	// Default keys-only sorting
	typename _SizeT 		= unsigned int>				// Default can sort up to 2^32 elements
struct SortStorageMultiCta
{
	typedef _KeyType KeyType;
	typedef _ValueType ValueType;
	typedef _SizeT SizeT;				// Integer type suitable for indexing storage elements (e.g., int, long long, etc.)

	// Pair of device vector pointers for keys
	KeyType* d_keys[2];
	
	// Pair of device vector pointers for values
	ValueType* d_values[2];

	// Number of elements for sorting in the above vectors 
	SizeT num_elements;
	
	// Selector into the pair of device vector pointers indicating valid 
	// sorting elements (i.e., where the results are)
	int selector;

	// Constructor
	SortStorageMultiCta() 
	{
		num_elements = 0;
		selector = 0;
		d_keys[0] = NULL;
		d_keys[1] = NULL;
		d_values[0] = NULL;
		d_values[1] = NULL;
	}
	
	// Constructor
	SortStorageMultiCta(int num_elements)  
	{
		this->num_elements = num_elements;
		selector = 0;
		d_keys[0] = NULL;
		d_keys[1] = NULL;
		d_values[0] = NULL;
		d_values[1] = NULL;
	}

	// Constructor
	SortStorageMultiCta(
		int num_elements, 
		KeyType* d_keys, 
		ValueType* d_values = NULL)
	{
		this->num_elements = num_elements;
		selector = 0;
		this->d_keys[0] = d_keys;
		d_keys[1] = NULL;
		this->d_values[0] = d_values;
		d_values[1] = NULL;
	}
};




}// namespace b40c

