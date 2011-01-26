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
 * LSB Sorting Base Class
 ******************************************************************************/

#pragma once

#include "b40c_kernel_utils.cu"
#include "radixsort_kernel_common.cu"

namespace b40c {

using namespace lsb_radix_sort;



/**
 * Base class for LSB radix sorting enactors.
 */
template <typename KeyType, typename ValueType, typename StorageType>
class BaseLsbSortEnactor 
{
	
protected:

	/**
	 * Whether or not this instance can be used to sort satellite values
	 */
	static bool KeysOnly() 
	{
		return IsKeysOnly<ValueType>();
	}

protected:

	// Device properties
	const CudaProperties cuda_props;
	
public: 
	
	// Prints sorting debug detail to stdout
	bool DEBUG;
	
protected: 	
	
	/**
	 * Constructor.
	 */
	BaseLsbSortEnactor(const CudaProperties &props = CudaProperties()) : 
		cuda_props(props), DEBUG(false) {}


public:
	

	/**
     * Destructor
     */
    virtual ~BaseLsbSortEnactor() {}

    
	/**
	 * Enacts a radix sorting operation on the specified device data.
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
	virtual cudaError_t EnactSort(StorageType &problem_storage) = 0;	
    
};


} // namespace b40c

