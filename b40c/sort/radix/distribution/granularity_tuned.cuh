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
#include "radixsort_api_granularity.cuh"

namespace b40c {
namespace radix_sort {
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
	typename SizeT>
struct TunedConfig :
	TunedConfig<FamilyClassifier<CUDA_ARCH>::FAMILY, KeyType, ValueType, SizeT> {};



//-----------------------------------------------------------------------------
// SM2.0 default granularity parameterization type
//-----------------------------------------------------------------------------

		4, 10, NONE, true, false, true, 
		8, 7, 0, 2, 
		1, 7, 2, 0, 5, 
		8, 6, 2, 1, 1, 6> 



//-----------------------------------------------------------------------------
// SM1.3 default granularity parameterization type
//-----------------------------------------------------------------------------

		4, 9, NONE, true, true, true, 
		5, 7, 1, 0, 
		1, 7, 2, 0, 5, 
		5, 6, 2, 1, 0, 5



//-----------------------------------------------------------------------------
// SM1.0 default granularity parameterization type
//-----------------------------------------------------------------------------

		4, 9, NONE, true, false, true,
		3, 7, 0, 0, 
		1, 7, 2, 0, 5,
		2, 7, 1, 1, 1, 7


}// namespace large_problem_tuning
}// namespace radix_sort
}// namespace b40c

