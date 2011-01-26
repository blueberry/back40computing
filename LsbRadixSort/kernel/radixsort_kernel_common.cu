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
 * Kernel types and subroutines utilities that are common across all B40C 
 * LSB radix sorting kernels  
 ******************************************************************************/

#pragma once

#include <b40c_kernel_utils.cu>
#include <b40c_vector_types.cu>
#include <b40c_kernel_data_movement.cu>

#include <radixsort_key_conversion.cu>

namespace b40c {

namespace lsb_radix_sort {


/******************************************************************************
 * Value type for keys-only sorting 
 ******************************************************************************/

/**
 * Value-type structure denoting keys-only sorting
 */
struct KeysOnly {};


/**
 * Returns whether or not the templated type indicates keys-only sorting
 */
template <typename V>
__forceinline__ __host__ __device__ bool IsKeysOnly() {return false;}


/**
 * Returns whether or not the templated type indicates keys-only sorting
 */
template <>
__forceinline__ __host__ __device__ bool IsKeysOnly<KeysOnly>() {return true;}



/******************************************************************************
 * Bitfield extraction kernel subroutines
 ******************************************************************************/

/**
 * Extracts a bit field from source and places the zero or sign-extended result 
 * in extract
 */
template <typename T, int BIT_START, int NUM_BITS> 
struct ExtractKeyBits 
{
	__device__ __forceinline__ static void Extract(int &bits, const T &source) 
	{
#if __CUDA_ARCH__ >= 200
		asm("bfe.u32 %0, %1, %2, %3;" : "=r"(bits) : "r"(source), "n"(BIT_START), "n"(NUM_BITS));
#else 
		const T MASK = (1 << NUM_BITS) - 1;
		bits = (source >> BIT_START) & MASK;
#endif
	}
};
	
/**
 * Extracts a bit field from source and places the zero or sign-extended result 
 * in extract
 */
template <int BIT_START, int NUM_BITS> 
struct ExtractKeyBits<unsigned long long, BIT_START, NUM_BITS> 
{
	__device__ __forceinline__ static void Extract(int &bits, const unsigned long long &source) 
	{
		const unsigned long long MASK = (1 << NUM_BITS) - 1;
		bits = (source >> BIT_START) & MASK;
	}
};
	


} // namespace lsb_radix_sort

} // namespace b40c

