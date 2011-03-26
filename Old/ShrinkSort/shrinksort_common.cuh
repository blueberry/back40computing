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
 * Types and subroutines utilities that are common across all B40C LSB radix 
 * sorting kernels and host enactors  
 ******************************************************************************/

#pragma once

#include "b40c_cuda_properties.cuh"

namespace b40c {
namespace shrink_sort {


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
 * Bit-field extraction kernel subroutines
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
	



/******************************************************************************
 * Traits for converting  for converting signed and floating point types to unsigned types
 * suitable for radix sorting  
 ******************************************************************************/

//-----------------------------------------------------------------------------
// Key-conversion operations.  If out of range, we assign the coverted key
// to all 1's (i.e., it will put it at the end)
//-----------------------------------------------------------------------------

template <typename UnsignedBits> 
struct UnsignedIntegerKeyConversion 
{
	typedef UnsignedBits ConvertedKeyType;
	
	static const bool MustApply = false;		// We may early-exit this pass

	__device__ __host__ __forceinline__ static void Preprocess(UnsignedBits &converted_key, bool in_range) 
	{
		if (!in_range) {
			converted_key = (UnsignedBits) -1;
		}
	}

	__device__ __host__ __forceinline__ static void Postprocess(UnsignedBits &converted_key) {}  
};


template <typename UnsignedBits> 
struct SignedIntegerKeyConversion 
{
	typedef UnsignedBits ConvertedKeyType;

	static const bool MustApply = true;		// We must not early-exit this pass (conversion necessary)

	__device__ __host__ __forceinline__ static void Preprocess(UnsignedBits &converted_key, bool in_range) 
	{
		if (in_range) {
			const UnsignedBits HIGH_BIT = ((UnsignedBits) 0x1) << ((sizeof(UnsignedBits) * 8) - 1);
			converted_key ^= HIGH_BIT;
		} else {
			converted_key = (UnsignedBits) -1;
		}
	}

	__device__ __host__ __forceinline__ static void Postprocess(UnsignedBits &converted_key)  
	{
		const UnsignedBits HIGH_BIT = ((UnsignedBits) 0x1) << ((sizeof(UnsignedBits) * 8) - 1);
		converted_key ^= HIGH_BIT;	
	}
};


template <typename UnsignedBits> 
struct FloatingPointKeyConversion 
{
	typedef UnsignedBits ConvertedKeyType;

	static const bool MustApply = true;		// We must not early-exit this pass (conversion necessary)

	__device__ __host__ __forceinline__ static void Preprocess(UnsignedBits &converted_key, bool in_range) 
	{
		if (in_range) {
			const UnsignedBits HIGH_BIT = ((UnsignedBits) 0x1) << ((sizeof(UnsignedBits) * 8) - 1);
			UnsignedBits mask = (converted_key & HIGH_BIT) ? (UnsignedBits) -1 : HIGH_BIT; 
			converted_key ^= mask;
		} else {
			converted_key = (UnsignedBits) -1;
		}
	}

	__device__ __host__ __forceinline__ static void Postprocess(UnsignedBits &converted_key) 
	{
		const UnsignedBits HIGH_BIT = ((UnsignedBits) 0x1) << ((sizeof(UnsignedBits) * 8) - 1);
		UnsignedBits mask = (converted_key & HIGH_BIT) ? HIGH_BIT : (UnsignedBits) -1; 
		converted_key ^= mask;
    }
};


//-----------------------------------------------------------------------------
// Key traits that dictate unsigned-integer-conversion-types the corresponding
// conversion operations
//-----------------------------------------------------------------------------

// Default unsigned types
template <typename T> struct KeyTraits : UnsignedIntegerKeyConversion<T> {};

// char
template <> struct KeyTraits<char> : SignedIntegerKeyConversion<unsigned char> {};

// signed char
template <> struct KeyTraits<signed char> : SignedIntegerKeyConversion<unsigned char> {};

// short
template <> struct KeyTraits<short> : SignedIntegerKeyConversion<unsigned short> {};

// int
template <> struct KeyTraits<int> : SignedIntegerKeyConversion<unsigned int> {};

// long
template <> struct KeyTraits<long> : SignedIntegerKeyConversion<unsigned long> {};

// long long
template <> struct KeyTraits<long long> : SignedIntegerKeyConversion<unsigned long long> {};

// float
template <> struct KeyTraits<float> : FloatingPointKeyConversion<unsigned int> {};

// double
template <> struct KeyTraits<double> : FloatingPointKeyConversion<unsigned long long> {};




} // namespace shrink_sort

} // namespace b40c

