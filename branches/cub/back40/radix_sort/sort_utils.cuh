/******************************************************************************
 * 
 * Copyright (c) 2011-2012, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2012, NVIDIA CORPORATION.  All rights reserved.
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
 ******************************************************************************/

/******************************************************************************
 * Types and subroutines utilities that are common across all B40C LSB radix 
 * sorting kernels and host enactors  
 ******************************************************************************/

#pragma once

#include <cub/cub.cuh>

namespace back40 {
namespace radix_sort {


/******************************************************************************
 * Bit-field extraction kernel subroutines
 ******************************************************************************/

/**
 * Bit extraction, specialized for non-64bit key types
 */
template <
	typename T,
	int BIT_OFFSET,
	int NUM_BITS,
	int LEFT_SHIFT>
struct Extract
{
	/**
	 * Super bitfield-extract (BFE, then left-shift).
	 */
	__device__ __forceinline__ static unsigned int SuperBFE(
		T source)
	{
		const T MASK = ((1ull << NUM_BITS) - 1) << BIT_OFFSET;
		const int SHIFT = LEFT_SHIFT - BIT_OFFSET;

		T bits = (source & MASK);
		if (SHIFT == 0) {
			return bits;
		} else {
			return util::MagnitudeShift<SHIFT>::Shift(bits);
		}
	}

	/**
	 * Super bitfield-extract (BFE, then left-shift, then add).
	 */
	__device__ __forceinline__ static unsigned int SuperBFE(
		T source,
		unsigned int addend)
	{
		const T MASK = ((1ull << NUM_BITS) - 1) << BIT_OFFSET;
		const int SHIFT = LEFT_SHIFT - BIT_OFFSET;

		T bits = (source & MASK);
		if (SHIFT == 0) {
			return bits + addend;
		} else {
			bits = (SHIFT > 0) ?
				(util::SHL_ADD(bits, SHIFT, addend)) :
				(util::SHR_ADD(bits, SHIFT * -1, addend));
			return bits;
		}
	}

};


/**
 * Bit extraction, specialized for 64bit key types
 */
template <
	int BIT_OFFSET,
	int NUM_BITS,
	int LEFT_SHIFT>
struct Extract<unsigned long long, BIT_OFFSET, NUM_BITS, LEFT_SHIFT>
{
	/**
	 * Super bitfield-extract (BFE, then left-shift).
	 */
	__device__ __forceinline__ static unsigned int SuperBFE(
		unsigned long long source)
	{
		const unsigned long long MASK = ((1ull << NUM_BITS) - 1) << BIT_OFFSET;
		const int SHIFT = LEFT_SHIFT - BIT_OFFSET;

		unsigned long long bits = (source & MASK);
		return util::MagnitudeShift<SHIFT>::Shift(bits);
	}

	/**
	 * Super bitfield-extract (BFE, then left-shift, then add).
	 */
	__device__ __forceinline__ static unsigned int SuperBFE(
		unsigned long long source,
		unsigned int addend)
	{
		return SuperBFE(source) + addend;
	}
};




/******************************************************************************
 * Traits for converting for converting signed and floating point types
 * to unsigned types suitable for radix sorting
 ******************************************************************************/


/**
 * Specialization for unsigned signed integers
 */
template <typename UnsignedBits>
struct UnsignedKeyTraits
{
	typedef UnsignedBits ConvertedKeyType;

	static const bool MUST_APPLY = false;

	struct IngressOp
	{
		__device__ __forceinline__ UnsignedBits operator()(UnsignedBits key)
		{
			return key;
		}
	};

	struct EgressOp
	{
		__device__ __host__ __forceinline__ UnsignedBits operator()(UnsignedBits key)
		{
			return key;
		}
	};
};


/**
 * Specialization for signed integers
 */
template <typename UnsignedBits>
struct SignedKeyTraits
{
	typedef UnsignedBits ConvertedKeyType;

	static const bool MUST_APPLY 			= true;
	static const UnsignedBits HIGH_BIT 		= ((UnsignedBits) 0x1) << ((sizeof(UnsignedBits) * 8) - 1);

	struct IngressOp
	{
		__device__ __forceinline__ UnsignedBits operator()(UnsignedBits key)
		{
			return key ^ HIGH_BIT;
		}
	};

	struct EgressOp
	{
		__device__ __host__ __forceinline__ UnsignedBits operator()(UnsignedBits key)
		{
			return key ^ HIGH_BIT;
		}
	};
};


/**
 * Specialization for floating point
 */
template <typename UnsignedBits>
struct FloatKeyTraits
{
	typedef UnsignedBits ConvertedKeyType;

	static const bool MUST_APPLY 			= true;
	static const UnsignedBits HIGH_BIT 		= ((UnsignedBits) 0x1) << ((sizeof(UnsignedBits) * 8) - 1);

	struct IngressOp
	{
		__device__ __forceinline__ UnsignedBits operator()(UnsignedBits key)
		{
			UnsignedBits mask = (key & HIGH_BIT) ? (UnsignedBits) -1 : HIGH_BIT;
			return key ^ mask;
		}
	};

	struct EgressOp
	{
		__device__ __host__ __forceinline__ UnsignedBits operator()(UnsignedBits key)
		{
			UnsignedBits mask = (key & HIGH_BIT) ? HIGH_BIT : (UnsignedBits) -1;
			return key ^ mask;
		}
	};
};




// Default unsigned types
template <typename T>
struct KeyTraits : UnsignedKeyTraits<T> {};

// char
template <> struct KeyTraits<char> : SignedKeyTraits<unsigned char> {};

// signed char
template <> struct KeyTraits<signed char> : SignedKeyTraits<unsigned char> {};

// short
template <> struct KeyTraits<short> : SignedKeyTraits<unsigned short> {};

// int
template <> struct KeyTraits<int> : SignedKeyTraits<unsigned int> {};

// long
template <> struct KeyTraits<long> : SignedKeyTraits<unsigned long> {};

// long long
template <> struct KeyTraits<long long> : SignedKeyTraits<unsigned long long> {};

// float
template <> struct KeyTraits<float> : FloatKeyTraits<unsigned int> {};

// double
template <> struct KeyTraits<double> : FloatKeyTraits<unsigned long long> {};




} // namespace radix_sort
} // namespace back40

