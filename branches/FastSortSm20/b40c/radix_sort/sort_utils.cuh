/******************************************************************************
 * 
 * Copyright (c) 2010-2012, Duane Merrill.  All rights reserved.
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
 * Common utilities for radix sorting
 ******************************************************************************/

#pragma once

#include <functional>
#include <b40c/util/device_intrinsics.cuh>
#include <b40c/util/ns_umbrella.cuh>

B40C_NS_PREFIX
namespace b40c {
namespace radix_sort {


/******************************************************************************
 * Bit-field extraction utilities
 ******************************************************************************/

/**
 * Bitfield-extract, left-shift
 */
template <int BIT_OFFSET, int NUM_BITS, int LEFT_SHIFT, typename T>
__device__ __forceinline__ unsigned int Extract(
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
 * Bitfield-extract, left-shift, add
 */
template <int BIT_OFFSET, int NUM_BITS, int LEFT_SHIFT, typename T>
__device__ __forceinline__ unsigned int Extract(
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

/**
 * Bitfield-extract, left-shift (64-bit)
 */
template <int BIT_OFFSET, int NUM_BITS, int LEFT_SHIFT>
__device__ __forceinline__ unsigned int Extract(
	unsigned long long source)
{
	const unsigned long long MASK = ((1ull << NUM_BITS) - 1) << BIT_OFFSET;
	const int SHIFT = LEFT_SHIFT - BIT_OFFSET;

	unsigned long long bits = (source & MASK);
	return util::MagnitudeShift<SHIFT>::Shift(bits);
}

/**
 * Bitfield-extract, left-shift, add (64-bit)
 */
template <int BIT_OFFSET, int NUM_BITS, int LEFT_SHIFT>
__device__ __forceinline__ unsigned int Extract(
	unsigned long long source,
	unsigned int addend)
{
	return Extract<BIT_OFFSET, NUM_BITS, LEFT_SHIFT>(source) + addend;
}





/******************************************************************************
 * Traits for converting for converting signed and floating point types
 * to unsigned types suitable for radix sorting
 ******************************************************************************/


/**
 * Specialization for unsigned signed integers
 */
template <typename _ConvertedKeyType>
struct UnsignedKeyTraits
{
	typedef _ConvertedKeyType 	ConvertedKeyType;

	struct IngressOp : std::unary_function<ConvertedKeyType, ConvertedKeyType>
	{
		__device__ __forceinline__ ConvertedKeyType operator()(ConvertedKeyType key)
		{
			return key;
		}
	};

	struct EgressOp : std::unary_function<ConvertedKeyType, ConvertedKeyType>
	{
		__device__ __host__ __forceinline__ ConvertedKeyType operator()(ConvertedKeyType converted_key)
		{
			return converted_key;
		}
	};
};


/**
 * Specialization for signed integers
 */
template <typename _ConvertedKeyType>
struct SignedKeyTraits
{
	typedef _ConvertedKeyType 	ConvertedKeyType;

	static const ConvertedKeyType HIGH_BIT = ConvertedKeyType(1) << ((sizeof(ConvertedKeyType) * 8) - 1);

	struct IngressOp : std::unary_function<ConvertedKeyType, ConvertedKeyType>
	{
		__device__ __forceinline__ ConvertedKeyType operator()(ConvertedKeyType key)
		{
			return key ^ HIGH_BIT;
		}
	};

	struct EgressOp : std::unary_function<ConvertedKeyType, ConvertedKeyType>
	{
		__device__ __host__ __forceinline__ ConvertedKeyType operator()(ConvertedKeyType converted_key)
		{
			return converted_key ^ HIGH_BIT;
		}
	};
};


/**
 * Specialization for floating point
 */
template <typename _ConvertedKeyType>
struct FloatKeyTraits
{
	typedef _ConvertedKeyType 	ConvertedKeyType;

	static const ConvertedKeyType HIGH_BIT = ConvertedKeyType(1) << ((sizeof(ConvertedKeyType) * 8) - 1);

	struct IngressOp : std::unary_function<ConvertedKeyType, ConvertedKeyType>
	{
		__device__ __forceinline__ ConvertedKeyType operator()(ConvertedKeyType key)
		{
			ConvertedKeyType mask = (key & HIGH_BIT) ? ConvertedKeyType(-1) : HIGH_BIT;
			return key ^ mask;
		}
	};

	struct EgressOp : std::unary_function<ConvertedKeyType, ConvertedKeyType>
	{
		__device__ __host__ __forceinline__ ConvertedKeyType operator()(ConvertedKeyType converted_key)
		{
			ConvertedKeyType mask = (converted_key & HIGH_BIT) ? HIGH_BIT : ConvertedKeyType(-1);
			return converted_key ^ mask;
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
} // namespace b40c

