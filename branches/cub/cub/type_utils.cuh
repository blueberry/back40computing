/******************************************************************************
 * 
 * Copyright 2010-2012 Duane Merrill
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
 * For more information, see our Google Code project site: 
 * http://code.google.com/p/back40computing/
 * 
 * Thanks!
 * 
 ******************************************************************************/

/******************************************************************************
 * Common CUB utility routines
 ******************************************************************************/

#pragma once

namespace cub {


/******************************************************************************
 * Macro utilities
 ******************************************************************************/

/**
 * Select maximum
 */
#define CUB_MAX(a, b) (((a) > (b)) ? (a) : (b))


/**
 * Select maximum
 */
#define CUB_MIN(a, b) (((a) < (b)) ? (a) : (b))


/**
 * x rounded up to the nearest multiple of y
 */
#define CUB_ROUND_UP_NEAREST(x, y) ((((x) + (y) - 1) / (y)) * y)


/******************************************************************************
 * Simple templated utilities
 ******************************************************************************/

/**
 * Perform a swap
 */
template <typename T> 
void __host__ __device__ __forceinline__ Swap(T &a, T &b) {
	T temp = a;
	a = b;
	b = temp;
}


/**
 * MagnitudeShift().  Allows you to shift left for positive magnitude values, 
 * right for negative.   
 */
template <int MAGNITUDE, int shift_left = (MAGNITUDE >= 0)>
struct MagnitudeShift
{
	template <typename K>
	__device__ __forceinline__ static K Shift(K key)
	{
		return key << MAGNITUDE;
	}
};


template <int MAGNITUDE>
struct MagnitudeShift<MAGNITUDE, 0>
{
	template <typename K>
	__device__ __forceinline__ static K Shift(K key)
	{
		return key >> (MAGNITUDE * -1);
	}
};


/******************************************************************************
 * Metaprogramming Utilities
 ******************************************************************************/

/**
 * Null type
 */
struct NullType {};


/**
 * Statically determine log2(N), rounded up, e.g.,
 * 		Log2<8>::VALUE == 3
 * 		Log2<3>::VALUE == 2
 */
template <int N, int CURRENT_VAL = N, int COUNT = 0>
struct Log2
{
	// Inductive case
	static const int VALUE = Log2<N, (CURRENT_VAL >> 1), COUNT + 1>::VALUE;
};

template <int N, int COUNT>
struct Log2<N, 0, COUNT>
{
	// Base case
	static const int VALUE = (1 << (COUNT - 1) < N) ?
		COUNT :
		COUNT - 1;
};


/**
 * If/Then/Else
 */
template <bool IF, typename ThenType, typename ElseType>
struct If
{
	// true
	typedef ThenType Type;
};

template <typename ThenType, typename ElseType>
struct If<false, ThenType, ElseType>
{
	// false
	typedef ElseType Type;
};


/**
 * Equals 
 */
template <typename A, typename B>
struct Equals
{
	enum {
		VALUE = 0,
		NEGATE = 1
	};
};

template <typename A>
struct Equals <A, A>
{
	enum {
		VALUE = 1,
		NEGATE = 0
	};
};



/**
 * Is volatile
 */
template <typename Tp>
struct IsVolatile
{
	enum { VALUE = 0 };
};
template <typename Tp>
struct IsVolatile<Tp volatile>
{
	enum { VALUE = 1 };
};



/**
 * Removes const and volatile qualifiers from type Tp
 */
template <typename Tp, typename Up = Tp>
struct RemoveQualifiers
{
	typedef Up Type;
};

template <typename Tp, typename Up>
struct RemoveQualifiers<Tp, volatile Up>
{
	typedef Up Type;
};

template <typename Tp, typename Up>
struct RemoveQualifiers<Tp, const Up>
{
	typedef Up Type;
};

template <typename Tp, typename Up>
struct RemoveQualifiers<Tp, const volatile Up>
{
	typedef Up Type;
};



} // namespace cub

