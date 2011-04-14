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
 * For more information, see our Google Code project site:
 * http://code.google.com/p/back40computing/
 *
 ******************************************************************************/

/******************************************************************************
 * Simple tuple types for assisting AOS <-> SOA work
 ******************************************************************************/

#pragma once

#include <b40c/util/basic_utils.cuh>

namespace b40c {
namespace util {


template <
	typename T0 = NullType,
	typename T1 = NullType,
	typename T2 = NullType,
	typename T3 = NullType>
struct Tuple;

/**
 * 1 element tuple
 */
template <typename _T0>
struct Tuple<_T0, NullType, NullType, NullType>
{
	enum {
		NUM_FIELDS = 1
	};

	// Typedefs
	typedef _T0 T0;

	// Fields
	T0 t0;

	// Constructors
	__host__ __device__ __forceinline__ Tuple() {}
	__host__ __device__ __forceinline__ Tuple(T0 t0) : t0(t0) {}

	// Manipulators

	template <typename TupleSlice>
	__host__ __device__ __forceinline__ void Set(const TupleSlice &tuple, const int offset)
	{
		t0[offset] = tuple.t0;
	}

	template <int LANE, typename TupleSlice>
	__host__ __device__ __forceinline__ void Set(const TupleSlice &tuple, const int offset)
	{
		t0[LANE][offset] = tuple.t0;
	}

	template <typename TupleSlice>
	__host__ __device__ __forceinline__ TupleSlice Get(const int offset) const
	{
		return TupleSlice(t0[offset]);
	}

	template <int LANE, typename TupleSlice>
	__host__ __device__ __forceinline__ TupleSlice Get(const int offset) const
	{
		return TupleSlice(t0[LANE][offset]);
	}
};


/**
 * 2 element tuple
 */
template <typename _T0, typename _T1>
struct Tuple<_T0, _T1, NullType, NullType>
{
	enum {
		NUM_FIELDS = 2
	};

	// Typedefs
	typedef _T0 T0;
	typedef _T1 T1;

	// Fields
	T0 t0;
	T1 t1;


	// Constructors
	__host__ __device__ __forceinline__ Tuple() {}
	__host__ __device__ __forceinline__ Tuple(T0 t0) : t0(t0) {}
	__host__ __device__ __forceinline__ Tuple(T0 t0, T1 t1) : t0(t0), t1(t1) {}

	// Manipulators

	template <typename TupleSlice>
	__host__ __device__ __forceinline__ void Set(const TupleSlice &tuple, const int offset)
	{
		t0[offset] = tuple.t0;
		t1[offset] = tuple.t1;
	}

	template <int LANE, typename TupleSlice>
	__host__ __device__ __forceinline__ void Set(const TupleSlice &tuple, const int offset)
	{
		t0[LANE][offset] = tuple.t0;
		t1[LANE][offset] = tuple.t1;
	}

	template <typename TupleSlice>
	__host__ __device__ __forceinline__ TupleSlice Get(const int offset) const
	{
		return TupleSlice(t0[offset], t1[offset]);
	}

	template <int LANE, typename TupleSlice>
	__host__ __device__ __forceinline__ TupleSlice Get(const int offset) const
	{
		return TupleSlice(t0[LANE][offset], t1[LANE][offset]);
	}
};


/**
 * 3 element tuple
 */
template <typename _T0, typename _T1, typename _T2>
struct Tuple<_T0, _T1, _T2, NullType>
{
	enum {
		NUM_FIELDS = 3
	};

	// Typedefs
	typedef _T0 T0;
	typedef _T1 T1;
	typedef _T2 T2;

	// Fields
	T0 t0;
	T1 t1;
	T2 t2;

	// Constructor
	__host__ __device__ __forceinline__ Tuple(T0 t0, T1 t1, T2 t2) : t0(t0), t1(t1), t2(t2) {}

	// Manipulators

	template <typename TupleSlice>
	__host__ __device__ __forceinline__ void Set(const TupleSlice &tuple, const int offset)
	{
		t0[offset] = tuple.t0;
		t1[offset] = tuple.t1;
		t2[offset] = tuple.t2;
	}

	template <int LANE, typename TupleSlice>
	__host__ __device__ __forceinline__ void Set(const TupleSlice &tuple, const int offset)
	{
		t0[LANE][offset] = tuple.t0;
		t1[LANE][offset] = tuple.t1;
		t2[LANE][offset] = tuple.t2;
	}

	template <typename TupleSlice>
	__host__ __device__ __forceinline__ TupleSlice Get(const int offset) const
	{
		return TupleSlice(
			t0[offset],
			t1[offset],
			t2[offset]);
	}

	template <int LANE, typename TupleSlice>
	__host__ __device__ __forceinline__ TupleSlice Get(const int offset) const
	{
		return TupleSlice(
			t0[LANE][offset],
			t1[LANE][offset],
			t2[LANE][offset]);
	}
};


/**
 * 4 element tuple
 */
template <typename _T0, typename _T1, typename _T2, typename _T3>
struct Tuple
{
	enum {
		NUM_FIELDS = 4
	};

	// Typedefs
	typedef _T0 T0;
	typedef _T1 T1;
	typedef _T2 T2;
	typedef _T3 T3;

	// Fields
	T0 t0;
	T1 t1;
	T2 t2;
	T3 t3;

	// Constructor
	__host__ __device__ __forceinline__ Tuple(T0 t0, T1 t1, T2 t2, T3 t3) : t0(t0), t1(t1), t2(t2), t3(t3) {}

	// Manipulators

	template <typename TupleSlice>
	__host__ __device__ __forceinline__ void Set(const TupleSlice &tuple, const int offset)
	{
		t0[offset] = tuple.t0;
		t1[offset] = tuple.t1;
		t2[offset] = tuple.t2;
		t3[offset] = tuple.t3;
	}

	template <int LANE, typename TupleSlice>
	__host__ __device__ __forceinline__ void Set(const TupleSlice &tuple, const int offset)
	{
		t0[LANE][offset] = tuple.t0;
		t1[LANE][offset] = tuple.t1;
		t2[LANE][offset] = tuple.t2;
		t3[LANE][offset] = tuple.t3;
	}

	template <typename TupleSlice>
	__host__ __device__ __forceinline__ TupleSlice Get(const int offset) const
	{
		return TupleSlice(
			t0[offset],
			t1[offset],
			t2[offset],
			t3[offset]);
	}

	template <int LANE, typename TupleSlice>
	__host__ __device__ __forceinline__ TupleSlice Get(const int offset) const
	{
		return TupleSlice(
			t0[LANE][offset],
			t1[LANE][offset],
			t2[LANE][offset],
			t3[LANE][offset]);
	}
};




} // namespace util
} // namespace b40c
