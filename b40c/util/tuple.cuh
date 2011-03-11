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

#include <b40c/util/srts_grid.cuh>
#include <b40c/util/basic_utils.cuh>

namespace b40c {
namespace util {


template <
	typename T0,
	typename T1 = NullType,
	typename T2 = NullType,
	typename T3 = NullType>
struct Tuple
{
	enum {
		NUM_FIELDS = 4 - Equals<T1, NullType>::VALUE - Equals<T2, NullType>::VALUE - Equals<T3, NullType>::VALUE
	};

	// Fields

	T0 t0;
	T1 t1;
	T2 t2;
	T3 t3;

	// Constructors

	Tuple(T0 t0) :
		t0(t0),
		t1(NullType()), t2(NullType()), t3(NullType()) {}

	Tuple(T0 t0, T1 t1) :
		t0(t0), t1(t1),
		t2(NullType()), t3(NullType()) {}

	Tuple(T0 t0, T1 t1, T2 t2) :
		t0(t0), t1(t1), t2(t2),
		t3(NullType()) {}

	Tuple(T0 t0, T1 t1, T2 t2, T3 t3) :
		t0(t0), t1(t1), t2(t2), t3(t3) {}

};


template <typename Tuple, int NUM_FIELDS = Tuple::NUM_FIELDS>
struct TupleAccess;

template <typename Tuple>
struct TupleAccess<Tuple, 1>
{
	template <typename TupleSoa>
	static __host__ __device__ __forceinline__ void Set(Tuple &tuple, TupleSoa tuple_soa, int index)
	{
		tuple_soa.t0[index] = tuple.t0;
	}

	template <typename TupleSoa, int LANE, int OFFSET>
	static __host__ __device__ __forceinline__ void Set(Tuple &tuple, TupleSoa tuple_soa)
	{
		tuple_soa.t0[LANE][OFFSET] = tuple.t0;
	}

	template <typename TupleSoa>
	static __host__ __device__ __forceinline__ Tuple Get(TupleSoa tuple_soa, int index)
	{
		return Tuple(tuple_soa.t0[index]);
	}

	template <typename TupleSoa, int LANE, int OFFSET>
	static __host__ __device__ __forceinline__ Tuple Get(TupleSoa tuple_soa, int index)
	{
		return Tuple(tuple_soa.t0[LANE][OFFSET]);
	}
};

template <typename Tuple>
struct TupleAccess<Tuple, 2>
{
	template <typename TupleSoa>
	static __host__ __device__ __forceinline__ void Set(Tuple &tuple, TupleSoa tuple_soa, int index)
	{
		tuple_soa.t0[index] = tuple.t0;
		tuple_soa.t1[index] = tuple.t1;
	}

	template <typename TupleSoa, int LANE, int OFFSET>
	static __host__ __device__ __forceinline__ void Set(Tuple &tuple, TupleSoa tuple_soa)
	{
		tuple_soa.t0[LANE][OFFSET] = tuple.t0;
		tuple_soa.t1[LANE][OFFSET] = tuple.t1;
	}

	template <typename TupleSoa>
	static __host__ __device__ __forceinline__ Tuple Get(TupleSoa tuple_soa, int index)
	{
		return Tuple(tuple_soa.t0[index], tuple_soa.t1[index]);
	}

	template <typename TupleSoa, int LANE, int OFFSET>
	static __host__ __device__ __forceinline__ Tuple Get(TupleSoa tuple_soa, int index)
	{
		return Tuple(tuple_soa.t0[LANE][OFFSET], tuple_soa.t1[LANE][OFFSET]);
	}
};

template <typename Tuple>
struct TupleAccess<Tuple, 3>
{
	template <typename TupleSoa>
	static __host__ __device__ __forceinline__ void Set(Tuple &tuple, TupleSoa tuple_soa, int index)
	{
		tuple_soa.t0[index] = tuple.t0;
		tuple_soa.t1[index] = tuple.t1;
		tuple_soa.t2[index] = tuple.t2;
	}

	template <typename TupleSoa, int LANE, int OFFSET>
	static __host__ __device__ __forceinline__ void Set(Tuple &tuple, TupleSoa tuple_soa, int index)
	{
		tuple_soa.t0[LANE][OFFSET] = tuple.t0;
		tuple_soa.t1[LANE][OFFSET] = tuple.t1;
		tuple_soa.t2[LANE][OFFSET] = tuple.t2;
	}

	template <typename TupleSoa>
	static __host__ __device__ __forceinline__ Tuple Get(TupleSoa tuple_soa, int index)
	{
		return Tuple(tuple_soa.t0[index], tuple_soa.t1[index], tuple_soa.t2[index]);
	}

	template <typename TupleSoa, int LANE, int OFFSET>
	static __host__ __device__ __forceinline__ Tuple Get(TupleSoa tuple_soa, int index)
	{
		return Tuple(tuple_soa.t0[LANE][OFFSET], tuple_soa.t1[LANE][OFFSET], tuple_soa.t2[LANE][OFFSET]);
	}
};

template <typename Tuple>
struct TupleAccess<Tuple, 4>
{
	template <typename TupleSoa>
	static __host__ __device__ __forceinline__ void Set(Tuple &tuple, TupleSoa tuple_soa, int index)
	{
		tuple_soa.t0[index] = tuple.t0;
		tuple_soa.t1[index] = tuple.t1;
		tuple_soa.t2[index] = tuple.t2;
		tuple_soa.t3[index] = tuple.t3;
	}

	template <typename TupleSoa, int LANE, int OFFSET>
	static __host__ __device__ __forceinline__ void Set(Tuple &tuple, TupleSoa tuple_soa, int index)
	{
		tuple_soa.t0[LANE][OFFSET] = tuple.t0;
		tuple_soa.t1[LANE][OFFSET] = tuple.t1;
		tuple_soa.t2[LANE][OFFSET] = tuple.t2;
		tuple_soa.t3[LANE][OFFSET] = tuple.t3;
	}

	template <typename TupleSoa>
	static __host__ __device__ __forceinline__ Tuple Get(TupleSoa tuple_soa, int index)
	{
		return Tuple(tuple_soa.t0[index], tuple_soa.t1[index], tuple_soa.t2[index], tuple_soa.t3[index]);
	}

	template <typename TupleSoa, int LANE, int OFFSET>
	static __host__ __device__ __forceinline__ Tuple Get(TupleSoa tuple_soa, int index)
	{
		return Tuple(tuple_soa.t0[LANE][OFFSET], tuple_soa.t1[LANE][OFFSET], tuple_soa.t2[LANE][OFFSET], tuple_soa.t3[LANE][OFFSET]);
	}
};




} // namespace util
} // namespace b40c
