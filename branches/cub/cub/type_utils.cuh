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
 * Common CUB type manipulation (metaprogramming) utilities
 ******************************************************************************/

#pragma once

#include "ns_umbrella.cuh"

CUB_NS_PREFIX
namespace cub {


/**
 * Null type
 */
struct NullType {};


/**
 * Statically determine log2(N), rounded up.
 *
 * For example:
 *     Log2<8>::VALUE		// 3
 *     Log2<3>::VALUE 		// 2
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
 * If ? Then : Else
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
 * Removes const and volatile qualifiers from type Tp.
 *
 * For example:
 *     typename RemoveQualifiers<volatile int>::Type 		// int;
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


/**
 * Allows the definition of structures that will detect the presence
 * of the specified type name within other classes
 */
#define CUB_HAS_NESTED_TYPE(detect_struct, nested_type_name)			\
	template <typename T>												\
	struct detect_struct												\
	{																	\
		template <typename C>											\
		static char& test(typename C::nested_type_name*);				\
		template <typename>												\
		static int& test(...);											\
		enum															\
		{																\
			VALUE = sizeof(test<T>(0)) < sizeof(int)					\
		};																\
	};


/**
 * Simple enable-if (similar to Boost)
 */
template <bool Condition, class T = void>
struct EnableIf
{
	typedef T Type;
};

template <class T>
struct EnableIf<false, T> {};


/******************************************************************************
 * Simple type traits utilities.
 *
 * For example:
 *     Traits<int>::CATEGORY 			// SIGNED_INTEGER
 *     Traits<NullType>::NULL_TYPE 		// true
 *     Traits<uint4>::CATEGORY 			// NOT_A_NUMBER
 *     Traits<uint4>::PRIMITIV; 		// false
 *
 ******************************************************************************/

/**
 * Basic type categories
 */
enum Category
{
	NOT_A_NUMBER,
	SIGNED_INTEGER,
	UNSIGNED_INTEGER,
	FLOATING_POINT
};


/**
 * Basic type traits
 */
template <Category _CATEGORY, bool _PRIMITIVE, bool _NULL_TYPE>
struct BaseTraits
{
	static const Category CATEGORY 		= _CATEGORY;
	enum {
		PRIMITIVE						= _PRIMITIVE,
		NULL_TYPE						= _NULL_TYPE
	};
};


/**
 * Numeric traits
 */
template <typename T> struct NumericTraits : 				BaseTraits<NOT_A_NUMBER, false, false> {};
template <> struct NumericTraits<NullType> : 				BaseTraits<NOT_A_NUMBER, false, true> {};

template <> struct NumericTraits<char> : 					BaseTraits<SIGNED_INTEGER, true, false> {};
template <> struct NumericTraits<signed char> : 			BaseTraits<SIGNED_INTEGER, true, false> {};
template <> struct NumericTraits<short> : 					BaseTraits<SIGNED_INTEGER, true, false> {};
template <> struct NumericTraits<int> : 					BaseTraits<SIGNED_INTEGER, true, false> {};
template <> struct NumericTraits<long> : 					BaseTraits<SIGNED_INTEGER, true, false> {};
template <> struct NumericTraits<long long> : 				BaseTraits<SIGNED_INTEGER, true, false> {};

template <> struct NumericTraits<unsigned char> : 			BaseTraits<UNSIGNED_INTEGER, true, false> {};
template <> struct NumericTraits<unsigned short> : 			BaseTraits<UNSIGNED_INTEGER, true, false> {};
template <> struct NumericTraits<unsigned int> : 			BaseTraits<UNSIGNED_INTEGER, true, false> {};
template <> struct NumericTraits<unsigned long> : 			BaseTraits<UNSIGNED_INTEGER, true, false> {};
template <> struct NumericTraits<unsigned long long> : 		BaseTraits<UNSIGNED_INTEGER, true, false> {};

template <> struct NumericTraits<float> : 					BaseTraits<FLOATING_POINT, true, false> {};
template <> struct NumericTraits<double> : 					BaseTraits<FLOATING_POINT, true, false> {};


/**
 * Type traits
 */
template <typename T>
struct Traits : NumericTraits<typename RemoveQualifiers<T>::Type> {};



/******************************************************************************
 * Simple array traits utilities.
 *
 * For example:
 *
 *     typedef int A[10];
 *     ArrayTraits<A>::DIMS 			// 1
 *     ArrayTraits<A>::ELEMENTS			// 10
 *     typename ArrayTraits<A>::Type	// int
 *
 *     typedef int B[10][20];
 *     ArrayTraits<B>::DIMS 			// 2
 *     ArrayTraits<B>::ELEMENTS			// 200
 *     typename ArrayTraits<B>::Type	// int
 *
 *     typedef int C;
 *     ArrayTraits<C>::DIMS 			// 0
 *     ArrayTraits<C>::ELEMENTS			// 1
 *     typename ArrayTraits<C>::Type	// int

 *     typedef int* D;
 *     ArrayTraits<D>::DIMS 			// 1
 *     ArrayTraits<D>::ELEMENTS			// 1
 *     typename ArrayTraits<D>::Type	// int
 *
 *     typedef int (*E)[2];
 *     ArrayTraits<E>::DIMS 			// 2
 *     ArrayTraits<E>::ELEMENTS			// 2
 *     typename ArrayTraits<E>::Type	// int
 *
 ******************************************************************************/

/**
 * Array traits
 */
template <typename ArrayType, int LENGTH = -1>
struct ArrayTraits;


/**
 * Specialization for non array type
 */
template <typename DimType, int LENGTH>
struct ArrayTraits
{
	typedef DimType Type;

	enum {
		ELEMENTS 	= 1,
		DIMS		= 0
	};
};


/**
 * Specialization for pointer type
 */
template <typename DimType, int LENGTH>
struct ArrayTraits<DimType*, LENGTH>
{
	typedef typename ArrayTraits<DimType>::Type Type;

	enum {
		ELEMENTS 	= ArrayTraits<DimType>::ELEMENTS,
		DIMS		= ArrayTraits<DimType>::DIMS + 1,
	};
};


/**
 * Specialization for array type
 */
template <typename DimType, int LENGTH>
struct ArrayTraits<DimType[LENGTH], -1>
{
	typedef typename ArrayTraits<DimType>::Type Type;

	enum {
		ELEMENTS 	= ArrayTraits<DimType>::ELEMENTS * LENGTH,
		DIMS		= ArrayTraits<DimType>::DIMS + 1,
	};
};


/******************************************************************************
 * Derive CUDA vector-types for built-in types
 *
 * For example:
 *
 *     VectorType<unsigned int, 2> pair; 			// uint2 pair;
 *
 ******************************************************************************/

enum {
	MAX_VEC_ELEMENTS = 4,	// The maximum number of elements in CUDA vector types
};

/**
 * Vector type
 */
template <typename T, int vec_elements> struct VectorType;

/**
 * Generic vector-1 type
 */
template <typename T>
struct VectorType<T, 1>
{
	T x;

	typedef VectorType<T, 1> Type;
};

/**
 * Generic vector-2 type
 */
template <typename T>
struct VectorType<T, 2>
{
	T x;
	T y;

	typedef VectorType<T, 2> Type;
};

/**
 * Generic vector-3 type
 */
template <typename T>
struct VectorType<T, 3>
{
	T x;
	T y;
	T z;

	typedef VectorType<T, 3> Type;
};

/**
 * Generic vector-4 type
 */
template <typename T>
struct VectorType<T, 4>
{
	T x;
	T y;
	T z;
	T w;

	typedef VectorType<T, 4> Type;
};

/**
 * Macro for expanding partially-specialized built-in vector types
 */
#define CUB_DEFINE_VECTOR_TYPE(base_type,short_type)              					\
  template<> struct VectorType<base_type, 1> { typedef short_type##1 Type; };		\
  template<> struct VectorType<base_type, 2> { typedef short_type##2 Type; };		\
  template<> struct VectorType<base_type, 3> { typedef short_type##3 Type; };		\
  template<> struct VectorType<base_type, 4> { typedef short_type##4 Type; };

// Expand CUDA vector types for built-in primitives
CUB_DEFINE_VECTOR_TYPE(char,               char)
CUB_DEFINE_VECTOR_TYPE(signed char,        char)
CUB_DEFINE_VECTOR_TYPE(short,              short)
CUB_DEFINE_VECTOR_TYPE(int,                int)
CUB_DEFINE_VECTOR_TYPE(long,               long)
CUB_DEFINE_VECTOR_TYPE(long long,          longlong)
CUB_DEFINE_VECTOR_TYPE(unsigned char,      uchar)
CUB_DEFINE_VECTOR_TYPE(unsigned short,     ushort)
CUB_DEFINE_VECTOR_TYPE(unsigned int,       uint)
CUB_DEFINE_VECTOR_TYPE(unsigned long,      ulong)
CUB_DEFINE_VECTOR_TYPE(unsigned long long, ulonglong)
CUB_DEFINE_VECTOR_TYPE(float,              float)
CUB_DEFINE_VECTOR_TYPE(double,             double)
CUB_DEFINE_VECTOR_TYPE(bool,               uchar)

// Undefine macros
#undef CUB_DEFINE_VECTOR_TYPE


} // namespace cub
CUB_NS_POSTFIX
