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
 * Derive CUDA vector-types for primitive types
 ******************************************************************************/

#pragma once

#include "ns_umbrella.cuh"
#include "thread/thread_load.cuh"
#include "thread/thread_store.cuh"

CUB_NS_PREFIX
namespace cub {


/******************************************************************************
 * Derive CUDA vector-types for primitive types
 *
 * For example:
 *
 *     typename VectorType<unsigned int, 2>::Type	// Aliases uint2
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
	typedef void ThreadLoadTag;
	typedef void ThreadStoreTag;

	// ThreadLoad
	template <LoadModifier MODIFIER>
	__device__ __forceinline__ 	void ThreadLoad(VectorType *ptr)
	{
		x = cub::ThreadLoad<MODIFIER>(&(ptr->x));
	}

	 // ThreadStore
	template <StoreModifier MODIFIER>
	__device__ __forceinline__ void ThreadStore(VectorType *ptr) const
	{
		cub::ThreadStore<MODIFIER>(&(ptr->x), x);
	}
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
	typedef void ThreadLoadTag;
	typedef void ThreadStoreTag;

	// ThreadLoad
	template <LoadModifier MODIFIER>
	__device__ __forceinline__ void ThreadLoad(VectorType *ptr)
	{
		x = cub::ThreadLoad<MODIFIER>(&(ptr->x));
		y = cub::ThreadLoad<MODIFIER>(&(ptr->y));
	}

	 // ThreadStore
	template <StoreModifier MODIFIER>
	__device__ __forceinline__ void ThreadStore(VectorType *ptr) const
	{
		cub::ThreadStore<MODIFIER>(&(ptr->x), x);
		cub::ThreadStore<MODIFIER>(&(ptr->y), y);
	}
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
	typedef void ThreadLoadTag;
	typedef void ThreadStoreTag;

	// ThreadLoad
	template <LoadModifier MODIFIER>
	__device__ __forceinline__ void ThreadLoad(VectorType *ptr)
	{
		x = cub::ThreadLoad<MODIFIER>(&(ptr->x));
		y = cub::ThreadLoad<MODIFIER>(&(ptr->y));
		z = cub::ThreadLoad<MODIFIER>(&(ptr->z));
	}

	 // ThreadStore
	template <StoreModifier MODIFIER>
	__device__ __forceinline__ void ThreadStore(VectorType *ptr) const
	{
		cub::ThreadStore<MODIFIER>(&(ptr->x), x);
		cub::ThreadStore<MODIFIER>(&(ptr->y), y);
		cub::ThreadStore<MODIFIER>(&(ptr->z), z);
	}

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
	typedef void ThreadLoadTag;
	typedef void ThreadStoreTag;

	// ThreadLoad
	template <LoadModifier MODIFIER>
	__device__ __forceinline__ void ThreadLoad(VectorType *ptr)
	{
		x = cub::ThreadLoad<MODIFIER>(&(ptr->x));
		y = cub::ThreadLoad<MODIFIER>(&(ptr->y));
		z = cub::ThreadLoad<MODIFIER>(&(ptr->z));
		w = cub::ThreadLoad<MODIFIER>(&(ptr->w));
	}

	 // ThreadStore
	template <StoreModifier MODIFIER>
	__device__ __forceinline__ void ThreadStore(VectorType *ptr) const
	{
		cub::ThreadStore<MODIFIER>(&(ptr->x), x);
		cub::ThreadStore<MODIFIER>(&(ptr->y), y);
		cub::ThreadStore<MODIFIER>(&(ptr->z), z);
		cub::ThreadStore<MODIFIER>(&(ptr->w), w);
	}
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
