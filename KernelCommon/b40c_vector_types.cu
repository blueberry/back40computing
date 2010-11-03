/**
 * Copyright 2010 Duane Merrill
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
 */

#pragma once

namespace b40c {

//------------------------------------------------------------------------------
// Vector types
//------------------------------------------------------------------------------

template <typename K, int vec_elements> struct VecType;


//
// Define general vector types
//

template <typename K> 
struct VecType<K, 1> {
	K x;
	typedef K Type;
};

template <typename K> 
struct VecType<K, 2> {
	K x;
	K y;
	typedef VecType<K, 2> Type;
};

template <typename K> 
struct VecType<K, 4> {
	K x;
	K y;
	K z;
	K w;
	typedef VecType<K, 4> Type;
};

//
// Specialize certain built-in vector types
//

#define B40C_DEFINE_VECTOR_TYPE(base_type,short_type)                           \
  template<> struct VecType<base_type, 1> { typedef short_type##1 Type; };      \
  template<> struct VecType<base_type, 2> { typedef short_type##2 Type; };      \
  template<> struct VecType<base_type, 4> { typedef short_type##4 Type; };     

B40C_DEFINE_VECTOR_TYPE(char,               char)
B40C_DEFINE_VECTOR_TYPE(short,              short)
B40C_DEFINE_VECTOR_TYPE(int,                int)
B40C_DEFINE_VECTOR_TYPE(long,               long)
B40C_DEFINE_VECTOR_TYPE(long long,          longlong)
B40C_DEFINE_VECTOR_TYPE(unsigned char,      uchar)
B40C_DEFINE_VECTOR_TYPE(unsigned short,     ushort)
B40C_DEFINE_VECTOR_TYPE(unsigned int,       uint)
B40C_DEFINE_VECTOR_TYPE(unsigned long,      ulong)
B40C_DEFINE_VECTOR_TYPE(unsigned long long, ulonglong)
B40C_DEFINE_VECTOR_TYPE(float,              float)
B40C_DEFINE_VECTOR_TYPE(double,             double)

#undef B40C_DEFINE_VECTOR_TYPE


//------------------------------------------------------------------------------
// Utilities for moving types through global memory with cache modifiers
//------------------------------------------------------------------------------

enum CacheModifier {
	NONE,
	CG,
	CS
};


__device__ __forceinline__ int LoadCG(int* d_ptr) 
{
	int retval;
	asm("ld.global.cg.s32 %0, [%1];" : "=r"(retval) : _B40C_ASM_PTR_(d_ptr));
	return retval;
}

//-----------------------------------------------------------------------------
// Global Load
//-----------------------------------------------------------------------------

template <typename T, CacheModifier CACHE_MODIFIER> struct GlobalLoad;

// Generic NONE modifier

template <typename T> struct GlobalLoad<T, NONE> 
{
	__device__ __forceinline__ static void Ld(T &dest, T* d_ptr, int offset) {
		dest = d_ptr[offset]; 
	}
};


#define B40C_DEFINE_GLOBAL_LOAD(base_type, short_type, ptx_type, reg_mod)																												\
	template <> struct GlobalLoad<base_type, CG> {																												\
		__device__ __forceinline__ static void Ld(base_type &dest, base_type* d_ptr, volatile int offset) {														\
			asm("ld.global.cg."#ptx_type" %0, [%1];" : "="#reg_mod(dest) : _B40C_ASM_PTR_(d_ptr + offset));																	\
		}																																							\
	};																																								\
	template <> struct GlobalLoad<base_type, CS> {																												\
		__device__ __forceinline__ static void Ld(base_type &dest, base_type* d_ptr, volatile int offset) {														\
			asm("ld.global.cs."#ptx_type" %0, [%1];" : "="#reg_mod(dest) : _B40C_ASM_PTR_(d_ptr + offset));																	\
		}																																							\
	};																																								\
	template <> struct GlobalLoad<short_type##1, CG> {																												\
		__device__ __forceinline__ static void Ld(short_type##1 &dest, short_type##1* d_ptr, volatile int offset) {														\
			asm("ld.global.cg."#ptx_type" %0, [%1];" : "="#reg_mod(dest) : _B40C_ASM_PTR_(d_ptr + offset));																	\
		}																																							\
	};																																								\
	template <> struct GlobalLoad<short_type##1, CS> {																												\
		__device__ __forceinline__ static void Ld(short_type##1 &dest, short_type##1* d_ptr, volatile int offset) {														\
			asm("ld.global.cs."#ptx_type" %0, [%1];" : "="#reg_mod(dest) : _B40C_ASM_PTR_(d_ptr + offset));																	\
		}																																							\
	};																																								\
	template <> struct GlobalLoad<short_type##2, CG> {																												\
		__device__ __forceinline__ static void Ld(short_type##2 &dest, short_type##2* d_ptr, volatile int offset) {													\
			asm("ld.global.cg.v2."#ptx_type" {%0, %1}, [%2];" : "="#reg_mod(dest.x), "="#reg_mod(dest.y) : _B40C_ASM_PTR_(d_ptr + offset));										\
		}																																							\
	};																																								\
	template <> struct GlobalLoad<short_type##2, CS> {																												\
		__device__ __forceinline__ static void Ld(short_type##2 &dest, short_type##2* d_ptr, volatile int offset) {													\
			asm("ld.global.cs.v2."#ptx_type" {%0, %1}, [%2];" : "="#reg_mod(dest.x), "="#reg_mod(dest.y) : _B40C_ASM_PTR_(d_ptr + offset));										\
		}																																							\
	};																																								\
	template <> struct GlobalLoad<short_type##4, CG> {																												\
		__device__ __forceinline__ static void Ld(short_type##4 &dest, short_type##4* d_ptr, volatile int offset) {													\
			asm("ld.global.cg.v4."#ptx_type" {%0, %1, %2, %3}, [%4];" : "="#reg_mod(dest.x), "="#reg_mod(dest.y), "="#reg_mod(dest.z), "="#reg_mod(dest.w) : _B40C_ASM_PTR_(d_ptr + offset));	\
		}																																							\
	};																																								\
	template <> struct GlobalLoad<short_type##4, CS> {																												\
		__device__ __forceinline__ static void Ld(short_type##4 &dest, short_type##4* d_ptr, volatile int offset) {													\
			asm("ld.global.cs.v4."#ptx_type" {%0, %1, %2, %3}, [%4];" : "="#reg_mod(dest.x), "="#reg_mod(dest.y), "="#reg_mod(dest.z), "="#reg_mod(dest.w) : _B40C_ASM_PTR_(d_ptr + offset));	\
		}																																							\
	};


B40C_DEFINE_GLOBAL_LOAD(char, char, s8, r)
B40C_DEFINE_GLOBAL_LOAD(short, short, s16, r)
B40C_DEFINE_GLOBAL_LOAD(int, int, s32, r)
B40C_DEFINE_GLOBAL_LOAD(long, long, s64, l)
B40C_DEFINE_GLOBAL_LOAD(long long, longlong, s64, l)
B40C_DEFINE_GLOBAL_LOAD(unsigned char, uchar, u8, r)
B40C_DEFINE_GLOBAL_LOAD(unsigned short, ushort, u16, r)
B40C_DEFINE_GLOBAL_LOAD(unsigned int, uint, u32, r)
B40C_DEFINE_GLOBAL_LOAD(unsigned long, ulong, u64, l)
B40C_DEFINE_GLOBAL_LOAD(unsigned long long, ulonglong, u64, l)
B40C_DEFINE_GLOBAL_LOAD(float, float, f32, r)
B40C_DEFINE_GLOBAL_LOAD(double, double, f64, l)

#undef B40C_DEFINE_GLOBAL_LOAD
	



} // namespace b40c

