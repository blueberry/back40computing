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
 * Thread utilities for reading memory (optionally using cache modifiers).
 * Cache modifiers will only be effected for built-in types (i.e., C++
 * primitives and CUDA vector-types).
 *
 * For example:
 *
 *     // 32-bit load using cache-global modifier:
 *
 *     		int *d_in;
 *     		int data = ThreadLoad<LOAD_CG>(d_in + threadIdx.x);
 *
 *
 *     // 16-bit load using default modifier
 *
 *     		short *d_in;
 *     		short data = ThreadLoad(d_in + threadIdx.x);
 *
 *
 *     // 256-bit load using cache-volatile modifier
 *
 *     		double4 *d_in;
 *     		double4 data = ThreadLoad<LOAD_CV>(d_in + threadIdx.x);
 *
 *
 *     // 96-bit load using default cache modifier (ignoring LOAD_CS)
 *
 *     		struct Foo { bool a; short b; } *d_struct = NULL;
 *     		Foo data = ThreadLoad<LOAD_CS>(d_in + threadIdx.x);
 *
 *
 ******************************************************************************/

#pragma once

#include "../ptx_intrinsics.cuh"
#include "../ns_umbrella.cuh"

CUB_NS_PREFIX
namespace cub {



/**
 * Enumeration of read cache modifiers.
 */
enum LoadModifier
{
	LOAD_NONE,		// Default (currently LOAD_CA for global loads, nothing for smem loads)
	LOAD_CA,		// Cache at all levels
	LOAD_CG,		// Cache at global level
	LOAD_CS, 		// Cache streaming (likely to be accessed once)
	LOAD_CV, 		// Cache as volatile (including cached system lines)
	LOAD_TEX,		// Texture (defaults to NONE if no tex reference is provided)

	LOAD_VS,		// Volatile shared

	LOAD_LIMIT
};


/**
 *
 */
template <LoadModifier MODIFIER> struct GenericLoad;


/**
 * Specialization for LOAD_NONE
 */
template <> struct GenericLoad<LOAD_NONE>
{
	template <typename T>
	static __device__ __forceinline__ T ThreadLoad(T *ptr)
	{
		return *ptr;
	}
};


/**
 * Specialization for LOAD_TEX
 */
template <> struct GenericLoad<LOAD_TEX> : GenericLoad<LOAD_NONE> {};


/**
 * Specialization for LOAD_VS
 */
template <> struct GenericLoad<LOAD_VS>
{
	template <typename T>
	static __device__ __forceinline__ T ThreadLoad(T *ptr)
	{
		return *ptr;
	}
};


/**
 * Generic ThreadLoad() operation
 */
template <LoadModifier MODIFIER, typename T>
__device__ __forceinline__ T ThreadLoad(T *ptr)
{
	return GenericLoad<MODIFIER>::ThreadLoad(ptr);
}





/**
 * Overload specializations for built-ins when compiling for SM20+
 */
#if __CUDA_ARCH__ >= 200

/**
 * Define a global ThreadLoad() specialization for the built-in (non-vector) "type"
 */
#define CUB_G_LOAD_0(type, raw_type, ptx_type, reg_mod, cub_modifier, ptx_modifier)		\
	template<>																			\
	type ThreadLoad<cub_modifier, type>(type* ptr) 										\
	{																					\
		raw_type raw;																	\
		asm("ld.global."#ptx_modifier"."#ptx_type" %0, [%1];" :							\
			"="#reg_mod(raw) : 															\
			_CUB_ASM_PTR_(ptr));														\
		type val = reinterpret_cast<type&>(raw);										\
		return val;																		\
	}

/**
 * Define a volatile shared ThreadLoad() specialization for the built-in (non-vector) "type"
 */
#define CUB_VS_LOAD_0(type, raw_type, ptx_type, reg_mod)								\
	template<>																			\
	type ThreadLoad<LOAD_VS, type>(type* ptr) 											\
	{																					\
		raw_type raw;																	\
		asm("ld.volatile.shared."#ptx_type" %0, [%1];" :								\
			"="#reg_mod(raw) : 															\
			_CUB_ASM_PTR_(ptr));														\
		type val = reinterpret_cast<type&>(raw);										\
		return val;																		\
	}

/**
 * Define a global ThreadLoad() specialization for the built-in vector-1 "type"
 */
#define CUB_G_LOAD_1(type, base_type, raw_type, ptx_type, reg_mod, cub_modifier, ptx_modifier)	\
	template<>																			\
	type ThreadLoad<cub_modifier, type>(type* ptr) 										\
	{																					\
		raw_type raw;																	\
		asm("ld.global."#ptx_modifier"."#ptx_type" %0, [%1];" :							\
			"="#reg_mod(raw) :															\
			_CUB_ASM_PTR_(ptr));														\
		type val = { reinterpret_cast<base_type&>(raw) };								\
		return val;																		\
	}

/**
 * Define a volatile shared ThreadLoad() specialization for the built-in vector-1 "type"
 */
#define CUB_VS_LOAD_1(type, base_type, raw_type, ptx_type, reg_mod)						\
	template<>																			\
	type ThreadLoad<LOAD_VS, type>(type* ptr) 											\
	{																					\
		raw_type raw;																	\
		asm("ld.volatile.shared."#ptx_type" %0, [%1];" :								\
			"="#reg_mod(raw) :															\
			_CUB_ASM_PTR_(ptr));														\
		type val = { reinterpret_cast<base_type&>(raw) };								\
		return val;																		\
	}

/**
 * Define a global ThreadLoad() specialization for the built-in vector-2 "type"
 */
#define CUB_G_LOAD_2(type, base_type, raw_type, ptx_type, reg_mod, cub_modifier, ptx_modifier)	\
	template<>																			\
	type ThreadLoad<cub_modifier, type>(type* ptr) 										\
	{																					\
		raw_type raw_x, raw_y;															\
		asm("ld.global."#ptx_modifier".v2."#ptx_type" {%0, %1}, [%2];" :				\
			"="#reg_mod(raw_x), 														\
			"="#reg_mod(raw_y) :														\
			_CUB_ASM_PTR_(ptr));														\
		type val = {																	\
			reinterpret_cast<base_type&>(raw_x), 										\
			reinterpret_cast<base_type&>(raw_y) };										\
		return val;																		\
	}

/**
 * Define a volatile shared ThreadLoad() specialization for the built-in vector-2 "type"
 */
#define CUB_VS_LOAD_2(type, base_type, raw_type, ptx_type, reg_mod)						\
	template<>																			\
	type ThreadLoad<LOAD_VS, type>(type* ptr) 											\
	{																					\
		raw_type raw_x, raw_y;															\
		asm("ld.volatile.shared.v2."#ptx_type" {%0, %1}, [%2];" :						\
			"="#reg_mod(raw_x), 														\
			"="#reg_mod(raw_y) :														\
			_CUB_ASM_PTR_(ptr));														\
		type val = {																	\
			reinterpret_cast<base_type&>(raw_x), 										\
			reinterpret_cast<base_type&>(raw_y) };										\
		return val;																		\
	}

/**
 * Define a global ThreadLoad() specialization for the built-in vector-4 "type"
 */
#define CUB_G_LOAD_4(type, base_type, raw_type, ptx_type, reg_mod, cub_modifier, ptx_modifier)	\
	template<>																			\
	type ThreadLoad<cub_modifier, type>(type* ptr) 										\
	{																					\
		raw_type raw_x, raw_y, raw_z, raw_w;											\
		asm("ld.global."#ptx_modifier".v4."#ptx_type" {%0, %1, %2, %3}, [%4];" :		\
			"="#reg_mod(raw_x), 														\
			"="#reg_mod(raw_y), 														\
			"="#reg_mod(raw_z), 														\
			"="#reg_mod(raw_w) :														\
			_CUB_ASM_PTR_(ptr));														\
		type val = {																	\
			reinterpret_cast<base_type&>(raw_x), 										\
			reinterpret_cast<base_type&>(raw_y), 										\
			reinterpret_cast<base_type&>(raw_z), 										\
			reinterpret_cast<base_type&>(raw_w) };										\
		return val;																		\
	}

/**
 * Define a volatile shared ThreadLoad() specialization for the built-in vector-4 "type"
 */
#define CUB_VS_LOAD_4(type, base_type, raw_type, ptx_type, reg_mod)						\
	template<>																			\
	type ThreadLoad<LOAD_VS, type>(type* ptr) 											\
	{																					\
		raw_type raw_x, raw_y, raw_z, raw_w;											\
		asm("ld.volatile.shared.v4."#ptx_type" {%0, %1, %2, %3}, [%4];" :				\
			"="#reg_mod(raw_x), 														\
			"="#reg_mod(raw_y), 														\
			"="#reg_mod(raw_z), 														\
			"="#reg_mod(raw_w) :														\
			_CUB_ASM_PTR_(ptr));														\
		type val = {																	\
			reinterpret_cast<base_type&>(raw_x), 										\
			reinterpret_cast<base_type&>(raw_y), 										\
			reinterpret_cast<base_type&>(raw_z), 										\
			reinterpret_cast<base_type&>(raw_w) };										\
		return val;																		\
	}

/**
 * Define a global ThreadLoad() specialization for the built-in 64-bit vector-4 "type"
 */
#define CUB_LOAD_4L(type, half_type, cub_modifier)										\
	template<>																			\
	type ThreadLoad<cub_modifier, type>(type* ptr) 										\
	{																					\
		type val; 																		\
		half_type* half_val = reinterpret_cast<half_type*>(&val);						\
		half_type* half_ptr = reinterpret_cast<half_type*>(ptr);						\
		half_val[0] = ThreadLoad<cub_modifier>(half_ptr);								\
		half_val[1] = ThreadLoad<cub_modifier>(half_ptr + 1);							\
		return val;																		\
	}


/**
 * Define ThreadLoad() specializations for the built-in (non-vector) "base_type"
 */
#define CUB_LOADS_0(base_type, raw_type, ptx_type, reg_mod)								\
	CUB_VS_LOAD_0(base_type, raw_type, ptx_type, reg_mod)								\
	CUB_G_LOAD_0(base_type, raw_type, ptx_type, reg_mod, LOAD_CA, ca)					\
	CUB_G_LOAD_0(base_type, raw_type, ptx_type, reg_mod, LOAD_CG, cg)					\
	CUB_G_LOAD_0(base_type, raw_type, ptx_type, reg_mod, LOAD_CS, cs)					\
	CUB_G_LOAD_0(base_type, raw_type, ptx_type, reg_mod, LOAD_CV, cv)

/**
 * Define ThreadLoad() specializations for the built-in vector-1 "vector-type"
 */
#define CUB_LOADS_1(type, base_type, raw_type, ptx_type, reg_mod)						\
	CUB_VS_LOAD_1(type, base_type, raw_type, ptx_type, reg_mod)							\
	CUB_G_LOAD_1(type, base_type, raw_type, ptx_type, reg_mod, LOAD_CA, ca)				\
	CUB_G_LOAD_1(type, base_type, raw_type, ptx_type, reg_mod, LOAD_CG, cg)				\
	CUB_G_LOAD_1(type, base_type, raw_type, ptx_type, reg_mod, LOAD_CS, cs)				\
	CUB_G_LOAD_1(type, base_type, raw_type, ptx_type, reg_mod, LOAD_CV, cv)

/**
 * Define ThreadLoad() specializations for the built-in vector-2 "vector-type"
 */
#define CUB_LOADS_2(type, base_type, raw_type, ptx_type, reg_mod)						\
	CUB_VS_LOAD_2(type, base_type, raw_type, ptx_type, reg_mod)							\
	CUB_G_LOAD_2(type, base_type, raw_type, ptx_type, reg_mod, LOAD_CA, ca)				\
	CUB_G_LOAD_2(type, base_type, raw_type, ptx_type, reg_mod, LOAD_CG, cg)				\
	CUB_G_LOAD_2(type, base_type, raw_type, ptx_type, reg_mod, LOAD_CS, cs)				\
	CUB_G_LOAD_2(type, base_type, raw_type, ptx_type, reg_mod, LOAD_CV, cv)

/**
 * Define ThreadLoad() specializations for the built-in vector-4 "vector-type"
 */
#define CUB_LOADS_4(type, base_type, raw_type, ptx_type, reg_mod)						\
	CUB_VS_LOAD_4(type, base_type, raw_type, ptx_type, reg_mod)							\
	CUB_G_LOAD_4(type, base_type, raw_type, ptx_type, reg_mod, LOAD_CA, ca)				\
	CUB_G_LOAD_4(type, base_type, raw_type, ptx_type, reg_mod, LOAD_CG, cg)				\
	CUB_G_LOAD_4(type, base_type, raw_type, ptx_type, reg_mod, LOAD_CS, cs)				\
	CUB_G_LOAD_4(type, base_type, raw_type, ptx_type, reg_mod, LOAD_CV, cv)

/**
 * Define ThreadLoad() specializations for the built-in 256-bit vector-4 "vector-type"
 */
#define CUB_LOADS_4L(type, half_type)					\
	CUB_LOAD_4L(type, half_type, LOAD_CA)				\
	CUB_LOAD_4L(type, half_type, LOAD_CG)				\
	CUB_LOAD_4L(type, half_type, LOAD_CS)				\
	CUB_LOAD_4L(type, half_type, LOAD_CV)

/**
 * Define base and vector-1/2 ThreadLoad() specializations for "type"
 */
#define CUB_LOADS_012(type, prefix, raw_type, ptx_type, reg_mod)		\
	CUB_LOADS_0(type, raw_type, ptx_type, reg_mod)						\
	CUB_LOADS_1(prefix##1, type, raw_type, ptx_type, reg_mod)			\
	CUB_LOADS_2(prefix##2, type, raw_type, ptx_type, reg_mod)

/**
 * Define base and vector-1/2/4 ThreadLoad() specializations for "type"
 */
#define CUB_LOADS_0124(type, prefix, raw_type, ptx_type, reg_mod)		\
	CUB_LOADS_012(type, prefix, raw_type, ptx_type, reg_mod)			\
	CUB_LOADS_4(prefix##4, type, raw_type, ptx_type, reg_mod)


/**
 * Expand ThreadLoad() implementations for all built-in types.
 */

// Signed
CUB_LOADS_0124(char, char, short, s8, h)
CUB_LOADS_0124(short, short, short, s16, h)
CUB_LOADS_0124(int, int, int, s32, r)
CUB_LOADS_0(signed char, short, s8, h)
CUB_LOADS_012(long long, longlong, long long, u64, l)
CUB_LOADS_4L(longlong4, longlong2);

// Unsigned
CUB_LOADS_0(bool, short, u8, h)
CUB_LOADS_0124(unsigned char, uchar, unsigned short, u8, h)
CUB_LOADS_0124(unsigned short, ushort, unsigned short, u16, h)
CUB_LOADS_0124(unsigned int, uint, unsigned int, u32, r)
CUB_LOADS_012(unsigned long long, ulonglong, unsigned long long, u64, l)
CUB_LOADS_4L(ulonglong4, ulonglong2);

// Floating point
CUB_LOADS_0124(float, float, float, f32, f)
CUB_LOADS_012(double, double, unsigned long long, u64, l)
CUB_LOADS_4L(double4, double2);

// Signed longs / unsigned longs
#if defined(__LP64__)
	// longs are 64-bit on non-Windows 64-bit compilers
	CUB_LOADS_012(long, long, long, u64, l)
	CUB_LOADS_4L(long4, long2);
	CUB_LOADS_012(unsigned long, ulong, unsigned long, u64, l)
	CUB_LOADS_4L(ulong4, ulong2);
#else
	// longs are 32-bit on everything else
	CUB_LOADS_0124(long, long, long, u32, r)
	CUB_LOADS_0124(unsigned long, ulong, unsigned long, u32, r)
#endif


/**
 * Undefine macros
 */

#undef CUB_G_LOAD_0
#undef CUB_G_LOAD_1
#undef CUB_G_LOAD_2
#undef CUB_G_LOAD_4
#undef CUB_SV_LOAD_0
#undef CUB_SV_LOAD_1
#undef CUB_SV_LOAD_2
#undef CUB_SV_LOAD_4
#undef CUB_LOAD_4L
#undef CUB_LOADS_0
#undef CUB_LOADS_1
#undef CUB_LOADS_2
#undef CUB_LOADS_4
#undef CUB_LOADS_4L
#undef CUB_LOADS_012
#undef CUB_LOADS_0124

#endif // (#ifdef __CUDA_ARCH__ >= 200)



} // namespace cub
CUB_NS_POSTFIX
