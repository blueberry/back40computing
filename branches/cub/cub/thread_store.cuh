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
 * Thread utilities for writing memory (optionally using cache modifiers)
 ******************************************************************************/

#pragma once

#include <cub/ptx_intrinsics.cuh>
#include <cub/ns_umbrella.cuh>

CUB_NS_PREFIX
namespace cub {



/**
 * Enumeration of write cache modifiers.
 */
enum StoreModifier {
	STORE_NONE,		// Default (currently STORE_WB)
	STORE_WB,		// Cache write-back all coherent levels
	STORE_CG,		// Cache at global level
	STORE_CS, 		// Cache streaming (likely to be accessed once)
	STORE_WT, 		// Cache write-through (to system memory)

	STORE_LIMIT
};


/**
 * Generic Store() operation
 */
template <typename T>
__device__ __forceinline__ void Store(T *ptr, const T& val)
{
	*ptr = val;
}


/**
 * Generic Store() operation
 */
template <StoreModifier STORE_MODIFIER, typename T>
__device__ __forceinline__ void Store(T *ptr, const T& val)
{
	*ptr = val;
}


/**
 * Overload specializations for built-ins when compiling for SM20+
 */
#if __CUDA_ARCH__ >= 200

/**
 * Define a Store() specialization for the built-in (non-vector) "type"
 */
#define CUB_STORE_0(type, raw_type, ptx_type, reg_mod, cub_modifier, ptx_modifier)		\
	template<>																			\
	void Store<cub_modifier, type>(type* ptr, const type& val)							\
	{																					\
		const raw_type raw = reinterpret_cast<const raw_type&>(val);					\
		asm("st.global."#ptx_modifier"."#ptx_type" [%0], %1;" : :						\
			_CUB_ASM_PTR_(ptr),															\
			#reg_mod(raw)); 															\
	}


/**
 * Define a Store() specialization for the built-in vector-1 "type"
 */
#define CUB_STORE_1(type, raw_type, ptx_type, reg_mod, cub_modifier, ptx_modifier)		\
	template<>																			\
	void Store<cub_modifier, type>(type* ptr, const type& val)							\
	{																					\
		const raw_type raw_x = reinterpret_cast<const raw_type&>(val.x);				\
		asm("st.global."#ptx_modifier"."#ptx_type" [%0], %1;" : :						\
			_CUB_ASM_PTR_(ptr),															\
			#reg_mod(raw_x));															\
	}

/**
 * Define a Store() specialization for the built-in vector-2 "type"
 */
#define CUB_STORE_2(type, raw_type, ptx_type, reg_mod, cub_modifier, ptx_modifier)		\
	template<>																			\
	void Store<cub_modifier, type>(type* ptr, const type& val)							\
	{																					\
		const raw_type raw_x = reinterpret_cast<const raw_type&>(val.x);				\
		const raw_type raw_y = reinterpret_cast<const raw_type&>(val.y);				\
		asm("st.global."#ptx_modifier".v2."#ptx_type" [%0], {%1, %2};" : :				\
			_CUB_ASM_PTR_(ptr),															\
			#reg_mod(raw_x), 															\
			#reg_mod(raw_y));															\
	}

/**
 * Define a Store() specialization for the built-in vector-4 "type"
 */
#define CUB_STORE_4(type, raw_type, ptx_type, reg_mod, cub_modifier, ptx_modifier)		\
	template<>																			\
	void Store<cub_modifier, type>(type* ptr, const type& val)							\
	{																					\
		const raw_type raw_x = reinterpret_cast<const raw_type&>(val.x);				\
		const raw_type raw_y = reinterpret_cast<const raw_type&>(val.y);				\
		const raw_type raw_z = reinterpret_cast<const raw_type&>(val.z);				\
		const raw_type raw_w = reinterpret_cast<const raw_type&>(val.w);				\
		asm("st.global."#ptx_modifier".v4."#ptx_type" [%0], {%1, %2, %3, %4};" : :		\
			_CUB_ASM_PTR_(ptr),															\
			#reg_mod(raw_x), 															\
			#reg_mod(raw_y), 															\
			#reg_mod(raw_z), 															\
			#reg_mod(raw_w));															\
	}

/**
 * Define a Store() specialization for the built-in 64-bit vector-4 "type".
 * Uses two vector-2 Stores.
 */
#define CUB_STORE_4L(type, half_type, cub_modifier)										\
	template<>																			\
	void Store<cub_modifier, type>(type* ptr, const type& val)							\
	{																					\
		const half_type* half_val = reinterpret_cast<const half_type*>(&val);			\
		half_type* half_ptr = reinterpret_cast<half_type*>(ptr);						\
		Store<cub_modifier>(half_ptr, half_val[0]);										\
		Store<cub_modifier>(half_ptr + 1, half_val[1]);									\
	}

/**
 * Define Store() specializations for the built-in (non-vector) "type"
 */
#define CUB_STORES_0(type, raw_type, ptx_type, reg_mod)									\
	CUB_STORE_0(type, raw_type, ptx_type, reg_mod, STORE_WB, wb)						\
	CUB_STORE_0(type, raw_type, ptx_type, reg_mod, STORE_CG, cg)						\
	CUB_STORE_0(type, raw_type, ptx_type, reg_mod, STORE_CS, cs)						\
	CUB_STORE_0(type, raw_type, ptx_type, reg_mod, STORE_WT, wt)

/**
 * Define Store() specializations for the built-in vector-1 "type"
 */
#define CUB_STORES_1(type, raw_type, ptx_type, reg_mod)					\
	CUB_STORE_1(type, raw_type, ptx_type, reg_mod, STORE_WB, wb)		\
	CUB_STORE_1(type, raw_type, ptx_type, reg_mod, STORE_CG, cg)		\
	CUB_STORE_1(type, raw_type, ptx_type, reg_mod, STORE_CS, cs)		\
	CUB_STORE_1(type, raw_type, ptx_type, reg_mod, STORE_WT, wt)

/**
 * Define Store() specializations for the built-in vector-2 "type"
 */
#define CUB_STORES_2(type, raw_type, ptx_type, reg_mod)					\
	CUB_STORE_2(type, raw_type, ptx_type, reg_mod, STORE_WB, wb)		\
	CUB_STORE_2(type, raw_type, ptx_type, reg_mod, STORE_CG, cg)		\
	CUB_STORE_2(type, raw_type, ptx_type, reg_mod, STORE_CS, cs)		\
	CUB_STORE_2(type, raw_type, ptx_type, reg_mod, STORE_WT, wt)

/**
 * Define Store() specializations for the built-in vector-4 "type"
 */
#define CUB_STORES_4(type, raw_type, ptx_type, reg_mod)					\
	CUB_STORE_4(type, raw_type, ptx_type, reg_mod, STORE_WB, wb)		\
	CUB_STORE_4(type, raw_type, ptx_type, reg_mod, STORE_CG, cg)		\
	CUB_STORE_4(type, raw_type, ptx_type, reg_mod, STORE_CS, cs)		\
	CUB_STORE_4(type, raw_type, ptx_type, reg_mod, STORE_WT, wt)

/**
 * Define Store() specializations for the built-in 256-bit vector-4 "type"
 */
#define CUB_STORES_4L(type, half_type)					\
	CUB_STORE_4L(type, half_type, STORE_WB)				\
	CUB_STORE_4L(type, half_type, STORE_CG)				\
	CUB_STORE_4L(type, half_type, STORE_CS)				\
	CUB_STORE_4L(type, half_type, STORE_WT)

/**
 * Define base and vector-1/2 Store() specializations for "type"
 */
#define CUB_STORES_012(type, prefix, raw_type, ptx_type, reg_mod)		\
	CUB_STORES_0(type, raw_type, ptx_type, reg_mod)						\
	CUB_STORES_1(prefix##1, raw_type, ptx_type, reg_mod)				\
	CUB_STORES_2(prefix##2, raw_type, ptx_type, reg_mod)

/**
 * Define base and vector-1/2/4 Store() specializations for "type"
 */
#define CUB_STORES_0124(type, prefix, raw_type, ptx_type, reg_mod)		\
	CUB_STORES_012(type, prefix, raw_type, ptx_type, reg_mod)			\
	CUB_STORES_4(prefix##4, raw_type, ptx_type, reg_mod)


/**
 * Expand Store() implementations for all built-in types.
 */

// Signed
CUB_STORES_0124(char, char, short, s8, h)
CUB_STORES_0124(short, short, short, s16, h)
CUB_STORES_0124(int, int, int, s32, r)
CUB_STORES_0(signed char, short, s8, h)
CUB_STORES_012(long long, longlong, long long, u64, l)
CUB_STORES_4L(longlong4, longlong2);

// Unsigned
CUB_STORES_0124(unsigned char, uchar, unsigned short, u8, h)
CUB_STORES_0124(unsigned short, ushort, unsigned short, u16, h)
CUB_STORES_0124(unsigned int, uint, unsigned int, u32, r)
CUB_STORES_012(unsigned long long, ulonglong, unsigned long long, u64, l)
CUB_STORES_4L(ulonglong4, ulonglong2);

// Floating point
CUB_STORES_0124(float, float, float, f32, f)
CUB_STORES_012(double, double, unsigned long long, u64, l)
CUB_STORES_4L(double4, double2);

// Signed longs / unsigned longs
#if defined(__LP64__)
	// longs are 64-bit on non-Windows 64-bit compilers
	CUB_STORES_012(long, long, long, u64, l)
	CUB_STORES_4L(long4, long2);
	CUB_STORES_012(unsigned long, ulong, unsigned long, u64, l)
	CUB_STORES_4L(ulong4, ulong2);
#else
	// longs are 32-bit on everything else
	CUB_STORES_0124(long, long, long, u32, r)
	CUB_STORES_0124(unsigned long, ulong, unsigned long, u32, r)
#endif


/**
 * Undefine macros
 */

#undef CUB_STORE_0
#undef CUB_STORE_1
#undef CUB_STORE_2
#undef CUB_STORE_4
#undef CUB_STORE_4L
#undef CUB_STORES_0
#undef CUB_STORES_1
#undef CUB_STORES_2
#undef CUB_STORES_4
#undef CUB_STORES_4L
#undef CUB_STORES_012
#undef CUB_STORES_0124

#endif // (#ifdef __CUDA_ARCH__ >= 200)


} // namespace cub
CUB_NS_POSTFIX
