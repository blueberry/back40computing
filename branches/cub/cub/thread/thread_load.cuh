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

#include <cuda.h>

#include "../ptx_intrinsics.cuh"
#include "../type_utils.cuh"
#include "../ns_wrapper.cuh"

CUB_NS_PREFIX
namespace cub {


//-----------------------------------------------------------------------------
// Tags and constants
//-----------------------------------------------------------------------------

/**
 * Enumeration of read cache modifiers.
 */
enum LoadModifier
{
	// Global store modifiers
	LOAD_NONE,		// Default (currently LOAD_CA for global loads, nothing for smem loads)
	LOAD_CA,		// Cache at all levels
	LOAD_CG,		// Cache at global level
	LOAD_CS, 		// Cache streaming (likely to be accessed once)
	LOAD_CV, 		// Cache as volatile (including cached system lines)
	LOAD_TEX,		// Texture (defaults to NONE if no tex reference is provided)
	LOAD_GLOBA_LIMIT,

	// Shared store modifiers
	LOAD_VS,		// Volatile shared

};


//-----------------------------------------------------------------------------
// Generic ThreadLoad() operation
//-----------------------------------------------------------------------------

/**
 * Define HasThreadLoad structure for testing the presence of nested
 * ThreadLoadTag type names within data types
 */
CUB_HAS_NESTED_TYPE(HasThreadLoad, ThreadLoadTag)


/**
 * Dispatch specializer
 */
template <LoadModifier MODIFIER, bool HAS_THREAD_LOAD>
struct ThreadLoadDispatch;

/**
 * Dispatch ThreadLoad() to value if it exposes a ThreadLoadTag typedef
 */
template <LoadModifier MODIFIER>
struct ThreadLoadDispatch<MODIFIER, true>
{
	template <typename T>
	static __device__ __forceinline__ T ThreadLoad(T *ptr)
	{
		T val;
		val.ThreadLoad<MODIFIER>(ptr);
		return val;
	}
};

/**
 * Generic LOAD_NONE specialization
 */
template <>
struct ThreadLoadDispatch<LOAD_NONE, false>
{
	template <typename T>
	static __device__ __forceinline__ T ThreadLoad(T *ptr)
	{
		// Straightforward dereference
		return *ptr;
	}
};

/**
 * Generic LOAD_VS specialization
 */
template <>
struct ThreadLoadDispatch<LOAD_VS, false>
{
	template <typename T>
	static __device__ __forceinline__ T ThreadLoad(T *ptr)
	{
		// Straightforward dereference of volatile pointer
		volatile T *volatile_ptr = ptr;
		return *volatile_ptr;
	}
};

/**
 * Generic ThreadLoad() operation.  Further specialized below.
 */
template <LoadModifier MODIFIER, typename T>
__device__ __forceinline__ T ThreadLoad(T *ptr)
{
	return ThreadLoadDispatch<MODIFIER, HasThreadLoad<T>::VALUE>::ThreadLoad(ptr);
}



//-----------------------------------------------------------------------------
// ThreadLoad() specializations by modifier and data type (i.e., primitives
// and CUDA vector types)
//-----------------------------------------------------------------------------

/**
 * Define a global ThreadLoad() specialization for type
 */
#define CUB_G_LOAD_0(type, asm_type, ptx_type, reg_mod, cub_modifier, ptx_modifier)		\
	template<>																			\
	type ThreadLoad<cub_modifier, type>(type* ptr) 										\
	{																					\
		asm_type raw;																	\
		asm("ld.global."#ptx_modifier"."#ptx_type" %0, [%1];" :							\
			"="#reg_mod(raw) : 															\
			_CUB_ASM_PTR_(ptr));														\
		type val = reinterpret_cast<type&>(raw);										\
		return val;																		\
	}

/**
 * Define a global ThreadLoad() specialization for the vector-1 type
 */
#define CUB_G_LOAD_1(type, component_type, asm_type, ptx_type, reg_mod, cub_modifier, ptx_modifier)	\
	template<>																			\
	type ThreadLoad<cub_modifier, type>(type* ptr) 										\
	{																					\
		asm_type raw;																	\
		asm("ld.global."#ptx_modifier"."#ptx_type" %0, [%1];" :							\
			"="#reg_mod(raw) :															\
			_CUB_ASM_PTR_(ptr));														\
		type val = { reinterpret_cast<component_type&>(raw) };							\
		return val;																		\
	}

/**
 * Define a volatile-shared ThreadLoad() specialization for the built-in
 * vector-1 type.  Simply use the component version.
 */
#define CUB_VS_LOAD_1(type, component_type, asm_type, ptx_type, reg_mod)				\
	template<>																			\
	type ThreadLoad<LOAD_VS, type>(type* ptr) 											\
	{																					\
		type val;																		\
		val.x = ThreadLoad<LOAD_VS>((component_type*) ptr);								\
		return val;																		\
	}

/**
 * Define a global ThreadLoad() specialization for the vector-2 type
 */
#define CUB_G_LOAD_2(type, component_type, asm_type, ptx_type, reg_mod, cub_modifier, ptx_modifier)	\
	template<>																			\
	type ThreadLoad<cub_modifier, type>(type* ptr) 										\
	{																					\
		asm_type raw_x, raw_y;															\
		asm("ld.global."#ptx_modifier".v2."#ptx_type" {%0, %1}, [%2];" :				\
			"="#reg_mod(raw_x), 														\
			"="#reg_mod(raw_y) :														\
			_CUB_ASM_PTR_(ptr));														\
		type val = {																	\
			reinterpret_cast<component_type&>(raw_x), 									\
			reinterpret_cast<component_type&>(raw_y) };									\
		return val;																		\
	}

/**
 * Define a volatile-shared ThreadLoad() specialization for the vector-2 type
 * Performs separate references if the component_type is only 1 byte (otherwise we lose
 * performance due to the bitfield ops to assemble the value)
 */
#define CUB_VS_LOAD_2(type, component_type, asm_type, ptx_type, reg_mod)				\
	template<>																			\
	type ThreadLoad<LOAD_VS, type>(type* ptr) 											\
	{																					\
		type val;																		\
		if ((sizeof(component_type) == 1) || (CUDA_VERSION < 4100))												\
		{																				\
			component_type *base_ptr = (component_type*) ptr;							\
			val.x = ThreadLoad<LOAD_VS>(base_ptr);										\
			val.y = ThreadLoad<LOAD_VS>(base_ptr + 1);									\
		} 																				\
		else																			\
		{																				\
			asm_type raw_x, raw_y;														\
			asm volatile ("{"																		\
				"	.reg ."_CUB_ASM_PTR_SIZE_" t1;"										\
				"	cvta.to.shared."_CUB_ASM_PTR_SIZE_" t1, %2;"						\
				"	ld.volatile.shared.v2."#ptx_type" {%0, %1}, [t1];"					\
				"}" :																	\
				"="#reg_mod(raw_x), 													\
				"="#reg_mod(raw_y) :													\
				_CUB_ASM_PTR_(ptr));													\
			val.x = reinterpret_cast<component_type&>(raw_x); 							\
			val.y = reinterpret_cast<component_type&>(raw_y);							\
		}																				\
		return val;																		\
	}

/**
 * Define a global ThreadLoad() specialization for the vector-4 type
 */
#define CUB_G_LOAD_4(type, component_type, asm_type, ptx_type, reg_mod, cub_modifier, ptx_modifier)	\
	template<>																			\
	type ThreadLoad<cub_modifier, type>(type* ptr) 										\
	{																					\
		asm_type raw_x, raw_y, raw_z, raw_w;											\
		asm volatile ("ld.global."#ptx_modifier".v4."#ptx_type" {%0, %1, %2, %3}, [%4];" :		\
			"="#reg_mod(raw_x), 														\
			"="#reg_mod(raw_y), 														\
			"="#reg_mod(raw_z), 														\
			"="#reg_mod(raw_w) :														\
			_CUB_ASM_PTR_(ptr));														\
		type val = {																	\
			reinterpret_cast<component_type&>(raw_x), 									\
			reinterpret_cast<component_type&>(raw_y), 									\
			reinterpret_cast<component_type&>(raw_z), 									\
			reinterpret_cast<component_type&>(raw_w) };									\
		return val;																		\
	}

/**
 * Define a volatile-shared ThreadLoad() specialization for the vector-4 type.
 * Performs separate references if the component_type is only 1 byte (otherwise we lose
 * performance due to the bitfield ops to assemble the value)
 */
#define CUB_VS_LOAD_4(type, component_type, asm_type, ptx_type, reg_mod)				\
	template<>																			\
	type ThreadLoad<LOAD_VS, type>(type* ptr) 											\
	{																					\
		type val;																		\
		if ((sizeof(component_type) == 1) || (CUDA_VERSION < 4100))												\
		{																				\
			component_type *base_ptr = (component_type*) ptr;							\
			val.x = ThreadLoad<LOAD_VS>(base_ptr);										\
			val.y = ThreadLoad<LOAD_VS>(base_ptr + 1);									\
			val.z = ThreadLoad<LOAD_VS>(base_ptr + 2);									\
			val.w = ThreadLoad<LOAD_VS>(base_ptr + 3);									\
		} 																				\
		else																			\
		{																				\
			asm_type raw_x, raw_y, raw_z, raw_w;										\
			asm volatile ("{"																		\
				"	.reg ."_CUB_ASM_PTR_SIZE_" t1;"										\
				"	cvta.to.shared."_CUB_ASM_PTR_SIZE_" t1, %4;"						\
				"	ld.volatile.shared.v4."#ptx_type" {%0, %1, %2, %3}, [t1];"			\
				"}" :																	\
				"="#reg_mod(raw_x), 													\
				"="#reg_mod(raw_y), 													\
				"="#reg_mod(raw_z), 													\
				"="#reg_mod(raw_w) :													\
				_CUB_ASM_PTR_(ptr));													\
			val.x = reinterpret_cast<component_type&>(raw_x); 							\
			val.y = reinterpret_cast<component_type&>(raw_y);							\
			val.z = reinterpret_cast<component_type&>(raw_z);							\
			val.w = reinterpret_cast<component_type&>(raw_w);							\
		}																				\
		return val;																		\
	}

/**
 * Define a ThreadLoad() specialization for the 64-bit
 * vector-4 type
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
 * Define ThreadLoad() specializations for the (non-vector) type
 */
#define CUB_LOADS_0(type, asm_type, ptx_type, reg_mod)									\
	CUB_G_LOAD_0(type, asm_type, ptx_type, reg_mod, LOAD_CA, ca)						\
	CUB_G_LOAD_0(type, asm_type, ptx_type, reg_mod, LOAD_CG, cg)						\
	CUB_G_LOAD_0(type, asm_type, ptx_type, reg_mod, LOAD_CS, cs)						\
	CUB_G_LOAD_0(type, asm_type, ptx_type, reg_mod, LOAD_CV, cv)

/**
 * Define ThreadLoad() specializations for the vector-1 component_type
 */
#define CUB_LOADS_1(type, component_type, asm_type, ptx_type, reg_mod)							\
	CUB_VS_LOAD_1(type, component_type, asm_type, ptx_type, reg_mod)							\
	CUB_G_LOAD_1(type, component_type, asm_type, ptx_type, reg_mod, LOAD_CA, ca)				\
	CUB_G_LOAD_1(type, component_type, asm_type, ptx_type, reg_mod, LOAD_CG, cg)				\
	CUB_G_LOAD_1(type, component_type, asm_type, ptx_type, reg_mod, LOAD_CS, cs)				\
	CUB_G_LOAD_1(type, component_type, asm_type, ptx_type, reg_mod, LOAD_CV, cv)

/**
 * Define ThreadLoad() specializations for the vector-2 component_type
 */
#define CUB_LOADS_2(type, component_type, asm_type, ptx_type, reg_mod)							\
	CUB_VS_LOAD_2(type, component_type, asm_type, ptx_type, reg_mod)							\
	CUB_G_LOAD_2(type, component_type, asm_type, ptx_type, reg_mod, LOAD_CA, ca)				\
	CUB_G_LOAD_2(type, component_type, asm_type, ptx_type, reg_mod, LOAD_CG, cg)				\
	CUB_G_LOAD_2(type, component_type, asm_type, ptx_type, reg_mod, LOAD_CS, cs)				\
	CUB_G_LOAD_2(type, component_type, asm_type, ptx_type, reg_mod, LOAD_CV, cv)

/**
 * Define ThreadLoad() specializations for the vector-4 component_type
 */
#define CUB_LOADS_4(type, component_type, asm_type, ptx_type, reg_mod)							\
	CUB_VS_LOAD_4(type, component_type, asm_type, ptx_type, reg_mod)							\
	CUB_G_LOAD_4(type, component_type, asm_type, ptx_type, reg_mod, LOAD_CA, ca)				\
	CUB_G_LOAD_4(type, component_type, asm_type, ptx_type, reg_mod, LOAD_CG, cg)				\
	CUB_G_LOAD_4(type, component_type, asm_type, ptx_type, reg_mod, LOAD_CS, cs)				\
	CUB_G_LOAD_4(type, component_type, asm_type, ptx_type, reg_mod, LOAD_CV, cv)

/**
 * Define ThreadLoad() specializations for the 256-bit vector-4 component_type
 */
#define CUB_LOADS_4L(type, half_type)					\
	CUB_LOAD_4L(type, half_type, LOAD_VS)				\
	CUB_LOAD_4L(type, half_type, LOAD_CA)				\
	CUB_LOAD_4L(type, half_type, LOAD_CG)				\
	CUB_LOAD_4L(type, half_type, LOAD_CS)				\
	CUB_LOAD_4L(type, half_type, LOAD_CV)

/**
 * Define vector-0/1/2 ThreadLoad() specializations for the component type
 */
#define CUB_LOADS_012(component_type, vec_prefix, asm_type, ptx_type, reg_mod)		\
	CUB_LOADS_0(component_type, asm_type, ptx_type, reg_mod)						\
	CUB_LOADS_1(vec_prefix##1, component_type, asm_type, ptx_type, reg_mod)			\
	CUB_LOADS_2(vec_prefix##2, component_type, asm_type, ptx_type, reg_mod)

/**
 * Define vector-0/1/2/4 ThreadLoad() specializations for the component type
 */
#define CUB_LOADS_0124(component_type, vec_prefix, asm_type, ptx_type, reg_mod)		\
	CUB_LOADS_012(component_type, vec_prefix, asm_type, ptx_type, reg_mod)			\
	CUB_LOADS_4(vec_prefix##4, component_type, asm_type, ptx_type, reg_mod)



/**
 * Expand ThreadLoad() implementations for primitive types.
 */

// Signed
CUB_LOADS_0124(char, char, short, s8, h)
CUB_LOADS_0(signed char, short, s8, h)
CUB_LOADS_0124(short, short, short, s16, h)
CUB_LOADS_0124(int, int, int, s32, r)
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
#undef CUB_LOADS_012
#undef CUB_LOADS_0124

} // namespace cub
CUB_NS_POSTFIX
