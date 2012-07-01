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
 * Thread utilities for writing memory (optionally using cache modifiers)
 ******************************************************************************/

#pragma once

#include "../ptx_intrinsics.cuh"
#include "../ns_umbrella.cuh"

CUB_NS_PREFIX
namespace cub {


//-----------------------------------------------------------------------------
// Tags and constants
//-----------------------------------------------------------------------------

/**
 * Enumeration of store modifiers.
 */
enum StoreModifier
{
	// Global store modifiers
	STORE_NONE,		// Default (currently STORE_WB)
	STORE_WB,		// Cache write-back all coherent levels
	STORE_CG,		// Cache at global level
	STORE_CS, 		// Cache streaming (likely to be accessed once)
	STORE_WT, 		// Cache write-through (to system memory)
	STORE_GLOBAL_LIMIT,

	// Shared store modifiers
	STORE_VS,		// Shared-volatile
};


//-----------------------------------------------------------------------------
// Generic ThreadStore() operation
//-----------------------------------------------------------------------------

/**
 * Define HasThreadStore structure for testing the presence of nested
 * ThreadStoreTag type names within data types
 */
CUB_HAS_NESTED_TYPE(HasThreadStore, ThreadStoreTag)


/**
 * Dispatch ThreadStore() to value if it exposes a ThreadStoreTag typedef
 */
template <typename T, bool HAS_THREAD_STORE = HasThreadStore<T>::VALUE>
struct ThreadStoreDispatch
{
	template <StoreModifier MODIFIER>
	static __device__ __forceinline__ void ThreadStore(T *ptr, const T& val)
	{
		val.ThreadStore<MODIFIER>(ptr);
	}
};


/**
 * Dispatch ThreadStore() by modifier
 */
template <typename T>
struct ThreadStoreDispatch<T, false>
{
	// Specialization by modifier.
	template <StoreModifier MODIFIER>
	static __device__ __forceinline__ void ThreadStore(T *ptr, const T& val);

	// Generic STORE_NONE specialization
	template <>
	static __device__ __forceinline__ void ThreadStore<STORE_NONE>(T *ptr, const T& val)
	{
		// Straightforward dereference
		*ptr = val;
	}

	// Generic STORE_VS specialization
	template <>
	static __device__ __forceinline__ void ThreadStore<STORE_VS>(T *ptr, const T& val)
	{
		// Use volatile pointer if T is a primitive
		typedef typename If<Traits<T>::PRIMITIVE, volatile T*, T*>::Type PtrT;

		*((PtrT) ptr) = val;

		// Prevent compiler from reordering or omitting memory accesses between rounds
		if (!Traits<T>::PRIMITIVE) __threadfence_block();
	}
};


/**
 * Generic ThreadStore() operation.  Further specialized below.
 */
template <StoreModifier MODIFIER, typename T>
__device__ __forceinline__ void ThreadStore(T *ptr, const T& val)
{
	ThreadStoreDispatch<T>::template ThreadStore<MODIFIER>(ptr, val);
}


//-----------------------------------------------------------------------------
// ThreadStore() specializations by modifier and data type (i.e., primitives
// and CUDA vector types)
//-----------------------------------------------------------------------------


/**
 * Define a global ThreadStore() specialization for type
 */
#define CUB_G_STORE_0(type, asm_type, ptx_type, reg_mod, cub_modifier, ptx_modifier)	\
	template<>																			\
	void ThreadStore<cub_modifier, type>(type* ptr, const type& val)					\
	{																					\
		const asm_type raw = reinterpret_cast<const asm_type&>(val);					\
		asm("st.global."#ptx_modifier"."#ptx_type" [%0], %1;" : :						\
			_CUB_ASM_PTR_(ptr),															\
			#reg_mod(raw)); 															\
	}

/**
 * Define a global ThreadStore() specialization for the vector-1 type
 */
#define CUB_G_STORE_1(type, component_type, asm_type, ptx_type, reg_mod, cub_modifier, ptx_modifier)	\
	template<>																			\
	void ThreadStore<cub_modifier, type>(type* ptr, const type& val)					\
	{																					\
		const asm_type raw_x = reinterpret_cast<const asm_type&>(val.x);				\
		asm("st.global."#ptx_modifier"."#ptx_type" [%0], %1;" : :						\
			_CUB_ASM_PTR_(ptr),															\
			#reg_mod(raw_x));															\
	}

/**
 * Define a volatile-shared ThreadStore() specialization for the vector-1 type
 */
#define CUB_VS_STORE_1(type, component_type, asm_type, ptx_type, reg_mod)				\
	template<>																			\
	void ThreadStore<STORE_VS, type>(type* ptr, const type& val)						\
	{																					\
		ThreadStore<STORE_VS>(															\
			(asm_type*) ptr,															\
			reinterpret_cast<const asm_type&>(val.x));									\
	}

/**
 * Define a global ThreadStore() specialization for the vector-2 type
 */
#define CUB_G_STORE_2(type, component_type, asm_type, ptx_type, reg_mod, cub_modifier, ptx_modifier)	\
	template<>																			\
	void ThreadStore<cub_modifier, type>(type* ptr, const type& val)					\
	{																					\
		const asm_type raw_x = reinterpret_cast<const asm_type&>(val.x);				\
		const asm_type raw_y = reinterpret_cast<const asm_type&>(val.y);				\
		asm("st.global."#ptx_modifier".v2."#ptx_type" [%0], {%1, %2};" : :				\
			_CUB_ASM_PTR_(ptr),															\
			#reg_mod(raw_x), 															\
			#reg_mod(raw_y));															\
	}

/**
 * Define a volatile-shared ThreadStore() specialization for the vector-2 type.
 * Performs separate references if the component_type is only 1 byte (otherwise we lose
 * performance due to the bitfield ops to disassemble the value)
 */
#define CUB_VS_STORE_2(type, component_type, asm_type, ptx_type, reg_mod)				\
	template<>																			\
	void ThreadStore<STORE_VS, type>(type* ptr, const type& val)						\
	{																					\
	if (sizeof(component_type) == 1)													\
		{																				\
			component_type *base_ptr = (component_type*) ptr;							\
			ThreadStore<STORE_VS>(base_ptr, (component_type) val.x);					\
			ThreadStore<STORE_VS>(base_ptr + 1, (component_type) val.y);				\
		} 																				\
		else																			\
		{																				\
			const asm_type raw_x = reinterpret_cast<const asm_type&>(val.x);			\
			const asm_type raw_y = reinterpret_cast<const asm_type&>(val.y);			\
			asm("st.shared.volatile.v2."#ptx_type" [%0], {%1, %2};" : :					\
				_CUB_ASM_PTR_(ptr),														\
				#reg_mod(raw_x), 														\
				#reg_mod(raw_y));														\
		}																				\
	}

/**
 * Define a global ThreadStore() specialization for the vector-4 type
 */
#define CUB_G_STORE_4(type, component_type, asm_type, ptx_type, reg_mod, cub_modifier, ptx_modifier)	\
	template<>																			\
	void ThreadStore<cub_modifier, type>(type* ptr, const type& val)					\
	{																					\
		const asm_type raw_x = reinterpret_cast<const asm_type&>(val.x);				\
		const asm_type raw_y = reinterpret_cast<const asm_type&>(val.y);				\
		const asm_type raw_z = reinterpret_cast<const asm_type&>(val.z);				\
		const asm_type raw_w = reinterpret_cast<const asm_type&>(val.w);				\
		asm("st.global."#ptx_modifier".v4."#ptx_type" [%0], {%1, %2, %3, %4};" : :		\
			_CUB_ASM_PTR_(ptr),															\
			#reg_mod(raw_x), 															\
			#reg_mod(raw_y), 															\
			#reg_mod(raw_z), 															\
			#reg_mod(raw_w));															\
	}

/**
 * Define a volatile-shared ThreadStore() specialization for the vector-4 type.
 * Performs separate references if the component_type is only 1 byte (otherwise we lose
 * performance due to the bitfield ops to disassemble the value)
 */
#define CUB_VS_STORE_4(type, component_type, asm_type, ptx_type, reg_mod)				\
	template<>																			\
	void ThreadStore<STORE_VS, type>(type* ptr, const type& val)						\
	{																					\
		if (sizeof(component_type) == 1)												\
		{																				\
			component_type *base_ptr = (component_type*) ptr;							\
			ThreadStore<STORE_VS>(base_ptr, (component_type) val.x);					\
			ThreadStore<STORE_VS>(base_ptr + 1, (component_type) val.y);				\
			ThreadStore<STORE_VS>(base_ptr + 2, (component_type) val.z);				\
			ThreadStore<STORE_VS>(base_ptr + 3, (component_type) val.w);				\
		} 																				\
		else																			\
		{																				\
			const asm_type raw_x = reinterpret_cast<const asm_type&>(val.x);			\
			const asm_type raw_y = reinterpret_cast<const asm_type&>(val.y);			\
			const asm_type raw_z = reinterpret_cast<const asm_type&>(val.z);			\
			const asm_type raw_w = reinterpret_cast<const asm_type&>(val.w);			\
			asm("st.volatile.shared.v4."#ptx_type" [%0], {%1, %2, %3, %4};" : :			\
				_CUB_ASM_PTR_(ptr),														\
				#reg_mod(raw_x), 														\
				#reg_mod(raw_y), 														\
				#reg_mod(raw_z), 														\
				#reg_mod(raw_w));														\
		}																				\
	}

/**
 * Define a ThreadStore() specialization for the 64-bit vector-4 type.
 * Uses two vector-2 Stores.
 */
#define CUB_STORE_4L(type, half_type, cub_modifier)										\
	template<>																			\
	void ThreadStore<cub_modifier, type>(type* ptr, const type& val)					\
	{																					\
		const half_type* half_val = reinterpret_cast<const half_type*>(&val);			\
		half_type* half_ptr = reinterpret_cast<half_type*>(ptr);						\
		ThreadStore<cub_modifier>(half_ptr, half_val[0]);								\
		ThreadStore<cub_modifier>(half_ptr + 1, half_val[1]);							\
	}

/**
 * Define ThreadStore() specializations for the (non-vector) type
 */
#define CUB_STORES_0(type, asm_type, ptx_type, reg_mod)									\
	CUB_G_STORE_0(type, asm_type, ptx_type, reg_mod, STORE_WB, wb)						\
	CUB_G_STORE_0(type, asm_type, ptx_type, reg_mod, STORE_CG, cg)						\
	CUB_G_STORE_0(type, asm_type, ptx_type, reg_mod, STORE_CS, cs)						\
	CUB_G_STORE_0(type, asm_type, ptx_type, reg_mod, STORE_WT, wt)

/**
 * Define ThreadStore() specializations for the vector-1 component_type
 */
#define CUB_STORES_1(type, component_type, asm_type, ptx_type, reg_mod)					\
	CUB_VS_STORE_1(type, component_type, asm_type, ptx_type, reg_mod)					\
	CUB_G_STORE_1(type, component_type, asm_type, ptx_type, reg_mod, STORE_WB, wb)		\
	CUB_G_STORE_1(type, component_type, asm_type, ptx_type, reg_mod, STORE_CG, cg)		\
	CUB_G_STORE_1(type, component_type, asm_type, ptx_type, reg_mod, STORE_CS, cs)		\
	CUB_G_STORE_1(type, component_type, asm_type, ptx_type, reg_mod, STORE_WT, wt)

/**
 * Define ThreadStore() specializations for the vector-2 component_type
 */
#define CUB_STORES_2(type, component_type, asm_type, ptx_type, reg_mod)					\
	CUB_VS_STORE_2(type, component_type, asm_type, ptx_type, reg_mod)					\
	CUB_G_STORE_2(type, component_type, asm_type, ptx_type, reg_mod, STORE_WB, wb)		\
	CUB_G_STORE_2(type, component_type, asm_type, ptx_type, reg_mod, STORE_CG, cg)		\
	CUB_G_STORE_2(type, component_type, asm_type, ptx_type, reg_mod, STORE_CS, cs)		\
	CUB_G_STORE_2(type, component_type, asm_type, ptx_type, reg_mod, STORE_WT, wt)

/**
 * Define ThreadStore() specializations for the vector-4 component_type
 */
#define CUB_STORES_4(type, component_type, asm_type, ptx_type, reg_mod)					\
	CUB_VS_STORE_4(type, component_type, asm_type, ptx_type, reg_mod)					\
	CUB_G_STORE_4(type, component_type, asm_type, ptx_type, reg_mod, STORE_WB, wb)		\
	CUB_G_STORE_4(type, component_type, asm_type, ptx_type, reg_mod, STORE_CG, cg)		\
	CUB_G_STORE_4(type, component_type, asm_type, ptx_type, reg_mod, STORE_CS, cs)		\
	CUB_G_STORE_4(type, component_type, asm_type, ptx_type, reg_mod, STORE_WT, wt)

/**
 * Define ThreadStore() specializations for the 256-bit vector-4 component_type
 */
#define CUB_STORES_4L(type, half_type)				\
	CUB_STORE_4L(type, half_type, STORE_VS)			\
	CUB_STORE_4L(type, half_type, STORE_WB)			\
	CUB_STORE_4L(type, half_type, STORE_CG)			\
	CUB_STORE_4L(type, half_type, STORE_CS)			\
	CUB_STORE_4L(type, half_type, STORE_WT)

/**
 * Define vector-0/1/2 ThreadStore() specializations for the component type
 */
#define CUB_STORES_012(component_type, vec_prefix, asm_type, ptx_type, reg_mod)		\
	CUB_STORES_0(component_type, asm_type, ptx_type, reg_mod)						\
	CUB_STORES_1(vec_prefix##1, component_type, asm_type, ptx_type, reg_mod)		\
	CUB_STORES_2(vec_prefix##2, component_type, asm_type, ptx_type, reg_mod)

/**
 * Define vector-0/1/2/4 ThreadStore() specializations for the component type
 */
#define CUB_STORES_0124(component_type, vec_prefix, asm_type, ptx_type, reg_mod)	\
	CUB_STORES_012(component_type, vec_prefix, asm_type, ptx_type, reg_mod)			\
	CUB_STORES_4(vec_prefix##4, component_type, asm_type, ptx_type, reg_mod)

/**
 * Expand ThreadStore() implementations for primitive types.
 */
// Signed
CUB_STORES_0124(char, char, short, s8, h)
CUB_STORES_0(signed char, short, s8, h)
CUB_STORES_0124(short, short, short, s16, h)
CUB_STORES_0124(int, int, int, s32, r)
CUB_STORES_012(long long, longlong, long long, u64, l)
CUB_STORES_4L(longlong4, longlong2);

// Unsigned
CUB_STORES_0124(unsigned char, uchar, unsigned short, u8, h)
CUB_STORES_0(bool, short, u8, h)
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

#undef CUB_G_STORE_0
#undef CUB_G_STORE_1
#undef CUB_G_STORE_2
#undef CUB_G_STORE_4
#undef CUB_SV_STORE_1
#undef CUB_SV_STORE_2
#undef CUB_SV_STORE_4
#undef CUB_STORE_4L
#undef CUB_STORES_0
#undef CUB_STORES_1
#undef CUB_STORES_2
#undef CUB_STORES_4
#undef CUB_STORES_4L
#undef CUB_STORES_012
#undef CUB_STORES_0124


} // namespace cub
CUB_NS_POSTFIX
