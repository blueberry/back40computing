/******************************************************************************
 * 
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
 * 
 ******************************************************************************/

/******************************************************************************
 * Kernel utilities for moving types through global memory with cache modifiers
 ******************************************************************************/

#pragma once

#include <b40c_cuda_properties.cu>

namespace b40c {

/**
 * Enumeration of data movement cache modifiers.
 */
enum CacheModifier {
	NONE,
	CG,
	CS, 
	CA,
	WB
};


/**
 * Routines for modified loads through cache.  We use structs specialized by value 
 * type and cache-modifier to implement load operations
 */
template <typename T, CacheModifier CACHE_MODIFIER> struct ModifiedLoad;

#if __CUDA_ARCH__ >= 200

	/**
	 * Defines specialized load ops for only the base type 
	 */
	#define B40C_DEFINE_BASE_GLOBAL_LOAD(base_type, ptx_type, reg_mod)																								\
		template <> struct ModifiedLoad<base_type, CG> {																												\
			__device__ __forceinline__ static void Ld(base_type &val, base_type* d_ptr, int offset) {																\
				asm("ld.global.cg."#ptx_type" %0, [%1];" : "="#reg_mod(val) : _B40C_ASM_PTR_(d_ptr + offset));														\
			}																																						\
		};																																							\
		template <> struct ModifiedLoad<base_type, CS> {																												\
			__device__ __forceinline__ static void Ld(base_type &val, base_type* d_ptr, int offset) {																\
				asm("ld.global.cs."#ptx_type" %0, [%1];" : "="#reg_mod(val) : _B40C_ASM_PTR_(d_ptr + offset));														\
			}																																						\
		};																																							\
		template <> struct ModifiedLoad<base_type, CA> {																												\
			__device__ __forceinline__ static void Ld(base_type &val, base_type* d_ptr, int offset) {																\
				asm("ld.global.ca."#ptx_type" %0, [%1];" : "="#reg_mod(val) : _B40C_ASM_PTR_(d_ptr + offset));														\
			}																																						\
		};																																								


	/**
	 * Defines specialized load ops for both the base type and for its derivative vector types
	 */
	#define B40C_DEFINE_GLOBAL_LOAD(base_type, dest_type, short_type, ptx_type, reg_mod)																												\
		template <> struct ModifiedLoad<base_type, CG> {																												\
			__device__ __forceinline__ static void Ld(dest_type &val, base_type* d_ptr, int offset) {																\
				asm("ld.global.cg."#ptx_type" %0, [%1];" : "="#reg_mod(val) : _B40C_ASM_PTR_(d_ptr + offset));														\
			}																																						\
		};																																							\
		template <> struct ModifiedLoad<base_type, CS> {																												\
			__device__ __forceinline__ static void Ld(dest_type &val, base_type* d_ptr, int offset) {																\
				asm("ld.global.cs."#ptx_type" %0, [%1];" : "="#reg_mod(val) : _B40C_ASM_PTR_(d_ptr + offset));														\
			}																																						\
		};																																							\
		template <> struct ModifiedLoad<base_type, CA> {																												\
			__device__ __forceinline__ static void Ld(dest_type &val, base_type* d_ptr, int offset) {																\
				asm("ld.global.ca."#ptx_type" %0, [%1];" : "="#reg_mod(val) : _B40C_ASM_PTR_(d_ptr + offset));														\
			}																																						\
		};																																							\
		template <> struct ModifiedLoad<short_type##1, CG> {																											\
			__device__ __forceinline__ static void Ld(short_type##1 &val, short_type##1* d_ptr, int offset) {														\
				asm("ld.global.cg."#ptx_type" %0, [%1];" : "="#reg_mod(val.x) : _B40C_ASM_PTR_(d_ptr + offset));														\
			}																																						\
		};																																							\
		template <> struct ModifiedLoad<short_type##1, CS> {																											\
			__device__ __forceinline__ static void Ld(short_type##1 &val, short_type##1* d_ptr, int offset) {														\
				asm("ld.global.cs."#ptx_type" %0, [%1];" : "="#reg_mod(val.x) : _B40C_ASM_PTR_(d_ptr + offset));														\
			}																																						\
		};																																							\
		template <> struct ModifiedLoad<short_type##1, CA> {																											\
			__device__ __forceinline__ static void Ld(short_type##1 &val, short_type##1* d_ptr, int offset) {														\
				asm("ld.global.ca."#ptx_type" %0, [%1];" : "="#reg_mod(val.x) : _B40C_ASM_PTR_(d_ptr + offset));														\
			}																																						\
		};																																							\
		template <> struct ModifiedLoad<short_type##2, CG> {																											\
			__device__ __forceinline__ static void Ld(short_type##2 &val, short_type##2* d_ptr, int offset) {														\
				asm("ld.global.cg.v2."#ptx_type" {%0, %1}, [%2];" : "="#reg_mod(val.x), "="#reg_mod(val.y) : _B40C_ASM_PTR_(d_ptr + offset));						\
			}																																						\
		};																																							\
		template <> struct ModifiedLoad<short_type##2, CS> {																											\
			__device__ __forceinline__ static void Ld(short_type##2 &val, short_type##2* d_ptr, int offset) {														\
				asm("ld.global.cs.v2."#ptx_type" {%0, %1}, [%2];" : "="#reg_mod(val.x), "="#reg_mod(val.y) : _B40C_ASM_PTR_(d_ptr + offset));						\
			}																																						\
		};																																							\
		template <> struct ModifiedLoad<short_type##2, CA> {																											\
			__device__ __forceinline__ static void Ld(short_type##2 &val, short_type##2* d_ptr, int offset) {														\
				asm("ld.global.ca.v2."#ptx_type" {%0, %1}, [%2];" : "="#reg_mod(val.x), "="#reg_mod(val.y) : _B40C_ASM_PTR_(d_ptr + offset));						\
			}																																						\
		};																																							\
		template <> struct ModifiedLoad<short_type##4, CG> {																											\
			__device__ __forceinline__ static void Ld(short_type##4 &val, short_type##4* d_ptr, int offset) {														\
				asm("ld.global.cg.v4."#ptx_type" {%0, %1, %2, %3}, [%4];" : "="#reg_mod(val.x), "="#reg_mod(val.y), "="#reg_mod(val.z), "="#reg_mod(val.w) : _B40C_ASM_PTR_(d_ptr + offset));	\
			}																																						\
		};																																							\
		template <> struct ModifiedLoad<short_type##4, CS> {																											\
			__device__ __forceinline__ static void Ld(short_type##4 &val, short_type##4* d_ptr, int offset) {														\
				asm("ld.global.cs.v4."#ptx_type" {%0, %1, %2, %3}, [%4];" : "="#reg_mod(val.x), "="#reg_mod(val.y), "="#reg_mod(val.z), "="#reg_mod(val.w) : _B40C_ASM_PTR_(d_ptr + offset));	\
			}																																						\
		};																																							\
		template <> struct ModifiedLoad<short_type##4, CA> {																											\
			__device__ __forceinline__ static void Ld(short_type##4 &val, short_type##4* d_ptr, int offset) {														\
				asm("ld.global.ca.v4."#ptx_type" {%0, %1, %2, %3}, [%4];" : "="#reg_mod(val.x), "="#reg_mod(val.y), "="#reg_mod(val.z), "="#reg_mod(val.w) : _B40C_ASM_PTR_(d_ptr + offset));	\
			}																																						\
		};

	// Cache-modified loads for built-in structures
	B40C_DEFINE_GLOBAL_LOAD(char, signed char, char, s8, r)
	B40C_DEFINE_BASE_GLOBAL_LOAD(signed char, s8, r)			// only need to define base: char2,char4, etc already defined from char
	B40C_DEFINE_GLOBAL_LOAD(short, short, short, s16, r)
	B40C_DEFINE_GLOBAL_LOAD(int, int, int, s32, r)
	B40C_DEFINE_GLOBAL_LOAD(long, long, long, s64, l)
	B40C_DEFINE_GLOBAL_LOAD(long long, long long, longlong, s64, l)
	B40C_DEFINE_GLOBAL_LOAD(unsigned char, unsigned char, uchar, u8, r)
	B40C_DEFINE_GLOBAL_LOAD(unsigned short, unsigned short, ushort, u16, r)
	B40C_DEFINE_GLOBAL_LOAD(unsigned int, unsigned int, uint, u32, r)
	B40C_DEFINE_GLOBAL_LOAD(unsigned long, unsigned long, ulong, u64, l)
	B40C_DEFINE_GLOBAL_LOAD(unsigned long long, unsigned long long, ulonglong, u64, l)
	B40C_DEFINE_GLOBAL_LOAD(float, float, float, f32, f)
	B40C_DEFINE_BASE_GLOBAL_LOAD(double, f64, d)	// loads of vector-doubles don't compile
	
	#undef B40C_DEFINE_BASE_GLOBAL_LOAD
	#undef B40C_DEFINE_GLOBAL_LOAD

	// Workaround for the fact that the assembler reports an error when attempting to 
	// make vector loads of doubles.
	template <> struct ModifiedLoad<double2, CG> {																												
		__device__ __forceinline__ static void Ld(double2 &val, double2* d_ptr, int offset) {													
			asm("ld.global.cg.f64 %0, [%1];" : "=d"(val.x) : _B40C_ASM_PTR_(d_ptr + offset));																	
			asm("ld.global.cg.f64 %0, [%1];" : "=d"(val.y) : _B40C_ASM_PTR_(d_ptr + offset + 1));																	
		}																																							
	};																																								
	template <> struct ModifiedLoad<double4, CG> {																												
		__device__ __forceinline__ static void Ld(double4 &val, double4* d_ptr, int offset) {													
			asm("ld.global.cg.f64 %0, [%1];" : "=d"(val.x) : _B40C_ASM_PTR_(d_ptr + offset));																	
			asm("ld.global.cg.f64 %0, [%1];" : "=d"(val.y) : _B40C_ASM_PTR_(d_ptr + offset + 1));																	
			asm("ld.global.cg.f64 %0, [%1];" : "=d"(val.z) : _B40C_ASM_PTR_(d_ptr + offset + 2));																	
			asm("ld.global.cg.f64 %0, [%1];" : "=d"(val.w) : _B40C_ASM_PTR_(d_ptr + offset + 3));																	
		}																																							
	};																																								
	
	// NONE-modified load 
	template <typename T> struct ModifiedLoad<T, NONE>
	{
		__device__ __forceinline__ static void Ld(T &val, T* d_ptr, int offset) {
			val = d_ptr[offset]; 
		}
	};
	
	// NONE-modified load 
	template <> struct ModifiedLoad<char, NONE>
	{
		__device__ __forceinline__ static void Ld(signed char &val, char* d_ptr, int offset) {
			val = d_ptr[offset]; 
		}
	};
	
#else // loads

	//
	// Nothing is cached in these architectures
	//
	
	// Load normally
	template <typename T, CacheModifier CACHE_MODIFIER> struct ModifiedLoad
	{
		__device__ __forceinline__ static void Ld(T &val, T* d_ptr, int offset) {
			val = d_ptr[offset]; 
		}
	};
	
	// Accomodate bizarre introduction of "signed" for char loads
	template <CacheModifier CACHE_MODIFIER> struct ModifiedLoad<char, CACHE_MODIFIER>
	{
		__device__ __forceinline__ static void Ld(signed char &val, char* d_ptr, int offset) {
			val = d_ptr[offset]; 
		}
	};

#endif	// loads


	
	
/**
 * Routines for modified stores through cache.  We use structs specialized by value 
 * type and cache-modifier to implement store operations
 */
template <typename T, CacheModifier CACHE_MODIFIER> struct ModifiedStore;

#if __CUDA_ARCH__ >= 200


	/**
	 * Defines specialized store ops for only the base type 
	 */

	#define B40C_DEFINE_BASE_GLOBAL_STORE(base_type, ptx_type, reg_mod)																								\
		template <> struct ModifiedStore<base_type, CG> {																											\
			__device__ __forceinline__ static void St(const base_type &val, base_type* d_ptr, int offset) {															\
				asm("st.global.cg."#ptx_type" [%0], %1;" : : _B40C_ASM_PTR_(d_ptr + offset), #reg_mod(val));															\
			}																																						\
		};																																							\
		template <> struct ModifiedStore<base_type, CS> {																											\
			__device__ __forceinline__ static void St(const base_type &val, base_type* d_ptr, int offset) {															\
				asm("st.global.cs."#ptx_type" [%0], %1;" : : _B40C_ASM_PTR_(d_ptr + offset), #reg_mod(val));															\
			}																																						\
		};																																							\
		template <> struct ModifiedStore<base_type, CA> {																											\
			__device__ __forceinline__ static void St(const base_type &val, base_type* d_ptr, int offset) {															\
				asm("st.global.wb."#ptx_type" [%0], %1;" : : _B40C_ASM_PTR_(d_ptr + offset), #reg_mod(val));															\
			}																																						\
		};																																								


	/**
	 * Defines specialized store ops for both the base type and for its derivative vector types
	 */
	#define B40C_DEFINE_GLOBAL_STORE(base_type, dest_type, short_type, ptx_type, reg_mod)																												\
		template <> struct ModifiedStore<base_type, CG> {																												\
			__device__ __forceinline__ static void St(const base_type &val, base_type* d_ptr, int offset) {																\
				asm("st.global.cg."#ptx_type" [%0], %1;" : : _B40C_ASM_PTR_(d_ptr + offset), #reg_mod(val));														\
			}																																						\
		};																																							\
		template <> struct ModifiedStore<base_type, CS> {																												\
			__device__ __forceinline__ static void St(const base_type &val, base_type* d_ptr, int offset) {																\
				asm("st.global.cs."#ptx_type" [%0], %1;" : : _B40C_ASM_PTR_(d_ptr + offset), #reg_mod(val));														\
			}																																						\
		};																																							\
		template <> struct ModifiedStore<base_type, WB> {																												\
			__device__ __forceinline__ static void St(const base_type &val, base_type* d_ptr, int offset) {																\
				asm("st.global.wb."#ptx_type" [%0], %1;" : : _B40C_ASM_PTR_(d_ptr + offset), #reg_mod(val));														\
			}																																						\
		};																																							\
		template <> struct ModifiedStore<short_type##1, CG> {																											\
			__device__ __forceinline__ static void St(const short_type##1 &val, short_type##1* d_ptr, int offset) {														\
				asm("st.global.cg."#ptx_type" [%0], %1;" : : _B40C_ASM_PTR_(d_ptr + offset), #reg_mod(val.x));														\
			}																																						\
		};																																							\
		template <> struct ModifiedStore<short_type##1, CS> {																											\
			__device__ __forceinline__ static void St(const short_type##1 &val, short_type##1* d_ptr, int offset) {														\
				asm("st.global.cs."#ptx_type" [%0], %1;" : : _B40C_ASM_PTR_(d_ptr + offset), #reg_mod(val.x));														\
			}																																						\
		};																																							\
		template <> struct ModifiedStore<short_type##1, WB> {																											\
			__device__ __forceinline__ static void St(const short_type##1 &val, short_type##1* d_ptr, int offset) {														\
				asm("st.global.wb."#ptx_type" [%0], %1;" : : _B40C_ASM_PTR_(d_ptr + offset), #reg_mod(val.x));														\
			}																																						\
		};																																							\
		template <> struct ModifiedStore<short_type##2, CG> {																											\
			__device__ __forceinline__ static void St(const short_type##2 &val, short_type##2* d_ptr, int offset) {														\
				asm("st.global.cg.v2."#ptx_type" [%0], {%1, %2};" : :  _B40C_ASM_PTR_(d_ptr + offset), #reg_mod(val.x), #reg_mod(val.y));						\
			}																																						\
		};																																							\
		template <> struct ModifiedStore<short_type##2, CS> {																											\
			__device__ __forceinline__ static void St(const short_type##2 &val, short_type##2* d_ptr, int offset) {														\
				asm("st.global.cs.v2."#ptx_type" [%0], {%1, %2};" : :  _B40C_ASM_PTR_(d_ptr + offset), #reg_mod(val.x), #reg_mod(val.y));						\
			}																																						\
		};																																							\
		template <> struct ModifiedStore<short_type##2, WB> {																											\
			__device__ __forceinline__ static void St(const short_type##2 &val, short_type##2* d_ptr, int offset) {														\
				asm("st.global.wb.v2."#ptx_type" [%0], {%1, %2};" : :  _B40C_ASM_PTR_(d_ptr + offset), #reg_mod(val.x), #reg_mod(val.y));						\
			}																																						\
		};																																							\
		template <> struct ModifiedStore<short_type##4, CG> {																											\
			__device__ __forceinline__ static void St(const short_type##4 &val, short_type##4* d_ptr, int offset) {														\
				asm("st.global.cg.v4."#ptx_type"  [%0], {%1, %2, %3, %4};" : : _B40C_ASM_PTR_(d_ptr + offset), #reg_mod(val.x), #reg_mod(val.y), #reg_mod(val.z), #reg_mod(val.w));	\
			}																																						\
		};																																							\
		template <> struct ModifiedStore<short_type##4, CS> {																											\
			__device__ __forceinline__ static void St(const short_type##4 &val, short_type##4* d_ptr, int offset) {														\
				asm("st.global.cs.v4."#ptx_type"  [%0], {%1, %2, %3, %4};" : : _B40C_ASM_PTR_(d_ptr + offset), #reg_mod(val.x), #reg_mod(val.y), #reg_mod(val.z), #reg_mod(val.w));	\
			}																																						\
		};																																							\
		template <> struct ModifiedStore<short_type##4, WB> {																											\
			__device__ __forceinline__ static void St(const short_type##4 &val, short_type##4* d_ptr, int offset) {														\
				asm("st.global.wb.v4."#ptx_type"  [%0], {%1, %2, %3, %4};" : : _B40C_ASM_PTR_(d_ptr + offset), #reg_mod(val.x), #reg_mod(val.y), #reg_mod(val.z), #reg_mod(val.w));	\
			}																																						\
		};

	// Cache-modified stores for built-in structures
	B40C_DEFINE_GLOBAL_STORE(char, signed char, char, s8, r)
	B40C_DEFINE_BASE_GLOBAL_STORE(signed char, s8, r)			// only need to define base: char2,char4, etc already defined from char
	B40C_DEFINE_GLOBAL_STORE(short, short, short, s16, r)
	B40C_DEFINE_GLOBAL_STORE(int, int, int, s32, r)
	B40C_DEFINE_GLOBAL_STORE(long, long, long, s64, l)
	B40C_DEFINE_GLOBAL_STORE(long long, long long, longlong, s64, l)
	B40C_DEFINE_GLOBAL_STORE(unsigned char, unsigned char, uchar, u8, r)
	B40C_DEFINE_GLOBAL_STORE(unsigned short, unsigned short, ushort, u16, r)
	B40C_DEFINE_GLOBAL_STORE(unsigned int, unsigned int, uint, u32, r)
	B40C_DEFINE_GLOBAL_STORE(unsigned long, unsigned long, ulong, u64, l)
	B40C_DEFINE_GLOBAL_STORE(unsigned long long, unsigned long long, ulonglong, u64, l)
	B40C_DEFINE_GLOBAL_STORE(float, float, float, f32, f)
	B40C_DEFINE_BASE_GLOBAL_STORE(double, f64, d)	// stores of vector-doubles don't compile
	
	#undef B40C_DEFINE_BASE_GLOBAL_STORE
	#undef B40C_DEFINE_GLOBAL_STORE

	// Workaround for the fact that the assembler reports an error when attempting to 
	// make vector stores of doubles.
	template <> struct ModifiedStore<double2, CG> {																												
		__device__ __forceinline__ static void St(const double2 &val, double2* d_ptr, int offset) {													
			asm("st.global.cg.f64 [%0], %1;" : : _B40C_ASM_PTR_(d_ptr + offset), "d"(val.x));																	
			asm("st.global.cg.f64 [%0], %1;" : : _B40C_ASM_PTR_(d_ptr + offset + 1), "d"(val.y));																	
		}																																							
	};																																								
	template <> struct ModifiedStore<double4, CG> {																												
		__device__ __forceinline__ static void St(const double4 &val, double4* d_ptr, int offset) {													
			asm("st.global.cg.f64 [%0], %1;" : : _B40C_ASM_PTR_(d_ptr + offset), "d"(val.x));																	
			asm("st.global.cg.f64 [%0], %1;" : : _B40C_ASM_PTR_(d_ptr + offset + 1), "d"(val.y));																	
			asm("st.global.cg.f64 [%0], %1;" : : _B40C_ASM_PTR_(d_ptr + offset + 2), "d"(val.z));																	
			asm("st.global.cg.f64 [%0], %1;" : : _B40C_ASM_PTR_(d_ptr + offset + 3), "d"(val.w));																	
		}																																							
	};																																								
	
	// NONE-modified store 
	template <typename T> struct ModifiedStore<T, NONE>
	{
		__device__ __forceinline__ static void St(const T &val, T* d_ptr, int offset) {
			d_ptr[offset] = val; 
		}
	};
	
	// NONE-modified store 
	template <> struct ModifiedStore<char, NONE>
	{
		__device__ __forceinline__ static void St(const signed char &val, char* d_ptr, int offset) {
			d_ptr[offset] = val; 
		}
	};
	
#else	// stores

	//
	// Nothing is cached in these architectures
	//
	
	// Store normally
	template <typename T, CacheModifier CACHE_MODIFIER> struct ModifiedStore
	{
		__device__ __forceinline__ static void St(const T &val, T* d_ptr, int offset) {
			d_ptr[offset] = val; 
		}
	};
	
	// Accomodate bizarre introduction of "signed" for char stores
	template <CacheModifier CACHE_MODIFIER> struct ModifiedStore<char, CACHE_MODIFIER>
	{
		__device__ __forceinline__ static void St(const signed char &val, char* d_ptr, int offset) {
			d_ptr[offset] = val; 
		}
	};

#endif	// stores
	



} // namespace b40c

