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
 * Thanks!
 * 
 ******************************************************************************/

/******************************************************************************
 * Kernel utilities for loading types through global memory with cache modifiers
 ******************************************************************************/

#pragma once

#include <b40c/util/cuda_properties.cuh>
#include <b40c/util/vector_types.cuh>

namespace b40c {
namespace util {


/**
 * Enumeration of data movement cache modifiers.
 */
namespace ld {
enum CacheModifier {
	NONE,				// default (currently CA)
	CG,					// cache global
	CA,					// cache all
	CS, 				// cache streaming

	LIMIT
};
} // namespace ld

// Mooch
#define CacheModifierToString(modifier)	(	(modifier == b40c::util::ld::NONE) ? 	"NONE" :	\
											(modifier == b40c::util::ld::CG) ? 		"CG" :		\
											(modifier == b40c::util::ld::CA) ? 		"CA" :		\
											(modifier == b40c::util::ld::CS) ? 		"CS" :		\
											(modifier == b40c::util::st::NONE) ? 	"NONE" :	\
											(modifier == b40c::util::st::CG) ? 		"CG" :		\
											(modifier == b40c::util::st::WB) ? 		"WB" :		\
											(modifier == b40c::util::st::CS) ? 		"CS" :		\
																					"<ERROR>")


/**
 * Routines for modified loads through cache.  We use structs specialized by value 
 * type and cache-modifier to implement load operations
 */
template <typename T, ld::CacheModifier CACHE_MODIFIER> struct ModifiedLoad;

#if __CUDA_ARCH__ >= 200

	/**
	 * Defines specialized load ops for only the base type 
	 */
	#define B40C_DEFINE_BASE_GLOBAL_LOAD(base_type, ptx_type, reg_mod, cast_type)																					\
		template <> struct ModifiedLoad<base_type, ld::NONE> {																										\
			__device__ __forceinline__ static void Ld(base_type &val, base_type* d_ptr, size_t offset) {															\
				val = d_ptr[offset];																																\
			}																																						\
		};																																							\
		template <> struct ModifiedLoad<base_type, ld::CG> {																										\
			__device__ __forceinline__ static void Ld(base_type &val, base_type* d_ptr, size_t offset) {															\
				asm("ld.global.cg."#ptx_type" %0, [%1];" : "="#reg_mod(*reinterpret_cast<cast_type*>(&val)) : _B40C_ASM_PTR_(d_ptr + offset));						\
			}																																						\
		};																																							\
		template <> struct ModifiedLoad<base_type, ld::CS> {																										\
			__device__ __forceinline__ static void Ld(base_type &val, base_type* d_ptr, size_t offset) {															\
				asm("ld.global.cs."#ptx_type" %0, [%1];" : "="#reg_mod(*reinterpret_cast<cast_type*>(&val)) : _B40C_ASM_PTR_(d_ptr + offset));						\
			}																																						\
		};																																							\
		template <> struct ModifiedLoad<base_type, ld::CA> {																										\
			__device__ __forceinline__ static void Ld(base_type &val, base_type* d_ptr, size_t offset) {															\
				asm("ld.global.ca."#ptx_type" %0, [%1];" : "="#reg_mod(*reinterpret_cast<cast_type*>(&val)) : _B40C_ASM_PTR_(d_ptr + offset));						\
			}																																						\
		};																																								


	/**
	 * Defines specialized load ops for both the base type and for its derivative vector types
	 */
	#define B40C_DEFINE_GLOBAL_LOAD(base_type, dest_type, short_type, ptx_type, reg_mod, cast_type)																	\
		template <> struct ModifiedLoad<base_type, ld::NONE> {																										\
			__device__ __forceinline__ static void Ld(dest_type &val, base_type* d_ptr, size_t offset) {															\
				val = d_ptr[offset];																																\
			}																																						\
		};																																							\
		template <> struct ModifiedLoad<base_type, ld::CG> {																										\
			__device__ __forceinline__ static void Ld(dest_type &val, base_type* d_ptr, size_t offset) {															\
				asm("ld.global.cg."#ptx_type" %0, [%1];" : "="#reg_mod(*reinterpret_cast<cast_type*>(&val)) : _B40C_ASM_PTR_(d_ptr + offset));						\
			}																																						\
		};																																							\
		template <> struct ModifiedLoad<base_type, ld::CS> {																										\
			__device__ __forceinline__ static void Ld(dest_type &val, base_type* d_ptr, size_t offset) {															\
				asm("ld.global.cs."#ptx_type" %0, [%1];" : "="#reg_mod(*reinterpret_cast<cast_type*>(&val)) : _B40C_ASM_PTR_(d_ptr + offset));						\
			}																																						\
		};																																							\
		template <> struct ModifiedLoad<base_type, ld::CA> {																										\
			__device__ __forceinline__ static void Ld(dest_type &val, base_type* d_ptr, size_t offset) {															\
				asm("ld.global.ca."#ptx_type" %0, [%1];" : "="#reg_mod(*reinterpret_cast<cast_type*>(&val)) : _B40C_ASM_PTR_(d_ptr + offset));						\
			}																																						\
		};																																							\
		template <> struct ModifiedLoad<short_type##1, ld::NONE> {																									\
			__device__ __forceinline__ static void Ld(short_type##1 &val, short_type##1* d_ptr, size_t offset) {													\
				val = d_ptr[offset];																																\
			}																																						\
		};																																							\
		template <> struct ModifiedLoad<short_type##1, ld::CG> {																									\
			__device__ __forceinline__ static void Ld(short_type##1 &val, short_type##1* d_ptr, size_t offset) {													\
				asm("ld.global.cg."#ptx_type" %0, [%1];" : "="#reg_mod(*reinterpret_cast<cast_type*>(&val.x)) : _B40C_ASM_PTR_(d_ptr + offset));					\
			}																																						\
		};																																							\
		template <> struct ModifiedLoad<short_type##1, ld::CS> {																									\
			__device__ __forceinline__ static void Ld(short_type##1 &val, short_type##1* d_ptr, size_t offset) {													\
				asm("ld.global.cs."#ptx_type" %0, [%1];" : "="#reg_mod(*reinterpret_cast<cast_type*>(&val.x)) : _B40C_ASM_PTR_(d_ptr + offset));					\
			}																																						\
		};																																							\
		template <> struct ModifiedLoad<short_type##1, ld::CA> {																									\
			__device__ __forceinline__ static void Ld(short_type##1 &val, short_type##1* d_ptr, size_t offset) {													\
				asm("ld.global.ca."#ptx_type" %0, [%1];" : "="#reg_mod(*reinterpret_cast<cast_type*>(&val.x)) : _B40C_ASM_PTR_(d_ptr + offset));					\
			}																																						\
		};																																							\
		template <> struct ModifiedLoad<short_type##2, ld::NONE> {																									\
			__device__ __forceinline__ static void Ld(short_type##2 &val, short_type##2* d_ptr, size_t offset) {													\
				val = d_ptr[offset];																																\
			}																																						\
		};																																							\
		template <> struct ModifiedLoad<short_type##2, ld::CG> {																									\
			__device__ __forceinline__ static void Ld(short_type##2 &val, short_type##2* d_ptr, size_t offset) {													\
				asm("ld.global.cg.v2."#ptx_type" {%0, %1}, [%2];" : "="#reg_mod(*reinterpret_cast<cast_type*>(&val.x)), "="#reg_mod(*reinterpret_cast<cast_type*>(&val.y)) : _B40C_ASM_PTR_(d_ptr + offset));						\
			}																																						\
		};																																							\
		template <> struct ModifiedLoad<short_type##2, ld::CS> {																									\
			__device__ __forceinline__ static void Ld(short_type##2 &val, short_type##2* d_ptr, size_t offset) {													\
				asm("ld.global.cs.v2."#ptx_type" {%0, %1}, [%2];" : "="#reg_mod(*reinterpret_cast<cast_type*>(&val.x)), "="#reg_mod(*reinterpret_cast<cast_type*>(&val.y)) : _B40C_ASM_PTR_(d_ptr + offset));						\
			}																																						\
		};																																							\
		template <> struct ModifiedLoad<short_type##2, ld::CA> {																									\
			__device__ __forceinline__ static void Ld(short_type##2 &val, short_type##2* d_ptr, size_t offset) {													\
				asm("ld.global.ca.v2."#ptx_type" {%0, %1}, [%2];" : "="#reg_mod(*reinterpret_cast<cast_type*>(&val.x)), "="#reg_mod(*reinterpret_cast<cast_type*>(&val.y)) : _B40C_ASM_PTR_(d_ptr + offset));						\
			}																																						\
		};

	/**
	 * Defines specialized load ops for the vec-4 derivative vector types
	 */
	#define B40C_DEFINE_GLOBAL_QUAD_LOAD(base_type, dest_type, short_type, ptx_type, reg_mod, cast_type)																						\
		template <> struct ModifiedLoad<short_type##4, ld::NONE> {																																\
			__device__ __forceinline__ static void Ld(short_type##4 &val, short_type##4* d_ptr, size_t offset) {																				\
				val = d_ptr[offset];																																							\
			}																																													\
		};																																														\
		template <> struct ModifiedLoad<short_type##4, ld::CG> {																																\
			__device__ __forceinline__ static void Ld(short_type##4 &val, short_type##4* d_ptr, size_t offset) {																				\
				asm("ld.global.cg.v4."#ptx_type" {%0, %1, %2, %3}, [%4];" : "="#reg_mod(*reinterpret_cast<cast_type*>(&val.x)), "="#reg_mod(*reinterpret_cast<cast_type*>(&val.y)), "="#reg_mod(*reinterpret_cast<cast_type*>(&val.z)), "="#reg_mod(*reinterpret_cast<cast_type*>(&val.w)) : _B40C_ASM_PTR_(d_ptr + offset));	\
			}																																													\
		};																																														\
		template <> struct ModifiedLoad<short_type##4, ld::CS> {																																\
			__device__ __forceinline__ static void Ld(short_type##4 &val, short_type##4* d_ptr, size_t offset) {																				\
				asm("ld.global.cs.v4."#ptx_type" {%0, %1, %2, %3}, [%4];" : "="#reg_mod(*reinterpret_cast<cast_type*>(&val.x)), "="#reg_mod(*reinterpret_cast<cast_type*>(&val.y)), "="#reg_mod(*reinterpret_cast<cast_type*>(&val.z)), "="#reg_mod(*reinterpret_cast<cast_type*>(&val.w)) : _B40C_ASM_PTR_(d_ptr + offset));	\
			}																																													\
		};																																														\
		template <> struct ModifiedLoad<short_type##4, ld::CA> {																																\
			__device__ __forceinline__ static void Ld(short_type##4 &val, short_type##4* d_ptr, size_t offset) {																				\
				asm("ld.global.ca.v4."#ptx_type" {%0, %1, %2, %3}, [%4];" : "="#reg_mod(*reinterpret_cast<cast_type*>(&val.x)), "="#reg_mod(*reinterpret_cast<cast_type*>(&val.y)), "="#reg_mod(*reinterpret_cast<cast_type*>(&val.z)), "="#reg_mod(*reinterpret_cast<cast_type*>(&val.w)) : _B40C_ASM_PTR_(d_ptr + offset));	\
			}																																													\
		};

	// Cache-modified loads for built-in structures
	B40C_DEFINE_GLOBAL_LOAD(char, signed char, char, s8, r, unsigned int)
	B40C_DEFINE_GLOBAL_LOAD(short, short, short, s16, r, unsigned int)
	B40C_DEFINE_GLOBAL_LOAD(int, int, int, s32, r, unsigned int)
	B40C_DEFINE_GLOBAL_LOAD(long long, long long, longlong, s64, l, long long)
	B40C_DEFINE_GLOBAL_LOAD(unsigned char, unsigned char, uchar, u8, r, unsigned int)
	B40C_DEFINE_GLOBAL_LOAD(unsigned short, unsigned short, ushort, u16, r, unsigned int)
	B40C_DEFINE_GLOBAL_LOAD(unsigned int, unsigned int, uint, u32, r, unsigned int)
	B40C_DEFINE_GLOBAL_LOAD(unsigned long long, unsigned long long, ulonglong, u64, l, unsigned long long)
	B40C_DEFINE_GLOBAL_LOAD(float, float, float, f32, f, float)

#if _B40C_LP64_
	B40C_DEFINE_GLOBAL_LOAD(long, long, long, s64, l, long)
	B40C_DEFINE_GLOBAL_LOAD(unsigned long, unsigned long, ulong, u64, l, unsigned long)
#else 	// _B40C_LP64_
	B40C_DEFINE_GLOBAL_LOAD(long, long, long, s32, r, long)
	B40C_DEFINE_GLOBAL_LOAD(unsigned long, unsigned long, ulong, u32, r, unsigned long)
#endif	// _B40C_LP64_

	// Cache-modified quad-loads for all 4-byte (and smaller) structures
	B40C_DEFINE_GLOBAL_QUAD_LOAD(char, signed char, char, s8, r, unsigned int)
	B40C_DEFINE_GLOBAL_QUAD_LOAD(short, short, short, s16, r, unsigned int)
	B40C_DEFINE_GLOBAL_QUAD_LOAD(int, int, int, s32, r, unsigned int)
	B40C_DEFINE_GLOBAL_QUAD_LOAD(unsigned char, unsigned char, uchar, u8, r, unsigned int)
	B40C_DEFINE_GLOBAL_QUAD_LOAD(unsigned short, unsigned short, ushort, u16, r, unsigned int)
	B40C_DEFINE_GLOBAL_QUAD_LOAD(unsigned int, unsigned int, uint, u32, r, unsigned int)
	B40C_DEFINE_GLOBAL_QUAD_LOAD(float, float, float, f32, f, float)

	B40C_DEFINE_BASE_GLOBAL_LOAD(signed char, s8, r, unsigned int)	// only need to define base: char2,char4, etc already defined from char

#if !_B40C_LP64_
	B40C_DEFINE_GLOBAL_QUAD_LOAD(long, long, long, s32, r, long)
	B40C_DEFINE_GLOBAL_QUAD_LOAD(unsigned long, unsigned long, ulong, u32, r, unsigned long)
#endif	// _B40C_LP64_

	
	// Workaround for the fact that the assembler reports an error when attempting to 
	// make vector loads of doubles.

	B40C_DEFINE_BASE_GLOBAL_LOAD(double, f64, d, double)

	template <ld::CacheModifier CACHE_MODIFIER>
	struct ModifiedLoad<double2, CACHE_MODIFIER> {
		__device__ __forceinline__ static void Ld(double2 &val, double2* d_ptr, size_t offset) {
			ModifiedLoad<double, CACHE_MODIFIER>::Ld(val.x, reinterpret_cast<double*>(d_ptr + offset), 0);
			ModifiedLoad<double, CACHE_MODIFIER>::Ld(val.y, reinterpret_cast<double*>(d_ptr + offset), 1);
		}
	};

	// Vec-4 loads for 64-bit types are implemented as two vec-2 loads

	template <ld::CacheModifier CACHE_MODIFIER>
	struct ModifiedLoad<double4, CACHE_MODIFIER> {
		__device__ __forceinline__ static void Ld(double4 &val, double4* d_ptr, size_t offset) {
			ModifiedLoad<double2, CACHE_MODIFIER>::Ld(*reinterpret_cast<double2*>(&val.x), reinterpret_cast<double2*>(d_ptr + offset), 0);
			ModifiedLoad<double2, CACHE_MODIFIER>::Ld(*reinterpret_cast<double2*>(&val.z), reinterpret_cast<double2*>(d_ptr + offset), 1);
		}																																							
	};																																								

	template <ld::CacheModifier CACHE_MODIFIER>
	struct ModifiedLoad<ulonglong4, CACHE_MODIFIER> {
		__device__ __forceinline__ static void Ld(ulonglong4 &val, ulonglong4* d_ptr, size_t offset) {
			ModifiedLoad<ulonglong2, CACHE_MODIFIER>::Ld(*reinterpret_cast<ulonglong2*>(&val.x), reinterpret_cast<ulonglong2*>(d_ptr + offset), 0);
			ModifiedLoad<ulonglong2, CACHE_MODIFIER>::Ld(*reinterpret_cast<ulonglong2*>(&val.z), reinterpret_cast<ulonglong2*>(d_ptr + offset), 1);
		}
	};

	template <ld::CacheModifier CACHE_MODIFIER>
	struct ModifiedLoad<longlong4, CACHE_MODIFIER> {
		__device__ __forceinline__ static void Ld(longlong4 &val, longlong4* d_ptr, size_t offset) {
			ModifiedLoad<longlong2, CACHE_MODIFIER>::Ld(*reinterpret_cast<longlong2*>(&val.x), reinterpret_cast<longlong2*>(d_ptr + offset), 0);
			ModifiedLoad<longlong2, CACHE_MODIFIER>::Ld(*reinterpret_cast<longlong2*>(&val.z), reinterpret_cast<longlong2*>(d_ptr + offset), 1);
		}																																							
	};																																								

#if _B40C_LP64_
	template <ld::CacheModifier CACHE_MODIFIER>
	struct ModifiedLoad<long4, CACHE_MODIFIER> {
		__device__ __forceinline__ static void Ld(long4 &val, long4* d_ptr, size_t offset) {
			ModifiedLoad<long2, CACHE_MODIFIER>::Ld(*reinterpret_cast<long2*>(&val.x), reinterpret_cast<long2*>(d_ptr + offset), 0);
			ModifiedLoad<long2, CACHE_MODIFIER>::Ld(*reinterpret_cast<long2*>(&val.z), reinterpret_cast<long2*>(d_ptr + offset), 1);
		}																																							
	};																																								

	template <ld::CacheModifier CACHE_MODIFIER>
	struct ModifiedLoad<ulong4, CACHE_MODIFIER> {
		__device__ __forceinline__ static void Ld(ulong4 &val, ulong4* d_ptr, size_t offset) {
			ModifiedLoad<ulong2, CACHE_MODIFIER>::Ld(*reinterpret_cast<ulong2*>(&val.x), reinterpret_cast<ulong2*>(d_ptr + offset), 0);
			ModifiedLoad<ulong2, CACHE_MODIFIER>::Ld(*reinterpret_cast<ulong2*>(&val.z), reinterpret_cast<ulong2*>(d_ptr + offset), 1);
		}																																							
	};																																								
#endif	// _B40C_LP64_

	
	#undef B40C_DEFINE_BASE_GLOBAL_QUAD_LOAD
	#undef B40C_DEFINE_BASE_GLOBAL_LOAD
	#undef B40C_DEFINE_GLOBAL_LOAD


#else // __CUDA_ARCH__

	//
	// Nothing is cached in these architectures
	//
	
	// Load normally
	template <typename T, ld::CacheModifier CACHE_MODIFIER> struct ModifiedLoad
	{
		template <typename SizeT>
		__device__ __forceinline__ static void Ld(T &val, T* d_ptr, SizeT offset) {
			val = d_ptr[offset]; 
		}
	};
	
#endif	// __CUDA_ARCH__


	
	
/**
 * Empty default transform function (leaves non-in_bounds values as they were)
 */
template <typename T>
__device__ __forceinline__ void NopLoadTransform(T &val, bool in_bounds) {}
	

/**
 * Load a tile of items
 */
template <
	typename T,													// Type to load
	typename SizeT,												// Integer type for indexing into global arrays
	int LOG_LOADS_PER_TILE, 									// Number of vector loads (log)
	int LOG_LOAD_VEC_SIZE,										// Number of items per vector load (log)
	int ACTIVE_THREADS,											// Active threads that will be loading
	ld::CacheModifier CACHE_MODIFIER,							// Cache modifier (e.g., CA/CG/CS/NONE/etc.)
	bool UNGUARDED_IO,											// Whether or not bounds-checking is to be done
	void Transform(T&, bool) = NopLoadTransform<T> > 			// Assignment function to transform the loaded value (can be used assign default values for items deemed not in bounds)
		struct LoadTile;


/**
 * Load of a tile of items using unguarded loads 
 */
template <
	typename T,
	typename SizeT,
	int LOG_LOADS_PER_TILE, 
	int LOG_LOAD_VEC_SIZE,
	int ACTIVE_THREADS,
	ld::CacheModifier CACHE_MODIFIER,
	void Transform(T&, bool)>
struct LoadTile <T, SizeT, LOG_LOADS_PER_TILE, LOG_LOAD_VEC_SIZE, ACTIVE_THREADS, CACHE_MODIFIER, true, Transform>
{
	static const int LOADS_PER_TILE = 1 << LOG_LOADS_PER_TILE;
	static const int LOAD_VEC_SIZE = 1 << LOG_LOAD_VEC_SIZE;
	
	// Aliased vector type
	typedef typename VecType<T, LOAD_VEC_SIZE>::Type VectorType; 		

	// Next vec element
	template <int LOAD, int VEC, int __dummy = 0>
	struct Iterate 
	{
		static __device__ __forceinline__ void Invoke(
			T data[][LOAD_VEC_SIZE],
			VectorType vectors[], 
			VectorType *d_in_vectors) 
		{
			Transform(data[LOAD][VEC], true);	// Apply transform function with in_bounds = true 
			Iterate<LOAD, VEC + 1>::Invoke(data, vectors, d_in_vectors);
		}
	};

	// First vec element
	template <int LOAD, int __dummy>
	struct Iterate<LOAD, 0, __dummy> 
	{
		static __device__ __forceinline__ void Invoke(
			T data[][LOAD_VEC_SIZE],
			VectorType vectors[], 
			VectorType *d_in_vectors) 
		{
			ModifiedLoad<VectorType, CACHE_MODIFIER>::Ld(vectors[LOAD], d_in_vectors, (LOAD * ACTIVE_THREADS));
			Transform(data[LOAD][0], true);		// Apply transform function with in_bounds = true 
			Iterate<LOAD, 1>::Invoke(data, vectors, d_in_vectors);
		}
	};

	// Next load
	template <int LOAD, int __dummy>
	struct Iterate<LOAD, LOAD_VEC_SIZE, __dummy> 
	{
		static __device__ __forceinline__ void Invoke(
			T data[][LOAD_VEC_SIZE],
			VectorType vectors[], 
			VectorType *d_in_vectors) 
		{
			Iterate<LOAD + 1, 0>::Invoke(data, vectors, d_in_vectors);
		}
	};
	
	// Terminate
	template <int __dummy>
	struct Iterate<LOADS_PER_TILE, 0, __dummy> 
	{
		static __device__ __forceinline__ void Invoke(
			T data[][LOAD_VEC_SIZE],
			VectorType vectors[], 
			VectorType *d_in_vectors) {} 
	};

	// Interface
	static __device__ __forceinline__ void Invoke(
		T data[][LOAD_VEC_SIZE],
		T *d_in,
		SizeT cta_offset,
		const SizeT &out_of_bounds)
	{
		// Use an aliased pointer to keys array to perform built-in vector loads
		VectorType *vectors = (VectorType *) data;
		VectorType *d_in_vectors = (VectorType *) (d_in + cta_offset + (threadIdx.x << LOG_LOAD_VEC_SIZE));
		Iterate<0,0>::Invoke(data, vectors, d_in_vectors);
	} 
};
	

/**
 * Load of a tile of items using guarded loads 
 */
template <
	typename T,
	typename SizeT,
	int LOG_LOADS_PER_TILE, 
	int LOG_LOAD_VEC_SIZE,
	int ACTIVE_THREADS,
	ld::CacheModifier CACHE_MODIFIER,
	void Transform(T&, bool)>
struct LoadTile <T, SizeT, LOG_LOADS_PER_TILE, LOG_LOAD_VEC_SIZE, ACTIVE_THREADS, CACHE_MODIFIER, false, Transform>
{
	static const int LOADS_PER_TILE = 1 << LOG_LOADS_PER_TILE;
	static const int LOAD_VEC_SIZE = 1 << LOG_LOAD_VEC_SIZE;

	// Iterate over vec-elements
	template <int LOAD, int VEC>
	struct Iterate 
	{
		static __device__ __forceinline__ void Invoke(
			T data[][LOAD_VEC_SIZE],
			T *d_in,
			SizeT cta_offset,
			SizeT out_of_bounds)
		{
			SizeT thread_offset = cta_offset + (threadIdx.x << LOG_LOAD_VEC_SIZE) + VEC;

			if (thread_offset < out_of_bounds) {
				ModifiedLoad<T, CACHE_MODIFIER>::Ld(data[LOAD][VEC], d_in, thread_offset);
				Transform(data[LOAD][VEC], true);
			} else {
				Transform(data[LOAD][VEC], false);	// !in_bounds
			}
			
			Iterate<LOAD, VEC + 1>::Invoke(data, d_in, cta_offset, out_of_bounds);
		}
	};

	// Iterate over loads
	template <int LOAD>
	struct Iterate<LOAD, LOAD_VEC_SIZE> 
	{
		static __device__ __forceinline__ void Invoke(
			T data[][LOAD_VEC_SIZE],
			T *d_in,
			SizeT cta_offset,
			SizeT out_of_bounds)
		{
			Iterate<LOAD + 1, 0>::Invoke(
				data, d_in, cta_offset + (ACTIVE_THREADS << LOG_LOAD_VEC_SIZE), out_of_bounds);
		}
	};
	
	// Terminate
	template <int VEC>
	struct Iterate<LOADS_PER_TILE, VEC> 
	{
		static __device__ __forceinline__ void Invoke(
			T data[][LOAD_VEC_SIZE],
			T *d_in,
			SizeT cta_offset,
			SizeT out_of_bounds) {}
	};

	// Interface
	static __device__ __forceinline__ void Invoke(
		T data[][LOAD_VEC_SIZE],
		T *d_in,
		SizeT cta_offset,
		SizeT out_of_bounds)
	{
		Iterate<0, 0>::Invoke(data, d_in, cta_offset, out_of_bounds);
	} 
};


/**
 * Initialize a tile of items
 */
template <
	typename T,
	int LOG_LOADS_PER_TILE,
	int LOG_LOAD_VEC_SIZE,
	T Default()>
struct InitializeTile
{
	static const int LOADS_PER_TILE = 1 << LOG_LOADS_PER_TILE;
	static const int LOAD_VEC_SIZE = 1 << LOG_LOAD_VEC_SIZE;

	// Iterate over vec-elements
	template <int LOAD, int VEC>
	struct Iterate
	{
		static __device__ __forceinline__ void Invoke(T data[][LOAD_VEC_SIZE])
		{
			data[LOAD][VEC] = Default();
			Iterate<LOAD, VEC + 1>::Invoke(data);
		}
	};

	// Iterate over loads
	template <int LOAD>
	struct Iterate<LOAD, LOAD_VEC_SIZE>
	{
		static __device__ __forceinline__ void Invoke(T data[][LOAD_VEC_SIZE])
		{
			Iterate<LOAD + 1, 0>::Invoke(data);
		}
	};

	// Terminate
	template <int VEC>
	struct Iterate<LOADS_PER_TILE, VEC>
	{
		static __device__ __forceinline__ void Invoke(T data[][LOAD_VEC_SIZE]) {}
	};

	// Interface
	static __device__ __forceinline__ void Invoke(T data[][LOAD_VEC_SIZE])
	{
		Iterate<0, 0>::Invoke(data, d_in, cta_offset, out_of_bounds);
	}
};



} // namespace util
} // namespace b40c

