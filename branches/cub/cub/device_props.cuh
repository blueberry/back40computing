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
 * Static CUDA device properties by SM architectural version.
 *
 * "PTX_ARCH" reflects the PTX arch-id targeted by the active compiler pass
 * (or zero during the host pass).
 *
 * "DeviceProps" reflects the PTX architecture targeted by the active compiler
 * pass.  It provides useful compile-time statics within device code.  E.g.,:
 *
 *     __shared__ int[DeviceProps::WARP_THREADS];
 *
 *     int padded_offset = threadIdx.x + (threadIdx.x >> DeviceProps::SMEM_BANKS);
 *
 ******************************************************************************/

#pragma once

#include <cub/ns_umbrella.cuh>

CUB_NS_PREFIX
namespace cub {


/**
 * CUDA architecture-id targeted by the active compiler pass
 */
enum {
#ifndef __CUDA_ARCH__
	PTX_ARCH = 0,						// Host path
#else
	PTX_ARCH = __CUDA_ARCH__,			// Device path
#endif
};


/**
 * Structure for statically reporting CUDA device properties, parameterized by SM
 * architecture.
 */
template <int SM_ARCH>
struct StaticDeviceProps;


/**
 * Device properties for the arch-id targeted by the active compiler pass.
 */
struct DeviceProps : StaticDeviceProps<PTX_ARCH> {};


/**
 * Device properties for SM30
 */
template <>
struct StaticDeviceProps<300>
{
	enum {
		WARP_THREADS				= 32,			// 32 threads per warp
		SMEM_BANK_BYTES				= 4,			// 4 byte bank words
		SMEM_BANKS					= 32, 			// 32 banks
		SMEM_BYTES					= 48 * 1024,	// 48KB shared memory
		SMEM_ALLOC_UNIT				= 256,			// 256B smem allocation segment size
		REGS_BY_BLOCK				= false,		// Allocates registers by warp
		REG_ALLOC_UNIT				= 256,			// 256 registers allocated at a time per warp
		WARP_ALLOC_UNIT				= 4,			// Registers are allocated at a granularity of every 4 warps per CTA
		MAX_SM_THREADS				= 2048,			// 2K max threads per SM
		MAX_SM_CTAS					= 16,			// 16 max CTAs per SM
		MAX_CTA_THREADS				= 1024,			// 1024 max threads per CTA
		MAX_SM_REGISTERS			= 64 * 1024,	// 64K max registers per SM
	};

	// Callback utility
	template <typename T>
	static void Callback(T &target, int sm_version)
	{
		target.template Callback<StaticDeviceProps>();
	}
};


/**
 * Device properties for SM20
 */
template <>
struct StaticDeviceProps<200>
{
	enum {
		WARP_THREADS				= 32,			// 32 threads per warp
		SMEM_BANK_BYTES				= 4,			// 4 byte bank words
		SMEM_BANKS					= 32, 			// 32 banks
		SMEM_BYTES					= 48 * 1024,	// 48KB shared memory
		SMEM_ALLOC_UNIT				= 128,			// 128B smem allocation segment size
		REGS_BY_BLOCK				= false,		// Allocates registers by warp
		REG_ALLOC_UNIT				= 64,			// 64 registers allocated at a time per warp
		WARP_ALLOC_UNIT				= 2,			// Registers are allocated at a granularity of every 2 warps per CTA
		MAX_SM_THREADS				= 1536,			// 1536 max threads per SM
		MAX_SM_CTAS					= 8,			// 8 max CTAs per SM
		MAX_CTA_THREADS				= 1024,			// 1024 max threads per CTA
		MAX_SM_REGISTERS			= 32 * 1024,	// 32K max registers per SM
	};

	// Callback utility
	template <typename T>
	static void Callback(T &target, int sm_version)
	{
		if (sm_version > 200) {
			StaticDeviceProps<300>::Callback(target, sm_version);
		} else {
			target.template Callback<StaticDeviceProps>();
		}
	}
};


/**
 * Device properties for SM12
 */
template <>
struct StaticDeviceProps<120>
{
	enum {
		WARP_THREADS				= 32,			// 32 threads per warp
		SMEM_BANK_BYTES				= 4,			// 4 byte bank words
		SMEM_BANKS					= 16, 			// 16 banks
		SMEM_BYTES					= 16 * 1024,	// 16KB shared memory
		SMEM_ALLOC_UNIT				= 512,			// 512B smem allocation segment size
		REGS_BY_BLOCK				= true,			// Allocates registers by CTA
		REG_ALLOC_UNIT				= 512,			// 512 registers allocated at time per CTA
		WARP_ALLOC_UNIT				= 2,			// Registers are allocated at a granularity of every 2 warps per CTA
		MAX_SM_THREADS				= 1024,			// 1024 max threads per SM
		MAX_SM_CTAS					= 8,			// 8 max CTAs per SM
		MAX_CTA_THREADS				= 512,			// 512 max threads per CTA
		MAX_SM_REGISTERS			= 16 * 1024,	// 16K max registers per SM
	};

	// Callback utility
	template <typename T>
	static void Callback(T &target, int sm_version)
	{
		if (sm_version > 120) {
			StaticDeviceProps<200>::Callback(target, sm_version);
		} else {
			target.template Callback<StaticDeviceProps>();
		}
	}
};


/**
 * Device properties for SM10
 */
template <>
struct StaticDeviceProps<100>
{
	enum {
		WARP_THREADS				= 32,			// 32 threads per warp
		SMEM_BANK_BYTES				= 4,			// 4 byte bank words
		SMEM_BANKS					= 16, 			// 16 banks
		SMEM_BYTES					= 16 * 1024,	// 16KB shared memory
		SMEM_ALLOC_UNIT				= 512,			// 512B smem allocation segment size
		REGS_BY_BLOCK				= true,			// Allocates registers by CTA
		REG_ALLOC_UNIT				= 256,			// 256 registers allocated at time per CTA
		WARP_ALLOC_UNIT				= 2,			// Registers are allocated at a granularity of every 2 warps per CTA
		MAX_SM_THREADS				= 768,			// 768 max threads per SM
		MAX_SM_CTAS					= 8,			// 8 max CTAs per SM
		MAX_CTA_THREADS				= 512,			// 512 max threads per CTA
		MAX_SM_REGISTERS			= 8 * 1024,		// 8K max registers per SM
	};

	// Callback utility
	template <typename T>
	static void Callback(T &target, int sm_version)
	{
		if (sm_version > 100) {
			StaticDeviceProps<120>::Callback(target, sm_version);
		} else {
			target.template Callback<StaticDeviceProps>();
		}
	}
};


/**
 * Device properties for SM21
 */
template <>
struct StaticDeviceProps<210> : StaticDeviceProps<200> {};		// Derives from SM20

/**
 * Device properties for SM13
 */
template <>
struct StaticDeviceProps<130> : StaticDeviceProps<120> {};		// Derives from SM12

/**
 * Device properties for SM11
 */
template <>
struct StaticDeviceProps<110> : StaticDeviceProps<100> {};		// Derives from SM10

/**
 * Unknown device properties
 */
template <int SM_ARCH>
struct StaticDeviceProps : StaticDeviceProps<100> {};			// Derives from SM10




} // namespace cub
CUB_NS_POSTFIX
