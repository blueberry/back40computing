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
 * Utilities for statically and dynamically inspecting CUDA device properties
 ******************************************************************************/

#pragma once

#include <cub/core/perror.cuh>

namespace cub {


/******************************************************************************
 * Macros for guiding compilation paths
 ******************************************************************************/

/**
 * CUDA architecture of the current compilation path
 */
#ifndef __CUDA_ARCH__
	#define __CUB_CUDA_ARCH__ 		0						// Host path
#else
	#define __CUB_CUDA_ARCH__ 		__CUDA_ARCH__			// Device path
#endif


/**
 * Invalid CUDA gpu device ordinal
 */
#define CUB_GPU_ORDINAL				(-1)



/******************************************************************************
 * Static device properties by SM architectural version
 ******************************************************************************/


/**
 * Structure for statically reporting CUDA device properties, parameterized by SM
 * architecture.
 */
template <int CUDA_ARCH>
struct DeviceProps;


/**
 * Device properties for SM10
 */
template <>
struct DeviceProps<100>
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
};


/**
 * Device properties for SM11
 */
template <>
struct DeviceProps<110> : DeviceProps<100> {};		// Derives from SM10


/**
 * Device properties for SM12
 */
template <>
struct DeviceProps<120>
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
};


/**
 * Device properties for SM13
 */
template <>
struct DeviceProps<130> : DeviceProps<120> {};		// Derives from SM12


/**
 * Device properties for SM20
 */
template <>
struct DeviceProps<200>
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
};


/**
 * Device properties for SM21
 */
template <>
struct DeviceProps<210> : DeviceProps<200> {};		// Derives from SM20



/**
 * Device properties for SM30
 */
template <>
struct DeviceProps<300>
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
};


/**
 * Unknown device properties
 */
template <int CUDA_ARCH>
struct DeviceProps : DeviceProps<300> {};			// Derives from SM30



/******************************************************************************
 * Dynamic device inspection (from host)
 ******************************************************************************/

/**
 * Empty Kernel
 */
template <typename T>
__global__ void EmptyKernel(void) { }





/**
 * Encapsulation of device properties for a specific device
 */
class CudaProperties
{
public:

	cudaError_t	init_error;		// cudaError_t associated with construction

	// Version information
	int sm_version;				// SM version X.YZ in XYZ integer form
	int ptx_version;			// PTX version X.YZ in XYZ integer form

	// Information about our target device
	int sm_count;				// Number of SMs
	int warp_threads;			// Number of threads per warp
	int smem_bank_bytes;		// Number of bytes per SM bank
	int smem_banks;				// Number of smem banks
	int smem_bytes;				// Smem bytes per SM
	int smem_alloc_unit;		// Smem segment size
	bool regs_by_block;			// Whether registers are allocated by CTA (or by warp)
	int reg_alloc_unit;
	int warp_alloc_unit;		// Granularity of warp allocation within the SM
	int max_sm_threads;			// Maximum number of threads per SM
	int max_sm_ctas;			// Maximum number of CTAs per SM
	int max_cta_threads;		// Maximum number of threads per CTA
	int max_sm_registers;		// Maximum number of registers per SM
	int max_sm_warps;			// Maximum number of warps per SM

public:

	/**
	 * Properties initializer
	 */
	template <typename DeviceProps>
	void InitProps()
	{
		warp_threads 		= DeviceProps::WARP_THREADS;
		smem_bank_bytes		= DeviceProps::SMEM_BANK_BYTES;
		smem_banks			= DeviceProps::SMEM_BANKS;
		smem_bytes			= DeviceProps::SMEM_BYTES;
		smem_alloc_unit		= DeviceProps::SMEM_ALLOC_UNIT;
		regs_by_block		= DeviceProps::REGS_BY_BLOCK;
		reg_alloc_unit		= DeviceProps::REG_ALLOC_UNIT;
		warp_alloc_unit		= DeviceProps::WARP_ALLOC_UNIT;
		max_sm_threads		= DeviceProps::MAX_SM_THREADS;
		max_sm_ctas			= DeviceProps::MAX_SM_CTAS;
		max_cta_threads		= DeviceProps::MAX_CTA_THREADS;
		max_sm_registers	= DeviceProps::MAX_SM_REGISTERS;
		max_sm_warps 		= max_sm_threads / warp_threads;
	}

	/**
	 * Initializer
	 */
	cudaError_t Init(int gpu_ordinal)
	{
		cudaError_t error = cudaSuccess;

		do {
			cudaDeviceProp device_props;
			if (error = Perror(cudaGetDeviceProperties(&device_props, gpu_ordinal),
				"cudaGetDeviceProperties failed", __FILE__, __LINE__)) break;
			sm_version = device_props.major * 100 + device_props.minor * 10;
			sm_count = device_props.multiProcessorCount;

			// Get SM version of compiled kernel assemblies
			cudaFuncAttributes flush_kernel_attrs;
			if (error = Perror(cudaFuncGetAttributes(&flush_kernel_attrs, EmptyKernel<void>),
				"cudaFuncGetAttributes failed", __FILE__, __LINE__)) break;
			ptx_version = flush_kernel_attrs.ptxVersion * 10;

			switch (sm_version) {
			case 100 :
				InitProps<DeviceProps<100> >();
				break;
			case 110 :
				InitProps<DeviceProps<110> >();
				break;
			case 120 :
				InitProps<DeviceProps<120> >();
				break;
			case 130 :
				InitProps<DeviceProps<130> >();
				break;
			case 200 :
				InitProps<DeviceProps<200> >();
				break;
			case 210 :
				InitProps<DeviceProps<210> >();
				break;
			case 300 :
				InitProps<DeviceProps<300> >();
				break;
			default:
				// Default to SM300
				InitProps<DeviceProps<100> >();
			}
		} while (0);

		return error;
	}


	/**
	 * Constructor.  Properties are retrieved for the current GPU ordinal.
	 */
	CudaProperties()
	{
		do {
			int gpu_ordinal;
			if (init_error = cudaGetDevice(&gpu_ordinal)) break;
			if (init_error = Init(gpu_ordinal)) break;
		} while (0);
	}

	/**
	 * Constructor.  Properties are retrieved for the specified GPU ordinal.
	 */
	CudaProperties(int gpu_ordinal)
	{
		init_error = Init(gpu_ordinal);
	}
};





} // namespace cub

