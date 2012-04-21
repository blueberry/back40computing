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
 * Utilities for statically and dynamically querying CUDA device properties
 ******************************************************************************/

#pragma once

#include <cub/type_utils.cuh>

namespace cub {


/******************************************************************************
 * Macros for guiding compilation paths
 ******************************************************************************/

/**
 * CUDA architecture of the current compilation path
 */
#ifndef __CUDA_ARCH__
	#define __CUB_CUDA_ARCH__ 0						// Host path
#else
	#define __CUB_CUDA_ARCH__ __CUDA_ARCH__			// Device path
#endif


/**
 * Invalid CUDA device ordinal
 */
#define CUB_INVALID_DEVICE				(-1)



/******************************************************************************
 * Static device properties by SM architectural version
 ******************************************************************************/


/**
 * Structure for statically reporting CUDA device properties, parameterized by SM
 * architecture.
 */
template <int CUDA_ARCH = __CUB_CUDA_ARCH__>
struct DeviceProps : DeviceProps<300> {};			// Newer defaults to SM30


/**
 * Device properties for SM10
 */
template <>
struct DeviceProps<100>
{
	enum {
		CUDA_ARCH					= 100,
		WARP_THREADS				= 32,			// 32 threads per warp
		SMEM_BANK_BYTES				= 4,			// 4 byte bank words
		SMEM_BANKS					= 16, 			// 16 banks
		SMEM_BYTES					= 16 * 1024,	// 16KB shared memory
		SMEM_SEG					= 512,			// 512B smem allocation segment size
		MAX_SM_THREADS				= 768,			// 768 max threads per SM
		MAX_SM_CTAS					= 8,			// 8 max CTAs per SM
		MAX_CTA_THREADS				= 512,			// 512 max threads per CTA
		MAX_SM_REGISTERS			= 8 * 1024,		// 8K max registers per SM

		// Derived properties
		LOG_WARP_THREADS			= Log2<WARP_THREADS>::VALUE,
		LOG_SMEM_BANK_BYTES			= Log2<SMEM_BANK_BYTES>::VALUE,
		LOG_SMEM_BANKS				= Log2<SMEM_BANKS>::VALUE,
		LOG_MAX_CTA_THREADS			= Log2<MAX_CTA_THREADS>::VALUE,
	};
};


/**
 * Device properties for SM11
 */
template <>
struct DeviceProps<110> : DeviceProps<100>
{
	enum {
		CUDA_ARCH					= 110,
	};
};


/**
 * Device properties for SM12
 */
template <>
struct DeviceProps<120>
{
	enum {
		CUDA_ARCH					= 120,
		WARP_THREADS				= 32,			// 32 threads per warp
		SMEM_BANK_BYTES				= 4,			// 4 byte bank words
		SMEM_BANKS					= 16, 			// 16 banks
		SMEM_BYTES					= 16 * 1024,	// 16KB shared memory
		SMEM_SEG					= 512,			// 512B smem allocation segment size
		MAX_SM_THREADS				= 1024,			// 1024 max threads per SM
		MAX_SM_CTAS					= 8,			// 8 max CTAs per SM
		MAX_CTA_THREADS				= 512,			// 512 max threads per CTA
		MAX_SM_REGISTERS			= 16 * 1024,	// 16K max registers per SM

		// Derived properties
		LOG_WARP_THREADS			= Log2<WARP_THREADS>::VALUE,
		LOG_SMEM_BANK_BYTES			= Log2<SMEM_BANK_BYTES>::VALUE,
		LOG_SMEM_BANKS				= Log2<SMEM_BANKS>::VALUE,
		LOG_MAX_CTA_THREADS			= Log2<MAX_CTA_THREADS>::VALUE,
	};
};


/**
 * Device properties for SM13
 */
template <>
struct DeviceProps<130> : DeviceProps<120>
{
	enum {
		CUDA_ARCH					= 130,
	};
};


/**
 * Device properties for SM20
 */
template <>
struct DeviceProps<200>
{
	enum {
		CUDA_ARCH					= 200,
		WARP_THREADS				= 32,			// 32 threads per warp
		SMEM_BANK_BYTES				= 4,			// 4 byte bank words
		SMEM_BANKS					= 32, 			// 32 banks
		SMEM_BYTES					= 48 * 1024,	// 48KB shared memory
		SMEM_SEG					= 128,			// 128B smem allocation segment size
		MAX_SM_THREADS				= 1536,			// 1536 max threads per SM
		MAX_SM_CTAS					= 8,			// 8 max CTAs per SM
		MAX_CTA_THREADS				= 1024,			// 1024 max threads per CTA
		MAX_SM_REGISTERS			= 32 * 1024,	// 32K max registers per SM

		// Derived properties
		LOG_WARP_THREADS			= Log2<WARP_THREADS>::VALUE,
		LOG_SMEM_BANK_BYTES			= Log2<SMEM_BANK_BYTES>::VALUE,
		LOG_SMEM_BANKS				= Log2<SMEM_BANKS>::VALUE,
		LOG_MAX_CTA_THREADS			= Log2<MAX_CTA_THREADS>::VALUE,
	};
};


/**
 * Device properties for SM21
 */
template <>
struct DeviceProps<210> : DeviceProps<200>
{
	enum {
		CUDA_ARCH					= 210,
	};
};



/**
 * Device properties for SM30
 */
template <>
struct DeviceProps<300>
{
	enum {
		CUDA_ARCH					= 300,
		WARP_THREADS				= 32,			// 32 threads per warp
		SMEM_BANK_BYTES				= 4,			// 4 byte bank words
		SMEM_BANKS					= 32, 			// 32 banks
		SMEM_BYTES					= 48 * 1024,	// 48KB shared memory
		SMEM_SEG					= 256,			// 256B smem allocation segment size
		MAX_SM_THREADS				= 2048,			// 2K max threads per SM
		MAX_SM_CTAS					= 16,			// 16 max CTAs per SM
		MAX_CTA_THREADS				= 1024,			// 1024 max threads per CTA
		MAX_SM_REGISTERS			= 64 * 1024,	// 64K max registers per SM

		// Derived properties
		LOG_WARP_THREADS			= Log2<WARP_THREADS>::VALUE,
		LOG_SMEM_BANK_BYTES			= Log2<SMEM_BANK_BYTES>::VALUE,
		LOG_SMEM_BANKS				= Log2<SMEM_BANKS>::VALUE,
		LOG_MAX_CTA_THREADS			= Log2<MAX_CTA_THREADS>::VALUE,
	};
};




/******************************************************************************
 * Dynamic device inspection (from host)
 ******************************************************************************/

/**
 * Empty Kernel
 */
template <typename T>
__global__ void FlushKernel(void) { }





/**
 * Encapsulation of device properties for a specific device
 */
class CudaProperties
{
public:

	// Information about our target device
	cudaDeviceProp 		device_props;			// CUDA properties structure
	int 				device_sm_version;		// SM version X.YZ in XYZ integer form

	// Information about our kernel assembly
	int 				kernel_ptx_version;		// PTX version X.YZ in XYZ integer form

public:

	/**
	 * Constructor.  Properties are retrieved for the current GPU ordinal.
	 */
	CudaProperties()
	{
		// Get current device properties
		int current_device;
		cudaGetDevice(&current_device);
		cudaGetDeviceProperties(&device_props, current_device);
		device_sm_version = device_props.major * 100 + device_props.minor * 10;

		// Get SM version of compiled kernel assemblies
		cudaFuncAttributes flush_kernel_attrs;
		cudaFuncGetAttributes(&flush_kernel_attrs, FlushKernel<void>);
		kernel_ptx_version = flush_kernel_attrs.ptxVersion * 10;
	}

	/**
	 * Constructor.  Properties are retrieved for the specified GPU ordinal.
	 */
	CudaProperties(int gpu)
	{
		// Get current device properties
		cudaGetDeviceProperties(&device_props, gpu);
		device_sm_version = device_props.major * 100 + device_props.minor * 10;

		// Get SM version of compiled kernel assemblies
		cudaFuncAttributes flush_kernel_attrs;
		cudaFuncGetAttributes(&flush_kernel_attrs, FlushKernel<void>);
		kernel_ptx_version = flush_kernel_attrs.ptxVersion * 10;
	}
};


/**
 * Encapsulation of kernel properties for a combination of {device, CTA size}
 */
struct KernelProperties
{
	cudaFuncAttributes 				kernel_attrs;			// CUDA kernel attributes
	CudaProperties 					cuda_props;				// CUDA device properties for the target device
	int 							max_cta_occupancy;		// Maximum CTA occupancy per SM for the target device

	/**
	 * Constructor
	 */
	template <typename KernelPtr>
	KernelProperties(
		KernelPtr Kernel,						// Kernel function pointer
		int cta_threads,						// Number of threads per CTA
		CudaProperties cuda_props) :		// CUDA properties for a specific device
			cuda_props(cuda_props)
	{
		Perror(cudaFuncGetAttributes(&kernel_attrs, Kernel), "cudaFuncGetAttributes failed", __FILE__, __LINE__);

		int max_block_occupancy = B40C_SM_CTAS(cuda_props.device_sm_version);
		int max_thread_occupancy = B40C_SM_THREADS(cuda_props.device_sm_version) / cta_threads;
		int max_smem_occupancy = (kernel_attrs.sharedSizeBytes > 0) ?
				(B40C_SMEM_BYTES(cuda_props.device_sm_version) / kernel_attrs.sharedSizeBytes) :
				max_block_occupancy;
		int max_reg_occupancy = B40C_SM_REGISTERS(cuda_props.device_sm_version) / (kernel_attrs.numRegs * cta_threads);

		max_cta_occupancy = B40C_MIN(
			B40C_MIN(max_block_occupancy, max_thread_occupancy),
			B40C_MIN(max_smem_occupancy, max_reg_occupancy));
	}

	/**
	 * Return dynamic padding to reduce occupancy to a multiple of the specified base_occupancy
	 */
	int SmemPadding(int base_occupancy)
	{
		div_t div_result = div(max_cta_occupancy, base_occupancy);
		if ((!div_result.quot) || (!div_result.rem)) {
			return 0;													// Perfect division (or cannot be padded)
		}

		int target_occupancy = div_result.quot * base_occupancy;
		int required_shared = B40C_SMEM_BYTES(cuda_props.device_sm_version) / target_occupancy;
		int padding = (required_shared - kernel_attrs.sharedSizeBytes) / 128 * 128;					// Round down to nearest 128B

		return padding;
	}
};



} // namespace cub

