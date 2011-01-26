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
 * CUDA Properties
 ******************************************************************************/

#pragma once

#include <cuda.h>

namespace b40c {

// CUDA architecture currently being compiled for
#ifndef __CUDA_ARCH__
	#define __B40C_CUDA_ARCH__ 0						// Host
#else
	#define __B40C_CUDA_ARCH__ __CUDA_ARCH__
#endif

// Thread per warp. (The CUDA Toolkit gives us warp-size, but not the log of it, which is also useful)
#define B40C_LOG_WARP_THREADS(arch)		(5)			// 32 threads in a warp 
#define B40C_WARP_THREADS(arch)			(1 << B40C_LOG_WARP_THREADS(arch))

// Memory banks per SM
#define B40C_SM20_LOG_MEM_BANKS()		(5)			// 32 banks on SM2.0+
#define B40C_SM10_LOG_MEM_BANKS()		(4)			// 16 banks on SM1.0-SM1.3
#define B40C_LOG_MEM_BANKS(arch)		((arch >= 200) ? B40C_SM20_LOG_MEM_BANKS() : 	\
															 B40C_SM10_LOG_MEM_BANKS())		

// Physical shared memory per SM (bytes)
#define B40C_SM20_SMEM_BYTES()			(49152)		// 48KB on SM2.0+
#define B40C_SM10_SMEM_BYTES()			(16384)		// 32KB on SM1.0-SM1.3
#define B40C_SMEM_BYTES(arch)			((arch >= 200) ? B40C_SM20_SMEM_BYTES() : 	\
														 B40C_SM10_SMEM_BYTES())		

// Physical threads per SM (bytes)
#define B40C_SM20_SM_THREADS()			(1536)		// 1536 threads on SM2.0+
#define B40C_SM12_SM_THREADS()			(1024)		// 1024 threads on SM1.2-SM1.3
#define B40C_SM10_SM_THREADS()			(768)		// 768 threads on SM1.0-SM1.1
#define B40C_SM_THREADS(arch)			((arch >= 200) ? B40C_SM20_SMEM_BYTES() : 	\
										 (arch >= 200) ? B40C_SM12_SMEM_BYTES() : 	\
														 B40C_SM10_SMEM_BYTES())		

// Register modifier for pointer-types (for inlining PTX assembly)
#if defined(_WIN64) || defined(__LP64__)
	#define _B40C_LP64_ true			
	// 64-bit register modifier for inlined asm
	#define _B40C_ASM_PTR_ "l"
#else
	#define _B40C_LP64_ false
	// 32-bit register modifier for inlined asm
	#define _B40C_ASM_PTR_ "r"
#endif


/**
 * Empty Kernel
 */
template <typename T>
__global__ void FlushKernel(void) { }


/**
 * Class encapsulating device properties for dynamic host-side inspection
 */
class CudaProperties 
{
public:
	
	// Information about our target device
	cudaDeviceProp 		device_props;
	int 				device_sm_version;
	
	// Information about our kernel assembly
	int 				kernel_ptx_version;
	
public:
	
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
};


} // namespace b40c

