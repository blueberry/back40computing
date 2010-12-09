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

#ifndef __CUDA_ARCH__
	#define __CUDA_ARCH__ 0
#endif

#define B40C_LOG_WARP_THREADS							(5)									// 32 threads in a warp.  CUDA gives us warp-size, but not the log of it.
#define B40C_WARP_THREADS								(1 << B40C_LOG_WARP_THREADS)
#define B40C_LOG_MEM_BANKS(version) 					((version >= 200) ? 5 : 4)			// 32 banks on fermi, 16 on tesla
#define B40C_MEM_BANKS(version)							(1 << B40C_LOG_MEM_BANKS(version))

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

