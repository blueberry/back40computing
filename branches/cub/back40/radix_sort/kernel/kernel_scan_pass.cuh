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
 *
 ******************************************************************************/

#pragma once

#include "../../util/operators.cuh"
#include "../../util/kernel_props.cuh"
#include "../../util/ns_wrapper.cuh"

#include "../../radix_sort/cta/cta_scan_pass.cuh"

BACK40_NS_PREFIX
namespace back40 {
namespace radix_sort {
namespace kernel {


/**
 * Kernel entry point
 */
template <
	typename CtaScanPassPolicy,
	typename T,
	typename SizeT>
__launch_bounds__ (CtaScanPassPolicy::CTA_THREADS, 1)
__global__
void ScanKernel(
	T			*d_in,
	T			*d_out,
	SizeT 		spine_elements)
{
	// CTA abstraction type
	typedef CtaScanPass<CtaScanPassPolicy, T> CtaScanPass;

	// Shared data structures
	__shared__ typename CtaScanPass::SmemStorage cta_smem_storage;

	// Only CTA-0 needs to run
	if (blockIdx.x > 0) return;

	CtaScanPass::Scan(
		cta_smem_storage,
		d_in,
		d_out,
		util::Sum<T>,
		spine_elements);
}



/**
 * Spine kernel properties
 */
struct SpineKernelProps : util::KernelProps
{
	// Kernel function type
	typedef void (*KernelFunc)(SizeT*, SizeT*, int);

	// Fields
	KernelFunc 					kernel_func;
	int 						log_tile_elements;
	cudaSharedMemConfig 		sm_bank_config;

	/**
	 * Initializer
	 */
	template <
		typename KernelPolicy,
		typename OpaquePolicy>
	cudaError_t Init(int sm_arch, int sm_count)
	{
		// Initialize fields
		kernel_func 			= ScanKernel<OpaquePolicy>;
		log_tile_elements 		= KernelPolicy::LOG_TILE_ELEMENTS;
		sm_bank_config 			= KernelPolicy::SMEM_CONFIG;

		// Initialize super class
		return util::KernelProps::Init(
			kernel_func,
			KernelPolicy::CTA_THREADS,
			sm_arch,
			sm_count);
	}

	/**
	 * Initializer
	 */
	template <typename KernelPolicy>
	cudaError_t Init(int sm_arch, int sm_count)
	{
		return Init<KernelPolicy, KernelPolicy>(sm_arch, sm_count);
	}
};



} // namespace kernel
} // namespace radix_sort
} // namespace back40
BACK40_NS_POSTFIX
