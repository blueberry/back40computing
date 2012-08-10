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

#include "../../util/ns_umbrella.cuh"

B40C_NS_PREFIX
namespace b40c {
namespace radix_sort {



/**
 * Kernel entry point
 */
template <
	typename CtaHybridPolicy,
	typename SizeT,
	typename KeyType,
	typename ValueType>
__launch_bounds__ (CtaHybridPolicy::CTA_THREADS, CtaHybridPolicy::MIN_CTA_OCCUPANCY)
__global__
void Kernel(
	BinDescriptor						*d_bins_in,
	BinDescriptor						*d_bins_out,
	KeyType 							*d_keys_in,
	KeyType 							*d_keys_out,
	KeyType 							*d_keys_final,
	ValueType 							*d_values_in,
	ValueType 							*d_values_out,
	ValueType 							*d_values_final,
	int									low_bit)
{
	// CTA abstraction type
	typedef CtaHybrid<CtaHybridPolicy, SizeT, KeyType, ValueType> CtaHybrid;

	// Shared memory pool
	__shared__ typename CtaHybrid::SmemStorage smem_storage;

	CtaHybrid::ProcessWorkRange(
		smem_storage,
		d_bins_in,
		d_bins_out,
		d_keys_in,
		d_keys_out,
		d_keys_final,
		d_values_in,
		d_values_out,
		d_values_final,
		low_bit);
}


/**
 * Hybrid kernel props
 */
struct BinDescriptorKernelProps : util::KernelProps
{
	// Kernel function type
	typedef void (*KernelFunc)(
		Hybrid*,
		Hybrid*,
		KeyType*,
		KeyType*,
		KeyType*,
		ValueType*,
		ValueType*,
		ValueType*,
		int);

	// Fields
	KernelFunc 					kernel_func;
	int 						tile_elements;
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
		kernel_func 			= block::Kernel<OpaquePolicy>;
		tile_elements 			= KernelPolicy::TILE_ELEMENTS;
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

} // namespace radix_sort
} // namespace b40c
B40C_NS_POSTFIX
