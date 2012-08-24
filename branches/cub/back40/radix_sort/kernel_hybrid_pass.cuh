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

#include "../../util/kernel_props.cuh"
#include "../../util/ns_wrapper.cuh"

#include "../../radix_sort/sort_utils.cuh"
#include "../../radix_sort/cta/cta_hybrid_pass.cuh"


BACK40_NS_PREFIX
namespace back40 {
namespace radix_sort {
namespace kernel {



/**
 * Kernel entry point
 */
template <
	typename CtaHybridPolicy,
	typename SizeT,
	typename KeyType,
	typename ValueType>
__launch_bounds__ (
	CtaHybridPolicy::CTA_THREADS,
	CtaHybridPolicy::MIN_CTA_OCCUPANCY)
__global__
void HybridKernel(
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
	typedef CtaHybridPass<CtaHybridPolicy, SizeT, KeyType, ValueType> CtaHybridPass;

	// Shared data structures
	__shared__ typename CtaHybridPass::SmemStorage 	cta_smem_storage;
	__shared__ BinDescriptor 						input_bin;

	// Retrieve work
	if (threadIdx.x == 0)
	{
		input_bin = d_bins_in[blockIdx.x];
/*
		printf("\tCTA %d loaded partition (low bit %d, current bit %d) of %d elements at offset %d\n",
			blockIdx.x,
			low_bit,
			input_bin.current_bit,
			input_bin.num_elements,
			input_bin.offset);
*/

		// Reset current partition descriptor
		d_bins_in[blockIdx.x].num_elements = 0;
	}

	__syncthreads();

	// Quit if there is no work
	if (input_bin.num_elements == 0) return;

	CtaHybridPass::Sort(
		cta_smem_storage,
		d_keys_in,
		d_keys_out,
		d_keys_final,
		d_values_in,
		d_values_out,
		d_values_final,
		input_bin.
		low_bit);
}


/**
 * Hybrid kernel props
 */
struct HybridKernelProps : util::KernelProps
{
	// Kernel function type
	typedef void (*KernelFunc)(
		BinDescriptor*,
		BinDescriptor*,
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
		typename CtaHybridPassPolicy,
		typename OpaqueCtaHybridPassPolicy>
	cudaError_t Init(int sm_arch, int sm_count)
	{
		// Initialize fields
		kernel_func 			= HybridKernel<OpaqueCtaHybridPassPolicy>;
		tile_elements 			= CtaHybridPassPolicy::TILE_ITEMS;
		sm_bank_config 			= CtaHybridPassPolicy::SMEM_CONFIG;

		// Initialize super class
		return util::KernelProps::Init(
			kernel_func,
			CtaHybridPassPolicy::CTA_THREADS,
			sm_arch,
			sm_count);
	}

	/**
	 * Initializer
	 */
	template <typename CtaHybridPassPolicy>
	cudaError_t Init(int sm_arch, int sm_count)
	{
		return Init<CtaHybridPassPolicy, CtaHybridPassPolicy>(sm_arch, sm_count);
	}

};

} // namespace kernel
} // namespace radix_sort
} // namespace back40
BACK40_NS_POSTFIX
