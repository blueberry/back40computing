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

#include "../../cub/cub.cuh"
#include "../ns_wrapper.cuh"

#include "sort_utils.cuh"
#include "cta_hybrid_pass.cuh"


BACK40_NS_PREFIX
namespace back40 {
namespace radix_sort {



/**
 * Kernel entry point
 */
template <
	typename 	CtaHybridPolicy,
	int 		MIN_CTA_OCCUPANCY,
	typename 	SizeT,
	typename 	KeyType,
	typename 	ValueType>
__launch_bounds__ (
	CtaHybridPolicy::CTA_THREADS,
	MIN_CTA_OCCUPANCY)
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
	typedef CtaHybridPass<
		CtaHybridPolicy,
		KeyType,
		ValueType,
		SizeT> CtaHybridPassT;

	// Shared data structures
	__shared__ typename CtaHybridPassT::SmemStorage 	smem_storage;
	__shared__ BinDescriptor 							input_bin;

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

	CtaHybridPassT::Sort(
		smem_storage,
		d_keys_in,
		d_keys_out,
		d_keys_final,
		d_values_in,
		d_values_out,
		d_values_final,
		input_bin.current_bit,
		low_bit,
		input_bin.num_elements);
}


/**
 * Hybrid kernel props
 */
template <
	typename KeyType,
	typename ValueType,
	typename SizeT>
struct HybridKernelProps : cub::KernelProps
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
	cudaSharedMemConfig 		sm_bank_config;

	/**
	 * Initializer
	 */
	template <
		typename CtaHybridPassPolicy,
		typename OpaqueCtaHybridPassPolicy,
		int MIN_CTA_OCCUPANCY>
	cudaError_t Init(const cub::CudaProps &cuda_props)	// CUDA properties for a specific device
	{
		// Initialize fields
		kernel_func 			= HybridKernel<OpaqueCtaHybridPassPolicy, MIN_CTA_OCCUPANCY, SizeT>;
		sm_bank_config 			= CtaHybridPassPolicy::SMEM_CONFIG;

		// Initialize super class
		return cub::KernelProps::Init(
			kernel_func,
			CtaHybridPassPolicy::CTA_THREADS,
			cuda_props);
	}

	/**
	 * Initializer
	 */
	template <typename CtaHybridPassPolicy>
	cudaError_t Init(const cub::CudaProps &cuda_props)	// CUDA properties for a specific device
	{
		return Init<CtaHybridPassPolicy, CtaHybridPassPolicy>(cuda_props);
	}

};


} // namespace radix_sort
} // namespace back40
BACK40_NS_POSTFIX
