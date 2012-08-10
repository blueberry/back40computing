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
	typename CtaDownsweepPassPolicy,
	typename SizeT,
	typename KeyType,
	typename ValueType>
__launch_bounds__ (
	CtaDownsweepPassPolicy::CTA_THREADS,
	CtaDownsweepPassPolicy::MIN_CTA_OCCUPANCY)
__global__ void DownsweepKernel(
	SizeT 								*d_spine,
	KeyType 							*d_in_keys,
	KeyType 							*d_out_keys,
	ValueType 							*d_in_values,
	ValueType 							*d_out_values,
	unsigned int 						current_bit,
	util::CtaWorkDistribution<SizeT> 	cta_work_distribution)
{
	// CTA abstraction type
	typedef CtaDownsweepPass<CtaDownsweepPassPolicy, SizeT, KeyType, ValueType> CtaDownsweepPass;

	// Shared memory
	__shared__ typename CtaDownsweepPass::SmemStorage 		cta_smem_storage;
	__shared__ util::CtaProgress<SizeT, TILE_ELEMENTS> 		cta_progress;

	// Determine our CTA's work range
	if (threadIdx.x == 0)
	{
		cta_progress.Init(cta_work_distribution);
	}

	// Read exclusive bin prefixes
	SizeT bin_prefix;
	if (threadIdx.x < CtaDownsweepPassPolicy::RADIX_DIGITS)
	{
		int spine_offset = (gridDim.x * threadIdx.x) + blockIdx.x;
		bin_prefix = d_spine[spine_digit_offset];

		if (blockIdx.x == 0)
		{
			SizeT next_spine_offset = d_spine[spine_offset + gridDim.x];
			SizeT elements = next_spine_offset - bin_prefix;

			BinDescriptor bin(bin_prefix, elements, current_bit);
			d_bins_out[threadIdx.x] = bin;
/*
			printf("Created partition %d (bit %d) of %d elements at offset %d\n",
				threadIdx.x,
				partition.current_bit,
				partition.num_elements,
				partition.offset);
*/
		}
	}

	// Sync to acquire work range
	__syncthreads();

	// Process downsweep
	CtaDownsweepPass::Downsweep(
		cta_smem_storage,
		d_in_keys + cta_progress.cta_offset,
		d_out_keys,
		d_in_values + cta_progress.cta_offset,
		d_out_values,
		current_bit,
		bin_prefixes,
		cta_progress.num_elements);
}


/**
 * Downsweep kernel props
 */
struct DownsweepKernelProps : util::KernelProps
{
	// Kernel function type
	typedef void (*KernelFunc)(
		BinDescriptor*,
		SizeT*,
		KeyType*,
		KeyType*,
		ValueType*,
		ValueType*,
		util::CtaWorkDistribution<SizeT>,
		unsigned int);

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
		kernel_func 			= downsweep::Kernel<OpaquePolicy>;
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
